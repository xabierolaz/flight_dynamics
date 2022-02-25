#!/usr/bin/env python

import rospy
import numpy as np
from gazebo_msgs.srv import ApplyBodyWrench, BodyRequest
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Wrench, Point, Vector3
from scipy.spatial.transform import Rotation
from std_msgs.msg import Float64MultiArray


# Setting up services that will be used
rospy.wait_for_service('/gazebo/apply_body_wrench')
rospy.init_node('drone_wind_python_script')
apply_body_wrench = rospy.ServiceProxy('/gazebo/apply_body_wrench', ApplyBodyWrench)
clear_body_wrench = rospy.ServiceProxy('/gazebo/clear_body_wrenches', BodyRequest)

# AIR DENSITY
RHO = 1.225
# DRAG COEFFICIENT
C_D = 1.1
# ELETRIC MOTOR DISTANCE TO CG
D = 0.26
# PROJECTED AREA
BEAM_THICKNESS = 0.05
A = BEAM_THICKNESS * 2.0 * D


def apply_body_wrench_client(quad_position, force, momentum, duration):
    """
    Applies a force to the quadcopter

    Parameters
    ----------
    quad_position : List
        [X, Y, Z] position of the quadcopter
    force : List
        [FX, FY, FZ] forces to be applied to the quadcopter, in the inertial direction
    momentum : List
        [MX, MY, MZ] momentum to be applied to the quadcopter, in the inertial direction
    duration : float
        Duration of time in seconds that the force and momentum is applied
    """    
    body_name = 'body'
    reference_frame = 'world'
    fx = float(force[0])
    fy = float(force[1])
    fz = float(force[2])
    mx = float(momentum[0])
    my = float(momentum[1])
    mz = float(momentum[2])
    reference_point = Point(x=0, y=0, z=0)
    wrench = Wrench(force=Vector3(x=fx, y=fy, z=fz), torque=Vector3(x=mx, y=my, z=mz))
    # print(fx, fy, fz, mx, my, mz)
    duration = rospy.Duration.from_sec(duration)
    apply_body_wrench(body_name, reference_frame, reference_point, wrench, rospy.get_rostime(), duration)


class aero_obj:
    """
    Describes an aerodynamic object, simplified as its coefficients, center of pressure, area and direction
    """    
    def __init__(self, cd, cl, cm, alpha_0,
                 alpha_stall, cl_stall, cd_stall, cm_stall,
                 cp, area, forward=[0, 1, 0], upward=[0, 0, 1], name=None):
        """
        Initializes all the necessary parameters for the object

        Parameters
        ----------
        cd : Float
            Coefficient of Drag C_D_Alpha
        cl : Float
            Coefficient of Lift C_L_Alpha
        cm : Float
            Coefficient of Momentum C_M_Alpha
        alpha_0 : Float
            Neutral angle of attack
        alpha_stall : Float
            Stall angle of attack
        cl_stall : Float
            Coefficient of Lift C_L_Alpha on the stall area
        cd_stall : Float
            Coefficient of Drag C_D_Alpha on the stall area
        cm_stall : Float
            Coefficient of Momentum C_M_Alpha on the stall area
        cp : List
            [X, Y, Z] center of pressure of the aerodynamic object
        area : Float
            Area of the aerodynamic object
        forward : list, optional
            forward direction (flight direction) in a vectorized form, based on the body coordinate system, by default [0, 1, 0]
        upward : list, optional
            upward direction (lift direction) in a vectorized form, based on the body coordinate system, by default [0, 0, 1]
        name : str, optional
            Name of the aerodynamic object, by default None
        """        
        self.name = name
        self.cd = cd
        self.cl = cl
        self.cm = cm
        self.alpha_0 = alpha_0
        self.alpha_stall = alpha_stall
        self.cd_stall = cd_stall
        self.cl_stall = cl_stall
        self.cm_stall = cm_stall
        self.cp = np.array(cp)
        self.area = area
        self.rho = 1.225
        self.forward = np.array(forward)
        self.upward = np.array(upward)
        self.perpendicular = np.cross(forward, upward)
        self.body_matrix = np.array([self.forward, self.perpendicular, self.upward])
        # print(self.name, self.forward, self.upward)

    def static_pressure(self, velocity):
        """
        Computes the static pressure based on the velocity

        Parameters
        ----------
        velocity : list
            [X, Y, Z] velocity of the quadcopter in the inertial plane of coordinates

        Returns
        -------
        numpy array
            [qx, qy, qz] the static pressure being applied
        """        
        velocity = np.array(velocity)
        q = 1 / 2 * self.rho * velocity * velocity
        return q

    def calc_force(self, velocity, rotation_matrix):
        """
        Calculates the forces and momentum being applied to the aerodynamic object

        Parameters
        ----------
        velocity : list [3] or numpy array
            [X, Y, Z] velocity of the aerodynamic object
        rotation_matrix : [3X3] numpy array
            Rotation matrix of the aerodynamic object in relation to the inertial frame of coordinates

        Returns
        -------
        numpy array [3], numpy array [3]
            returns the computed forces and the momentum as [FX, FY, FZ] and [MX, MY, MZ]
        """        

        # Uncomment this line if you want a fixed wind for debugging purposes
        # velocity = np.array([25, 25, 0])
        if np.linalg.norm(velocity) <= 0.01:
            return np.zeros(3), np.zeros(3)

        inertial_forward = rotation_matrix @ self.forward
        inertial_upward = rotation_matrix @ self.upward
        lift_drag_plane = np.cross(inertial_forward, inertial_upward)

        lift_drag_plane = lift_drag_plane / np.linalg.norm(lift_drag_plane)
        cosSweepAngle = np.clip(np.dot(lift_drag_plane, velocity), -1, 1)
        cosSweepAngle2 = 1
        sweep_angle = np.arccos(cosSweepAngle)

        if sweep_angle > 0.5 * np.pi:
            sweep_angle = sweep_angle - np.pi
        elif sweep_angle < -0.5 * np.pi:
            sweep_angle = sweep_angle + np.pi

        velocity_plane = np.cross(lift_drag_plane, np.cross(velocity, lift_drag_plane))
        if np.linalg.norm(velocity_plane) <= 0.01:
            return np.zeros(3), np.zeros(3)

        drag_direction = (-velocity_plane)
        drag_direction *= 1/np.linalg.norm(drag_direction)

        lift_direction = np.cross(lift_drag_plane, velocity_plane)
        lift_direction *= 1/np.linalg.norm(lift_direction)

        moment_direction = lift_drag_plane
        moment_direction *= 1/np.linalg.norm(moment_direction)

        cos_alpha = np.clip(np.dot(inertial_forward, velocity_plane)/
                            (np.linalg.norm(inertial_forward)*np.linalg.norm(velocity_plane)), -1, 1)

        alpha_sign = -np.dot(inertial_upward, velocity_plane) / \
                     (np.linalg.norm(inertial_upward) + np.linalg.norm(velocity_plane))

        if alpha_sign > 0:
            alpha = self.alpha_0 + np.arccos(cos_alpha)
        else:
            alpha = self.alpha_0 - np.arccos(cos_alpha)

        cm_modifier = 1
        if alpha > 0.5 * np.pi:
            alpha = alpha - np.pi
            cm_modifier = -1
        elif alpha < -0.5 * np.pi:
            alpha = alpha + np.pi
            cm_modifier = -1

        if alpha > self.alpha_stall:
            cl = np.clip(self.cl * self.alpha_stall + self.cl_stall * (alpha - self.alpha_stall), 0, 20)
            cd = np.clip(self.cd * self.alpha_stall + self.cd_stall * (alpha - self.alpha_stall), 0, 20)
            cm = np.clip(self.cm * self.alpha_stall + self.cm_stall * (alpha - self.alpha_stall), 0, 20)
        elif alpha < - self.alpha_stall:
            cl = np.clip(self.cl * -self.alpha_stall + self.cl_stall * (alpha + self.alpha_stall), -20, 0)
            cd = np.clip(self.cd * -self.alpha_stall + self.cd_stall * (alpha + self.alpha_stall), -20, 0)
            cm = np.clip(self.cm * -self.alpha_stall + self.cm_stall * (alpha + self.alpha_stall), -20, 0)
        else:
            cl = self.cl * alpha * cosSweepAngle2
            cd = self.cd * alpha * cosSweepAngle2
            cm = self.cm * alpha * cosSweepAngle2

        linear_velocity = np.linalg.norm(velocity_plane)
        static_force = self.static_pressure(linear_velocity)

        lift_force = static_force * self.area * cl * lift_direction

        drag_force = static_force * self.area * abs(cd) * drag_direction

        momentum_cp = static_force * self.area * cm * moment_direction * cm_modifier

        forces = lift_force + drag_force
        momentum = momentum_cp - np.cross(forces, rotation_matrix @ self.cp)
        # print('--' * 20 + '   ' + self.name + '   ' + '--' * 20)
        # print(lift_drag_plane, static_force * self.area * cm)
        # print(cm_modifier)
        # print(lift_force, drag_force, momentum_cp, np.cross(forces, rotation_matrix @ np.abs(self.cp)))
        # print(inertial_forward, inertial_upward, np.linalg.norm(velocity_plane))
        # print(alpha, rotation_matrix @ self.cp, moment_direction)
        # print(forces, momentum)
        # print(drag_direction, lift_direction, moment_direction)
        # print(rotation_matrix @ self.cp, np.cross(forces, rotation_matrix @ self.cp))

        return forces, momentum


class aero_body:
    """
    The junction of diverse aerodynamic objects makes one aerodynamic body
    """    
    def __init__(self):
        """
        Declaring each aerodynamic object in the aerodynamic body
        """        
        self.main_wing = aero_obj(cd=0.1, cl=0.7387, cm=0.0736, alpha_0=0.08727,
                                  alpha_stall=0.209, cl_stall=-2, cd_stall=0.2, cm_stall=0.0736,
                                  cp=[0, -0.138017, 0.009807], area=0.572, name='main_wing')

        self.main_body = aero_obj(cd=0.1, cl=3.7387, cm=0.0736, alpha_0=0.08727,
                                  alpha_stall=0.209, cl_stall=-2, cd_stall=0.2, cm_stall=0.0736,
                                  cp=[0, 0.13556, 0.00554], area=0.127, name='main_body')

        self.horizontal_tail_wing = aero_obj(cd=0.1, cl=1.269, cm=0.01177, alpha_0=0.06,
                                             alpha_stall=0.209, cl_stall=-0.150, cd_stall=0.4, cm_stall=0.01177,
                                             cp=[0, -0.9323, 0], area=0.06125, name='HTP')

        self.vertical_tail_wing = aero_obj(cd=0.1, cl=1.269, cm=0.01177, alpha_0=0.0,
                                           alpha_stall=0.209, cl_stall=-0.150, cd_stall=0.4, cm_stall=0.01177,
                                           cp=[0, -0.9323, 0.150], area=0.030625, upward=[1, 0, 0], forward=[0, 1, 0],
                                           name='VTP')

        self.body_list = [self.main_wing, self.main_body, self.horizontal_tail_wing, self.vertical_tail_wing]
        # self.body_list = [self.main_wing]

    def calc_forces(self, velocity, rotation_matrix):
        """
        Sum all the forces of each aerodynamic object

        Parameters
        ----------
        velocity : List
            Velocity of the aerodynamic body
        rotation_matrix : np array [3x3]
            The rotation matrix of the aerodynamic body from body to inertial

        Returns
        -------
        np array [3], np array [3]
            returns the sum of forces and the sum of momentums.
        """        
        force, momentum = np.zeros(3), np.zeros(3)
        for body in self.body_list:
            f, m = body.calc_force(velocity, rotation_matrix)
            force += f
            momentum += m
        # print(force, momentum)
        return force, momentum


class wind():
    def __init__(self):
        """
        Generates random wind based on some parameters.
        """        
        self.frequency = 30.0
        self.rate = rospy.Rate(self.frequency)
        self.sub = rospy.Subscriber("/gazebo/model_states", ModelStates, self.location_callback)
        self.quad_velocity = np.zeros(3)
        self.quad_position = np.zeros(3)
        self.quad_matrix = np.eye(3)
        self.quad_euler = np.zeros(3)
        self.strength_modifier = 10  # max speed in m/s
        self.quad = aero_body()
        self.wind_name = ['North', 'East', 'South', 'West', 'North']

    def calc_wind_direction(self, wind_direction):
        """
        Calculates the wind direction and prints it on screen as cardinal direction

        Parameters
        ----------
        wind_direction : float
            Yaw direction of the wind

        Returns
        -------
        str
            returns the cardinal wind direction
        """
        list_directions = [90, 180, 270, 360]
        for i, direction in enumerate(list_directions):
            if wind_direction < direction:
                if wind_direction < i*90 + 90/4:
                    wind_str = self.wind_name[i]
                    return wind_str
                elif wind_direction < i*90 + 90/2 + 90/4:
                    wind_str = self.wind_name[i]+'-'+self.wind_name[i+1].lower()
                    return wind_str
                else:
                    wind_str = self.wind_name[i+1]
                    return wind_str
        return 'n'



    def location_callback(self, msg):
        """
        Gets the location of the quadcopter with ROSPY package

        Parameters
        ----------
        msg : gazebo msg
            the gazebo message to be read
        """
        ind = msg.name.index('marvin')
        velocityObj = msg.twist[ind].linear
        positionObj = msg.pose[ind].position
        attitudeObj = msg.pose[ind].orientation
        self.quad_position = np.array([positionObj.x, positionObj.y, positionObj.z])
        self.quad_velocity = np.array([velocityObj.x, velocityObj.y, velocityObj.z])
        self.quad_quaternion = np.array([attitudeObj.x, attitudeObj.y, attitudeObj.z, attitudeObj.w])
        rotation = Rotation.from_quat(self.quad_quaternion)
        self.quad_euler = rotation.as_euler('xyz')
        self.quad_matrix = rotation.as_matrix()

    def run_wind(self):
        """
        Main point of the algorithm, runs the random wind, calculates the forces based on the aerodynamic objects,
        and applies that forces to the quadcopter.
        """
        strength = np.random.random()  # by default is 0 to 1, so normally will be 0.5
        self.strength = strength
        self.duration = 30.0 * strength  # duration of wind = strength*30 seconds
        velocity_kmh = self.strength * self.strength_modifier * 3.6
        self.time_init = rospy.get_time()
        diff_time = 0
        instant_angle = np.deg2rad(10)
        self.main_direction = np.random.uniform(-np.pi, np.pi)
        cardinal_wind = self.calc_wind_direction(np.rad2deg(self.main_direction+np.pi))
        print('Blowing with {:.2f} m/s ({:.2f} km/h) in {} direction for {:.2f} seconds'.format(self.strength * self.strength_modifier,
                                                                                velocity_kmh, cardinal_wind, self.duration))
        while diff_time < self.duration:
            self.instant_direction = np.random.uniform(self.main_direction - instant_angle,
                                                       self.main_direction + instant_angle)
            wind_rotation = Rotation.from_euler('z', self.instant_direction)
            wind_vector = np.matmul(wind_rotation.as_matrix(), np.array([self.strength * self.strength_modifier, 0, 0]))
            diff_time = rospy.get_time() - self.time_init
            total_half_time = float(self.duration) / 2.0
            if diff_time < (self.duration / 2.0):
                time_modifier = diff_time / total_half_time
            else:
                time_modifier = 1 - (diff_time - total_half_time) / total_half_time
            relative_wind = wind_vector * time_modifier - self.quad_velocity
            force, momentum = self.quad.calc_forces(velocity=relative_wind,
                                                    rotation_matrix=self.quad_matrix)

            apply_body_wrench_client(self.quad_position, force, momentum, 1.0 / self.frequency)
            self.rate.sleep()
        # random_rest = np.random.uniform(0, 20)
        # print("Resting for {:.2f} seconds".format(random_rest))
        # rospy.sleep(random_rest)


if __name__ == "__main__":
    w = wind()
    while True:
        w.run_wind()
