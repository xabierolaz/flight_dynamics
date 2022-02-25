#!/usr/bin/env python

from time import time
import rospy
import numpy as np
from gazebo_msgs.srv import ApplyBodyWrench, BodyRequest
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Wrench, Point, Vector3
from scipy.spatial.transform import Rotation
from std_msgs.msg import Float64MultiArray
from table_read import coefficients
from gazebo_msgs.srv import SetModelState
from geometry_msgs.msg import Pose
from gazebo_msgs.msg import ModelStates, ModelState

rospy.wait_for_service('/gazebo/apply_body_wrench')
rospy.init_node('drone_wind_python_script')
apply_body_wrench = rospy.ServiceProxy('/gazebo/apply_body_wrench', ApplyBodyWrench)
clear_body_wrench = rospy.ServiceProxy('/gazebo/clear_body_wrenches', BodyRequest)
# AIR DENSITY
RHO = 1.2
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


class aero_body:
    def __init__(self):
        """Reads the table of parameters
        """
        path = 'wind_table_v2.ods'
        self.coef = coefficients(path)
        self.rho = 1.225

        # Area and Chord are being kept unitary. The area is confirmed to be unitary, what is the reference chord value?
        self.area = 1
        self.chord = 1

        self.cp = np.array([0, 0.01, 0.02])

    def get_forces(self, coefficients, velocity):
        """
        Calculates the forces applied to the quadcopter based on the velocity of the wind and 
        the aerodynamics coefficients.

        Parameters
        ----------
        coefficients : dict
            dictionary with the CL, CD and CM coefficients
        velocity : list
            [X, Y, Z] relative velocity of the wind

        Returns
        -------
        List
            returns the lift, drag and momentum forces
        """
        velocity = np.linalg.norm(np.array(velocity))
        lift = 1 / 2 * self.rho * self.area * velocity * velocity * coefficients['C_L']
        drag = 1 / 2 * self.rho * self.area * velocity * velocity * coefficients['C_D']
        momentum_yaw = 1 / 2 * self.rho * self.area * self.chord * velocity * velocity * coefficients['C_yaw_w']
        momentum_pitch = 1 / 2 * self.rho * self.area * self.chord * velocity * velocity * coefficients['C_pitch_w']
        momentum_roll = 1 / 2 * self.rho * self.area * self.chord * velocity * velocity * coefficients['C_roll_w']
        lateral = 1 / 2 * self.rho * self.area * velocity * velocity * coefficients['C_lateral_w']
        return lift, drag, momentum_pitch, momentum_roll, momentum_yaw, lateral

    def calc_forces(self, velocity, quad_rotation, wind_rotation, position):
        """
        Calculates the forces applied to the quadcopter based on the velocity of the 
        relative wind and the attitude of the quadcopter

        Parameters
        ----------
        velocity : list
            [X, Y, Z] relative velocity of the wind

        quad_rotation : scipy rotation
            the rotation object of the quadcopter
        wind_rotation : scipy rotation
            the rotation object of the wind

        Returns
        -------
        np array [3], np array [3]
            returns the computed forces and momentums
        """
        if np.linalg.norm(velocity) <= 0.01:
            return np.zeros(3), np.zeros(3)

        quad_direction_rotation = Rotation.from_euler('z', 0, degrees=True)
        Rot_quad_wind = quad_direction_rotation.as_matrix() @ (wind_rotation.as_matrix().T @ quad_rotation.as_matrix())
        att_quad_wind = Rotation.from_matrix(Rot_quad_wind)
        euler_quad_wind = att_quad_wind.as_euler('xyz', degrees=True)


        wind_forward = velocity/np.linalg.norm(velocity)
        quad_plane = quad_rotation.as_matrix() @ np.array([0, 0, 1])
        wind_sideways = np.cross(wind_forward, quad_plane)
        wind_upward = np.cross(wind_sideways, wind_forward)

        drag_direction = -velocity
        drag_direction *= 1/np.linalg.norm(drag_direction)

        lift_direction = wind_upward
        lift_direction *= 1/np.linalg.norm(lift_direction)

        lateral_direction = np.cross(drag_direction, lift_direction)

        moment_x_direction = wind_sideways

        moment_z_direction = wind_upward

        moment_y_direction = wind_forward

        relative_att = euler_quad_wind
        coef_i = self.coef.get_coefficients(relative_att)
        norm_velocity = np.linalg.norm(velocity)
        lift, drag, momentum_x, momentum_y, momentum_z, lateral = self.get_forces(coefficients=coef_i, 
                                                velocity=norm_velocity)
        lift = lift * lift_direction
        drag = drag * drag_direction
        lateral = lateral * lateral_direction

       
        momentum = wind_rotation.as_matrix() @ np.array([momentum_x, -momentum_y, -momentum_z])
        # momentum =  np.array([momentum_x, momentum_y, momentum_z])

        force = lift + drag + lateral
        force_momentum = - np.cross(force, quad_rotation.as_matrix() @ self.cp)

        momentum = momentum + force_momentum

        #print('-'*50) 
        #print(euler_quad_wind)
        #print(moment_x_direction, moment_y_direction, moment_z_direction)
        # print(velocity)
        # print(lift_direction, drag_direction)
        # print(relative_att, coef_i)
        # print(moment_x_direction, moment_y_direction, moment_z_direction, drag_direction, lift_direction)
        print(wind_rotation.as_euler('xyz', degrees=True), position, quad_rotation.as_euler('xyz', degrees=True), self.get_forces(coefficients=coef_i, velocity=norm_velocity))
        #print(momentum_x, momentum_y, momentum_z)
        return force, momentum


class wind():
    """
        Generates random wind based on some parameters.
    """     
    def __init__(self):
        self.frequency = 50.0
        self.rate = rospy.Rate(self.frequency)
        self.sub = rospy.Subscriber("/gazebo/model_states", ModelStates, self.location_callback)
        self.quad_velocity = np.zeros(3)
        self.quad_position = np.zeros(3)
        self.quad_matrix = np.eye(3)
        self.quad_rotation = Rotation.from_matrix(np.eye(3))
        self.quad_euler = np.zeros(3)
        self.strength_modifier = 15  # max speed in m/s
        self.quad = aero_body()
        self.wind_name = ['North', 'East', 'South', 'West', 'North']
        self.name = 'marvin'

        # Debugging
        self.fixed = True
        self.fixed_wind = 10
        self.fixed_direction = 0
        # Wind at 0 comes from the north (Y), from the positive direction to the negative direction
        # The first 90 degrees rotation makes the wind come from the west (X), from the negative to the positive wind_direction

        #Reset conditions
        self.max_distance = 50 #Meters
        self.max_velocity = 30 #Meters per second
        self.max_attitude = 100 #Degrees

        self.reset_command = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)

    def reset(self):
        distance = np.linalg.norm(self.quad_position)
        velocity = np.linalg.norm(self.quad_velocity)
        attitude = np.linalg.norm(self.quad_euler[0:2])
        #print(attitude)

        conditions = [
            distance > self.max_distance,
            velocity > self.max_velocity,
            attitude > self.max_attitude
        ]
        if any(conditions):
            self.reset_service()

    def reset_service(self):
        command = ModelState()
        command.model_name = self.name
        location = Pose()
        location.position.x = 0
        location.position.y = 0
        location.position.z = 0.3

        location.orientation.x = 0
        location.orientation.y = 0
        location.orientation.z = 0
        location.orientation.w = 1

        command.pose = location
        self.reset_command(command)



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
        self.quad_euler = rotation.as_euler('xyz', degrees=True)
        self.quad_matrix = rotation.as_matrix()
        self.quad_rotation = rotation

    def run_wind(self, reset = False):
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

        if not self.fixed:
            self.main_direction = np.random.uniform(-np.pi, np.pi)
        else:
            self.main_direction = np.deg2rad(self.fixed_direction)

        cardinal_wind = self.calc_wind_direction(np.rad2deg(self.main_direction+np.pi))
        print('Blowing with {:.2f} m/s ({:.2f} km/h) in {} direction for {:.2f} seconds'.format(self.strength * self.strength_modifier,
                                                                                velocity_kmh, cardinal_wind, self.duration))
        while diff_time < self.duration:
            if not self.fixed:
                self.instant_direction = np.random.uniform(self.main_direction - instant_angle,
                                                        self.main_direction + instant_angle)

                wind_rotation = Rotation.from_euler('z', self.instant_direction)
                
                wind_vector = wind_rotation.as_matrix() @ np.array([0, self.strength * self.strength_modifier, 0])
            else:
                wind_rotation = Rotation.from_euler('z', self.main_direction)

                wind_vector = wind_rotation.as_matrix() @ np.array([0, self.fixed_wind, 0])

            diff_time = rospy.get_time() - self.time_init
            total_half_time = float(self.duration) / 2.0
            if self.fixed:
                time_modifier = 1
            else:
                if diff_time < (self.duration / 2.0):
                    time_modifier = diff_time / total_half_time
                else:
                    time_modifier = 1 - (diff_time - total_half_time) / total_half_time
            
            relative_wind = wind_vector * time_modifier + self.quad_velocity


            force, momentum = self.quad.calc_forces(velocity=relative_wind,
                                                    quad_rotation=self.quad_rotation,
                                                    wind_rotation=wind_rotation, position = self.quad_position)
            apply_body_wrench_client(self.quad_position, force, momentum, 1.0 / self.frequency)
            self.rate.sleep()
            if reset:
                self.reset()
        # random_rest = np.random.uniform(0, 20)
        # print("Resting for {:.2f} seconds".format(random_rest))
        # rospy.sleep(random_rest)


if __name__ == "__main__":
    w = wind()
    while True:
        w.run_wind(reset=True)
