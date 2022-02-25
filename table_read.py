import pandas as pd
import numpy as np
from scipy.interpolate import LinearNDInterpolator


class coefficients:
    ''' 
    Gets the coefficients from the wind_table, based on the attitude of the quadrotor
    The table values are interpolated using scipy LinearNDInterpolator. 
    This interpolator can be used with any N dimensions.
    '''
    def __init__(self, path):
        '''__init__ 
        Inits the coefficients function

        Arguments:
            path {str} -- [the path to the table of values]
        '''
        self.df = pd.read_excel(path, index_col=0)
        self.df = self.correct_df(self.df)
        self.max_values = np.array([135, 180, 180])
        self.min_values = np.array([-90, -135, -180])
        self.linInter= LinearNDInterpolator(self.df[['PITCH (º)', 'ROLL (º)', 'YAW (º)']].to_numpy(), 
                                        self.df[['CD', 'CL', 'C_pitch_w', 'C_roll_w', 'C_yaw_w', 'C_lateral_w']].to_numpy())
    @staticmethod
    def correct_df(df):       
        '''correct_df corrects the dataframe, using symetry, so that missing values are generated.

        Arguments:
            df {[pandas dataframe]} -- the pandas dataframe to be corrected

        Returns:
            [pandas dataframe] -- the corrected pandas dataframe
        '''
        # print(df.head())
        # Roll is indifferent with yaw = 0
        # df_temp = df[df['YAW (º)'] == 0].copy()
        # df_temp['ROLL (º)'] = 12
        # df_temp['C_roll_w'] = -0.01
        # df = df.append(df_temp)

        # # Roll is indifferent with yaw = 90
        # df_temp = df[df['YAW (º)'] == 90].copy()
        # df_temp['ROLL (º)'] = 12
        # df_temp['C_roll_w'] = -0.01
        # df = df.append(df_temp)

        # # Roll is indifferent with yaw = 90
        # df_temp = df[df['YAW (º)'] == 180].copy()
        # df_temp['ROLL (º)'] = 12
        # df_temp['C_roll_w'] = -0.01
        # df = df.append(df_temp)

        # # Roll is Symnetric
        # df_temp = df.copy()
        # df_temp['ROLL (º)'] = -df_temp['ROLL (º)']
        # df_temp['C_roll_w'] = -df_temp['C_roll_w']
        # df = df.append(df_temp)

        # Yaw is Symnetric
        df_temp = df[df['YAW (º)'] > 0].copy()
        df_temp['YAW (º)'] = -df_temp['YAW (º)']
        df_temp['C_yaw_w'] = -df_temp['C_yaw_w']

        # Lateral force is symmetric
        df_temp['C_lateral_w'] = -df_temp['C_lateral_w']


        
        df = df.append(df_temp)
        df.to_excel('processed_wind_table.ods')
        return df

    def get_coefficients(self, desired_angle):
        '''get_coefficients Gets the coefficients based on the three euler angles of the quadcopter

        Arguments:
            desired_angle {List or numpy array} -- [The euler angles of the quadcopter, in degrees]

        Returns:
            [dict] -- [A dictionary with all the coefficients]
        '''
        desired_angle = np.array(desired_angle)
        # print(desired_angle, '----------------')
        
        # Clips the values so they are within the table region, values outside of the table region are unknown
        # Unkown values are cliped to the nearest known value
        desired = np.clip(desired_angle, self.min_values, self.max_values)
        # print(desired)
        coef_array = self.linInter(desired).flatten()
        if np.isnan(coef_array.sum()):
            coef_array = np.zeros(5)
        coef_names = ['C_D', 'C_L', 'C_pitch_w', 'C_roll_w', 'C_yaw_w', 'C_lateral_w']
        coef_dict = {x: y for x, y in zip(coef_names, coef_array)}
        return coef_dict

if __name__ == '__main__':
    
    coef = coefficients()
    for i in range(119):
        print(coef.get_coefficients([0, -9+i/6, 0]))