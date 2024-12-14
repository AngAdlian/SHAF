from pykinect2 import PyKinectV2
from pykinect2.PyKinectV2 import *
from pykinect2 import PyKinectRuntime

import ctypes
import _ctypes
import sys
import numpy as np
# import cv2
import os
import math


class KinectRuntime(object):
    def __init__(self):
        self._done = False
        self._kinect = PyKinectRuntime.PyKinectRuntime(
            PyKinectV2.FrameSourceTypes_Color | PyKinectV2.FrameSourceTypes_Infrared | PyKinectV2.FrameSourceTypes_Body)
        self._bodies = None

    def Pitch(self, quaternion):
        value1 = 2.0 * (quaternion.w * quaternion.x + quaternion.y * quaternion.z)
        value2 = 1.0 - 2.0 * (quaternion.x * quaternion.x + quaternion.y * quaternion.y)
        roll = math.atan2(value1, value2)
        return roll * (180.0 / math.pi)

    def Yaw(self, quaternion):
        value = 2.0 * (quaternion.w * quaternion.y - quaternion.z * quaternion.x)
        if value > 1.0:
            value = 1.0
        elif value < -1.0:
            value = -1.0
        pitch = math.asin(value)
        return pitch * (180.0 / math.pi)

    def Roll(self, quaternion):
        value1 = 2.0 * (quaternion.w * quaternion.z + quaternion.x * quaternion.y)
        value2 = 1.0 - 2.0 * (quaternion.y * quaternion.y + quaternion.z * quaternion.z)
        yaw = math.atan2(value1, value2)
        return yaw * (180.0 / math.pi)

    def draw_body(self, joints, orientation):
        nums = 0
        joints_data = []
        while nums < PyKinectV2.JointType_Count - 4:
            x = joints[nums].Position.x
            y = joints[nums].Position.y
            z = joints[nums].Position.z
            print("x: ", x, " y: ", y, " z: ", z)

            rx = self.Pitch(orientation[nums].Orientation)
            ry = self.Yaw(orientation[nums].Orientation)
            rz = self.Roll(orientation[nums].Orientation)

            print("rx: ", rx, " ry: ", ry, " rz: ", rz)
            joints_data.append((x, y, z))
            nums += 1
        return joints_data

    def run(self):
        # -------- Main Program Loop -----------
        print("Main Program Loop")
        with open('../../../data/body_joints.csv', 'w') as f:
            f.close()

        while not self._done:

            # --- Getting frames and body joints
            if self._kinect.has_new_body_frame():
                self._bodies = self._kinect.get_last_body_frame()

            if self._bodies is not None:
                # if self._kinect.has_new_depth_frame():
                for i in range(0, self._kinect.max_body_count):
                    body = self._bodies.bodies[i]
                    print("body.is_tracked: ", body.is_tracked)
                    if not body.is_tracked:
                        continue
                    joints = body.joints
                    orientation = body.joint_orientations
                    # convert joint coordinates to color space

                    # joint_points = self._kinect.body_joints_to_color_space(joints)
                    # depth_points = self._kinect.body_joints_to_depth_space(joints)

                    XYZ_points = self.draw_body(joints, orientation)
                    # print(XYZ_points)

                    with open('../../../data/body_joints.csv', 'a') as f:
                        f.write(','.join([str(x) + ' ' + str(y) + ' ' + str(z) for x, y, z in XYZ_points]) + '\n')

        self._kinect.close()


if __name__ == "__main__":
    if not os.path.exists('data'):
        os.makedirs('data')
    game = KinectRuntime();
    game.run();