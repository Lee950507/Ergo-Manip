#!/usr/bin/env python3
import numpy as np
import math
import matplotlib.pyplot as plt
import message_filters

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.spatial.transform import Rotation as R
from itertools import product
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable, get_cmap
from geometry_msgs.msg import PoseStamped

import sys
import os
import rospy
import signal
import subprocess
import time

# 获取 ROS 工作空间的路径
workspace_path = '/home/clover/catkin_ws'

# 添加编译后的库路径
sys.path.append(os.path.join(workspace_path, 'devel', 'lib'))
# export PYTHONPATH=$PYTHONPATH:/home/clover/catkin_ws/devel/lib

from libpython_curi_dual_arm_ic import Python_CURI_Control

def launch_roslaunch():
    launch_file = "~/catkin_ws/src/curi_whole_body_interface/launch/python_curi_dual_arm_ic_qbhand.launch"  # 替换为你的 launch 文件路径
    # 启动 roslaunch
    command = f"roslaunch {launch_file}"
    return subprocess.Popen(command, shell=True)


def signal_handler(sig, frame):
    print('Python shutdown signal received...')
    rospy.signal_shutdown("shutdown by manual")  # 标记节点为关闭
    # 终止 roslaunch_process
    if 'roslaunch_process' in locals():
        print('Shutdown roslaunch process.')
        roslaunch_process.terminate()
        roslaunch_process.wait()  # 等待进程终止
    print('Python shutdown.')
    sys.exit(0)


if __name__ == '__main__':
    rospy.init_node('ros_node')
    signal.signal(signal.SIGINT, signal_handler)
    # 启动 roslaunch
    roslaunch_process = launch_roslaunch()
    time.sleep(1)

    curi = Python_CURI_Control(0, [])
    curi.start()

    time.sleep(1)

    initial_robot_left_pose_matrix = ...
    initial_robot_right_pose_matrix = ...
    curi.set_tcp_moveL(initial_robot_left_pose_matrix, initial_robot_right_pose_matrix)

    while curi.get_curi_mode(0) != 2 and curi.get_curi_mode(1) != 2:
        print("waiting robot external control")
        time.sleep(1)

    print("Start executing...")
    time.sleep(3)

    while True:
        robot_left_pose_matrix = ...
        robot_right_pose_matrix = ...
        curi.set_tcp_servo(robot_left_pose_matrix, robot_right_pose_matrix)



