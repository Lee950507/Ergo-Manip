#!/usr/bin/env python3
import numpy as np
import math
import matplotlib.pyplot as plt
import utils
import transformation as tsf
from iros2025_code import main_opt_static as mos
import message_filters

from itertools import product
from matplotlib.colors import Normalize
from geometry_msgs.msg import PoseStamped
from scipy.interpolate import CubicSpline
from scipy.spatial import KDTree

from EMGProcessor import EMGProcessor
import motion_planning_composite_filed_moving_base_ros as cf_plan

import sys
import os
import rospy
import signal
import subprocess
import time
import queue
import threading

# 获取 ROS 工作空间的路径
workspace_path = '/home/clover/catkin_ws'

# 添加编译后的库路径
sys.path.append(os.path.join(workspace_path, 'devel', 'lib'))
# export PYTHONPATH=$PYTHONPATH:/home/clover/catkin_ws/devel/lib

from libpython_curi_dual_arm_ic import Python_CURI_Control

last_relative_pose_wrists = None
last_object_pose = None
global ind
ind = 1


def launch_roslaunch():
    launch_file = "~/catkin_ws/src/curi_whole_body_interface/launch/python_curi_dual_arm_ic_qbhand.launch"  # 替换为你的 launch 文件路径
    # 启动 roslaunch
    command = f"roslaunch {launch_file}"
    return subprocess.Popen(command, shell=True)


def vrpn_launch_roslaunch():
    launch_file = "~/catkin_ws/src/vrpn_client_ros/launch/sample.launch"  # 替换为你的 launch 文件路径
    # 启动 roslaunch
    command = f"roslaunch {launch_file} server:=192.168.10.7"
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


def transform_to_pose(pose_stamped):
    return np.array([
        pose_stamped.pose.position.x,
        pose_stamped.pose.position.y,
        pose_stamped.pose.position.z,
        pose_stamped.pose.orientation.x,
        pose_stamped.pose.orientation.y,
        pose_stamped.pose.orientation.z,
        pose_stamped.pose.orientation.w
    ])


def compress_bounds(joint_angle_bounds, q, compression_factor=0.5):
    new_bounds = []
    joint_center = q

    for i, (lower, upper) in enumerate(joint_angle_bounds):
        range_half = (upper - lower) * compression_factor / 2
        center = joint_center[i]
        new_lower = center - range_half
        new_upper = center + range_half

        new_bounds.append((new_lower, new_upper))

    return new_bounds


def trans_shoulder2global(joint_pos, shoulder_pos, arm='right'):
    if arm == 'left':
        joint_pos[[0, 1]] = -joint_pos[[1, 0]]
        joint_pos[1] = -joint_pos[1]
        joint_pos = joint_pos + shoulder_pos
    if arm == 'right':
        joint_pos[[0, 1]] = -joint_pos[[1, 0]]
        joint_pos = joint_pos + shoulder_pos
    return joint_pos


def trans_global2shoulder(shoulder, elbow, wrist, arm='left'):
    if arm == 'left':
        elbow_new = elbow - shoulder
        elbow_new = np.array([elbow_new[1], -elbow_new[0], elbow_new[2]])
        wrist_new = wrist - shoulder
        wrist_new = np.array([wrist_new[1], -wrist_new[0], wrist_new[2]])
    if arm == 'right':
        elbow_new = elbow - shoulder
        elbow_new = np.array([-elbow_new[1], -elbow_new[0], elbow_new[2]])
        wrist_new = wrist - shoulder
        wrist_new = np.array([-wrist_new[1], -wrist_new[0], wrist_new[2]])
    return elbow_new, wrist_new


def minimum_jerk_trajectory(start, end, t_start, t_end, t_sample):
    # 时间差
    T = t_end - t_start

    # Minimum jerk 的多项式系数
    c0 = start[0]
    c1 = start[1]
    c2 = start[2] / 2.0
    c3 = (20 * (end[0] - start[0]) - (8 * end[1] + 12 * start[1]) * T - (3 * start[2] - end[2]) * T ** 2) / (2 * T ** 3)
    c4 = (-30 * (end[0] - start[0]) + (14 * end[1] + 16 * start[1]) * T + (3 * start[2] - 2 * end[2]) * T ** 2) / (
                2 * T ** 4)
    c5 = (12 * (end[0] - start[0]) - (6 * end[1] + 6 * start[1]) * T - (start[2] - end[2]) * T ** 2) / (2 * T ** 5)

    # 时间序列
    time_steps = np.arange(t_start, t_end, t_sample)
    trajectory = []
    for t in time_steps:
        dt = t - t_start
        # 位置
        position = c0 + c1 * dt + c2 * dt ** 2 + c3 * dt ** 3 + c4 * dt ** 4 + c5 * dt ** 5
        # 速度
        velocity = c1 + 2 * c2 * dt + 3 * c3 * dt ** 2 + 4 * c4 * dt ** 3 + 5 * c5 * dt ** 4
        # 加速度
        acceleration = 2 * c2 + 6 * c3 * dt + 12 * c4 * dt ** 2 + 20 * c5 * dt ** 3
        trajectory.append([position, velocity, acceleration])

    return np.array(trajectory)


# 新增：平滑轨迹函数
def smooth_trajectory(waypoints, smoothing_factor=0.8, iterations=3):
    """
    对轨迹点进行平滑处理
    waypoints: 原始轨迹点 [N, 3]
    smoothing_factor: 平滑因子 (0-1)，越大平滑效果越强
    iterations: 平滑迭代次数
    """
    if len(waypoints) <= 2:
        return waypoints

    smoothed = np.array(waypoints, copy=True)

    for _ in range(iterations):
        # 保留起点和终点
        original_start = smoothed[0].copy()
        original_end = smoothed[-1].copy()

        # 应用移动平均
        for i in range(1, len(smoothed) - 1):
            smoothed[i] = smoothed[i] * (1 - smoothing_factor) + \
                          (smoothed[i - 1] + smoothed[i + 1]) * 0.5 * smoothing_factor

        # 恢复起点和终点
        smoothed[0] = original_start
        smoothed[-1] = original_end

    return smoothed


# 替换：新的轨迹生成函数，使用三次样条插值
def generate_smooth_trajectory(waypoints, speed_limit, t_total, t_sample):
    """
    基于给定的路径点生成平滑轨迹
    waypoints: 路径点 [N, 3]
    speed_limit: 最大速度限制
    t_total: 总时间
    t_sample: 采样时间间隔
    """
    num_waypoints = len(waypoints)
    if num_waypoints < 2:
        raise ValueError("需要至少两个 waypoints 来生成轨迹")

    # 路径长度估计
    path_lengths = [0]
    total_length = 0

    for i in range(1, num_waypoints):
        segment_length = np.linalg.norm(waypoints[i] - waypoints[i - 1])
        total_length += segment_length
        path_lengths.append(total_length)

    # 基于路径长度的时间分配
    t_waypoints = np.zeros(num_waypoints)
    for i in range(1, num_waypoints):
        t_waypoints[i] = t_total * path_lengths[i] / total_length if total_length > 0 else t_total * i / (
                    num_waypoints - 1)

    # 存储最终轨迹
    full_trajectory = []

    # 使用三次样条插值来生成更平滑的轨迹
    # 为每个维度创建样条
    splines = []
    for dim in range(waypoints.shape[1]):
        spline = CubicSpline(t_waypoints, waypoints[:, dim])
        splines.append(spline)

    # 生成轨迹采样点
    t_samples = np.arange(0, t_total, t_sample)
    positions = np.zeros((len(t_samples), waypoints.shape[1]))
    velocities = np.zeros((len(t_samples), waypoints.shape[1]))
    accelerations = np.zeros((len(t_samples), waypoints.shape[1]))

    for dim, spline in enumerate(splines):
        positions[:, dim] = spline(t_samples)
        velocities[:, dim] = spline.derivative(1)(t_samples)
        accelerations[:, dim] = spline.derivative(2)(t_samples)

    # 速度限制
    speeds = np.linalg.norm(velocities, axis=1)
    max_speed = np.max(speeds) if len(speeds) > 0 else 0

    if max_speed > speed_limit and max_speed > 0:
        # 调整时间尺度来满足速度限制
        scale_factor = max_speed / speed_limit
        t_total_adjusted = t_total * scale_factor
        t_samples_adjusted = np.arange(0, t_total_adjusted, t_sample)

        # 重新计算轨迹
        positions = np.zeros((len(t_samples_adjusted), waypoints.shape[1]))
        velocities = np.zeros((len(t_samples_adjusted), waypoints.shape[1]))
        accelerations = np.zeros((len(t_samples_adjusted), waypoints.shape[1]))

        for dim, spline in enumerate(splines):
            # 调整时间尺度
            adjusted_spline = CubicSpline(t_waypoints * scale_factor, waypoints[:, dim])
            positions[:, dim] = adjusted_spline(t_samples_adjusted)
            velocities[:, dim] = adjusted_spline.derivative(1)(t_samples_adjusted) / scale_factor
            accelerations[:, dim] = adjusted_spline.derivative(2)(t_samples_adjusted) / (scale_factor ** 2)

    # 组装最终轨迹 (只返回位置，保持与原有接口兼容)
    return positions


def generate_trajectory_with_speed_limit(waypoints, speed_limit, t_total, t_sample):
    """保留原函数以兼容旧代码，但内部使用新的平滑轨迹生成算法"""
    positions = generate_smooth_trajectory(waypoints, speed_limit, t_total, t_sample)

    # 为了兼容原接口，构造包含位置、速度和加速度的数组
    # 但速度和加速度设为0，因为原代码只使用位置
    n_samples = positions.shape[0]
    full_trajectory = np.zeros((n_samples, 3))
    full_trajectory[:, 0] = positions[:, 0]  # 只使用x坐标，与原代码兼容

    return full_trajectory


latest_robot_msg = None
latest_shouR_msg = None
latest_elbowR_msg = None
latest_wristR_msg = None


def robot_callback(msg):
    global latest_robot_msg
    latest_robot_msg = msg


def shouR_callback(msg):
    global latest_shouR_msg
    latest_shouR_msg = msg


def elbowR_callback(msg):
    global latest_elbowR_msg
    latest_elbowR_msg = msg


def wristR_callback(msg):
    global latest_wristR_msg
    latest_wristR_msg = msg


def setup_subscribers():
    rospy.Subscriber('/vrpn_client_node/robot/pose', PoseStamped, robot_callback)
    rospy.Subscriber('/vrpn_client_node/shouR/pose', PoseStamped, shouR_callback)
    rospy.Subscriber('/vrpn_client_node/elbowR/pose', PoseStamped, elbowR_callback)
    rospy.Subscriber('/vrpn_client_node/wristR/pose', PoseStamped, wristR_callback)


if __name__ == '__main__':
    rospy.init_node('cf_hrc')
    signal.signal(signal.SIGINT, signal_handler)
    # 启动 roslaunch
    roslaunch_process = launch_roslaunch()
    time.sleep(1)  # 等待一段时间以确保 ROS 节点启动
    # 启动控制器

    curi = Python_CURI_Control(0, [])
    curi.start()

    time.sleep(1)  #
    vrpn_roslaunch_process = vrpn_launch_roslaunch()

    ## Initialization of robot end effector poses
    robot_left_position_init = np.array([0.85, -0.1, 1.35])
    robot_right_position_init = np.array([0.85, -0.25, 0.85])

    robot_left_rotation_matrix_init = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    robot_right_rotation_matrix_init = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])

    robot_left_pose_matrix_init = np.r_[
        np.c_[robot_left_rotation_matrix_init, robot_left_position_init.T], np.array([[0, 0, 0, 1]])]
    robot_right_pose_matrix_init = np.r_[
        np.c_[robot_right_rotation_matrix_init, robot_right_position_init.T], np.array([[0, 0, 0, 1]])]

    base2torso_matrix = np.array([[1, 0, 0, -0.29], [0, 1, 0, 0], [0, 0, 1, -0.985], [0, 0, 0, 1]])
    initial_robot_left_pose_matrix = base2torso_matrix @ robot_left_pose_matrix_init
    initial_robot_right_pose_matrix = base2torso_matrix @ robot_right_pose_matrix_init

    print("init_left_arm_pub", initial_robot_left_pose_matrix)

    print("init_right_arm_pub", initial_robot_right_pose_matrix)

    curi.set_tcp_moveL(initial_robot_left_pose_matrix, initial_robot_right_pose_matrix)

    while curi.get_curi_mode(0) != 2 and curi.get_curi_mode(1) != 2:
        print("waiting robot external control")
        time.sleep(1)

    print("Start planning...")
    time.sleep(3)
    subscriber_robot = rospy.wait_for_message('/vrpn_client_node/robot/pose', PoseStamped)
    # subscriber_shouL = rospy.wait_for_message('/vrpn_client_node/shouL/pose', PoseStamped)
    subscriber_shouR = rospy.wait_for_message('/vrpn_client_node/shouR/pose', PoseStamped)
    # subscriber_elbowL = rospy.wait_for_message('/vrpn_client_node/elbowL/pose', PoseStamped)
    subscriber_elbowR = rospy.wait_for_message('/vrpn_client_node/elbowR/pose', PoseStamped)
    # subscriber_wristL = rospy.wait_for_message('/vrpn_client_node/wristL/pose', PoseStamped)
    subscriber_wristR = rospy.wait_for_message('/vrpn_client_node/wristR/pose', PoseStamped)
    print("Optitrack data collected successfuldataly!")

    sub_robot = transform_to_pose(subscriber_robot)
    # sub_shouL = transform_to_pose(subscriber_shouL)
    sub_shouR = transform_to_pose(subscriber_shouR)
    # sub_elbowL = transform_to_pose(subscriber_elbowL)
    sub_elbowR = transform_to_pose(subscriber_elbowR)
    # sub_wristL = transform_to_pose(subscriber_wristL)
    sub_wristR = transform_to_pose(subscriber_wristR)

    T_optitrack2robotbase = np.linalg.inv(
        tsf.transform_optitrack_origin_to_optitrack_robot(
            sub_robot) @ tsf.transform_optitrack_robot_to_robot_base())
    # shouL_position_init = T_optitrack2robotbase[:3, :3] @ sub_shouL[:3] + T_optitrack2robotbase[:3, 3]
    shouR_position_init = T_optitrack2robotbase[:3, :3] @ sub_shouR[:3] + T_optitrack2robotbase[:3, 3]
    # elbowL_position_init = T_optitrack2robotbase[:3, :3] @ sub_elbowL[:3] + T_optitrack2robotbase[:3, 3]
    elbowR_position_init = T_optitrack2robotbase[:3, :3] @ sub_elbowR[:3] + T_optitrack2robotbase[:3, 3]
    # wristL_position_init = T_optitrack2robotbase[:3, :3] @ sub_wristL[:3] + T_optitrack2robotbase[:3, 3]
    wristR_position_init = T_optitrack2robotbase[:3, :3] @ sub_wristR[:3] + T_optitrack2robotbase[:3, 3]

    joint_angle_bounds = [
        (-math.pi / 18, 17 * math.pi / 18),  # Joint 1
        (-math.pi / 18, 17 * math.pi / 18),  # Joint 2
        (-np.pi / 3, np.pi / 2),  # Joint 3
        (-np.pi / 2, np.pi / 3)  # Joint 4
    ]
    optimal_q = [0, 0, 0, -math.pi / 4]

    skeleton_joint_name, skeleton_joints, skeleton_parent_indices, skeleton_joint_local_translation = \
        utils.read_skeleton_motion('/data/demo_2_test_chenzui_only_optitrack2hotu.npy')
    skeleton_joint = skeleton_joints[500, :]
    global_positions, global_rotations = utils.forward_kinematics(skeleton_joint_local_translation,
                                                                  skeleton_joint, skeleton_parent_indices)
    global_positions[:, 2] = global_positions[:, 2] * 1.2

    global_positions[4] = global_positions[3] + (elbowR_position_init - shouR_position_init)
    # global_positions[7] = global_positions[6] + (elbowL_position_init - shouL_position_init)
    global_positions[5] = global_positions[3] + (wristR_position_init - shouR_position_init)
    # global_positions[8] = global_positions[6] + (wristL_position_init - shouL_position_init)

    shou_center = shouR_position_init
    global_positions = global_positions + np.array([shou_center[0], shou_center[1], 0])

    initial_position = global_positions[5]

    # Body dimensions
    d_ual, d_uar, d_lal, d_lar = mos.calculate_arm_dimensions(shouR_position_init, elbowR_position_init,
                                                              wristR_position_init, shouR_position_init,
                                                              elbowR_position_init, wristR_position_init)

    # 计算初始"最优"位置（仅用于可视化对比），这里采用 optimal_q 得到的手腕位置
    _, optimal_position = mos.forward_kinematics(optimal_q, d_uar, d_lar)
    # optimal_position = trans_shoulder2global(optimal_position, global_positions[6], arm='left')
    optimal_position = trans_shoulder2global(optimal_position, global_positions[3], arm='right')

    # p_elbowL_init, p_wristL_init = trans_global2shoulder(shouL_position_init, elbowL_position_init,
    #                                                      wristL_position_init, arm='left')
    p_elbowR_init, p_wristR_init = trans_global2shoulder(global_positions[3], global_positions[4], global_positions[5],
                                                         arm='right')

    current_q = mos.inverse_kinematics(p_elbowR_init, p_wristR_init, d_uar, d_lar)
    current_score = utils.calculate_upper_limb_score_with_joint_angles(current_q)

    # hand_current = global_positions[8]
    # elbow_current = global_positions[7]
    hand_current = global_positions[5]
    elbow_current = global_positions[4]

    # shoulder = global_positions[6].copy()
    shoulder = global_positions[3].copy()

    # 为动画记录历史轨迹（可选）
    trajectory_hand = [hand_current.copy()]
    trajectory_elbow = [elbow_current.copy()]

    score_history = []
    joint_history = []

    # 规划方法选择: 0=Straight(对照组), 1=TSEF, 2=SDF, 3=CF（每次运行只验证一种，通过环境变量 PLANNING_METHOD 设置）
    PLANNING_METHOD = int(os.environ.get('PLANNING_METHOD', 2))
    method_names = {0: 'Straight', 1: 'TSEF', 2: 'HD-SDF', 3: 'CF'}
    print("Using planning method: {} ({})".format(PLANNING_METHOD, method_names.get(PLANNING_METHOD, 'Unknown')))

    task_goal_global = np.array([1.15, 0.4, 1.05])
    reference_trajectory = None
    ref_kdtree = None
    if PLANNING_METHOD in (2, 3):
        reference_trajectory = cf_plan.generate_reference_trajectory(
            task_goal_global + np.array([0.0, 0.0, 0.3]), task_goal_global, num_points=100, trajectory_type='straight')
        ref_kdtree = KDTree(reference_trajectory) if reference_trajectory is not None else None

    # 0=对照组直线规划(w_goal=1); 1=TSEF; 2=HD-SDF; 3=CF
    if PLANNING_METHOD == 0:
        w_goal, w_ref, w_ergo = 1.0, 0.0, 0.0
    elif PLANNING_METHOD == 1:
        w_goal, w_ref, w_ergo = 0.6, 0.0, 0.4
    elif PLANNING_METHOD == 2:
        w_goal, w_ref, w_ergo = 0.40, 0.60, 0.0
    else:
        w_goal, w_ref, w_ergo = 0.45, 0.25, 0.30

    max_plan_steps = 1000
    goal_threshold = 0.01
    step_size = 0.04
    control_dt = 0.01
    # 位移平滑：每步最大变化量（米/轴），避免底层力矩变化率过大
    max_displacement_delta_per_step = 0.004  # 4mm per axis per step
    displacement_cmd = np.zeros(3)  # 实际下发的位移（平滑后）
    # 运动方向滤波：对规划得到的 unit direction 做 EMA，避免 cmd 方向跳变（method 2/3 接近 ref 时易突变）
    motion_direction_smooth_alpha = 0.65  # 越大方向越平滑、跟踪越慢
    motion_direction_smooth = None
    # 参考轨迹向量平滑：接近 ref 时最近点会跳变，对 ref_vec 做 EMA 过渡
    ref_vec_smooth = None
    ref_vec_smooth_alpha = 0.55  # 越大 ref 方向越平滑
    # 接近 reference 时的过渡：距离 < ref_transition_dist 时逐渐减弱 ref 权重，避免突变
    ref_transition_dist = 0.025  # 25mm 内线性减弱 ref 权重
    ref_transition_dist_zero = 0.012  # 12mm 内 ref 权重为 0，完全由 goal/ergo 主导
    # OptiTrack 输入平滑：指数移动平均，减弱抖动对规划与位移输出的影响
    optitrack_smooth_alpha = 0.8  # 越大平滑越强、延迟越大，建议 0.5~0.8
    shouR_position_smooth = None
    elbowR_position_smooth = None
    wristR_position_smooth = None
    # 机器人 cmd position 平滑：对下发的末端位置做 EMA，避免 motion direction 跳变、减轻力矩变化率
    robot_cmd_smooth_alpha = 0.7  # 越大平滑越强、跟踪越慢，建议 0.5~0.8
    robot_left_position_smooth = None

    print("left_arm_current", curi.get_tcp(0))
    print("right_arm_current", curi.get_tcp(1))

    # 等待一些初始消息到达
    print("Waiting for initial pose messages...")
    start_time = rospy.Time.now()
    while (latest_robot_msg is None or latest_shouR_msg is None or latest_elbowR_msg is None or latest_wristR_msg is None) and \
            (rospy.Time.now() - start_time).to_sec() < 3.0:  # 5秒超时
        time.sleep(0.1)

    if latest_robot_msg is None or latest_shouR_msg is None or latest_elbowR_msg is None or latest_wristR_msg is None:
        rospy.logwarn("Not all initial pose messages received. Proceeding anyway.")

    recorded_shoulder_positions = []
    recorded_elbow_positions = []
    recorded_wrist_positions = []
    recorded_timestamps = []
    optimized_robot_positions = []
    planned_motion_directions = []  # 每步规划的 motion direction (unit vector), 保存为 npy

    setup_subscribers()

    folder = '/home/clover/Chenzui/Ergo-Manip/data/composite_field/0316/chenzui/3_mid'
    os.makedirs(folder, exist_ok=True)
    start_time = time.time()
    # emg_processor = EMGProcessor(channel_num=5, sample_fre=200, start_time=start_time, save=True, save_folder=folder)
    # data_queue = queue.Queue()
    # threads = [
    #     threading.Thread(target=emg_processor.read_emg, args=(data_queue,), name="EMG-Reader"),
    #     threading.Thread(target=emg_processor.process_emg, args=(data_queue,), name="EMG-Processor")
    # ]
    # for t in threads:
    #     t.daemon = True
    #     t.start()
    # time.sleep(10.0)
    # print("EMG processor initialized")

    print("start robot executing...")
    for step in range(max_plan_steps):
        if not (latest_robot_msg and latest_shouR_msg and latest_elbowR_msg and latest_wristR_msg):
            time.sleep(0.01)
            continue
        sub_robot = transform_to_pose(latest_robot_msg)
        sub_shouR = transform_to_pose(latest_shouR_msg)
        sub_elbowR = transform_to_pose(latest_elbowR_msg)
        sub_wristR = transform_to_pose(latest_wristR_msg)
        T_optitrack2robotbase = np.linalg.inv(
            tsf.transform_optitrack_origin_to_optitrack_robot(
                sub_robot) @ tsf.transform_optitrack_robot_to_robot_base())
        shouR_position = T_optitrack2robotbase[:3, :3] @ sub_shouR[:3] + T_optitrack2robotbase[:3, 3]
        elbowR_position = T_optitrack2robotbase[:3, :3] @ sub_elbowR[:3] + T_optitrack2robotbase[:3, 3]
        wristR_position = T_optitrack2robotbase[:3, :3] @ sub_wristR[:3] + T_optitrack2robotbase[:3, 3]

        print("sub_shouR:", sub_shouR)
        print("sub_robot:", sub_robot)
        print("sub_elbowR:", sub_elbowR)
        print("sub_wristR:", sub_wristR)

        # OptiTrack 数据指数移动平均平滑，减少抖动
        if shouR_position_smooth is None:
            shouR_position_smooth = shouR_position.copy()
            elbowR_position_smooth = elbowR_position.copy()
            wristR_position_smooth = wristR_position.copy()
        else:
            shouR_position_smooth = optitrack_smooth_alpha * shouR_position_smooth + (1 - optitrack_smooth_alpha) * shouR_position
            elbowR_position_smooth = optitrack_smooth_alpha * elbowR_position_smooth + (1 - optitrack_smooth_alpha) * elbowR_position
            wristR_position_smooth = optitrack_smooth_alpha * wristR_position_smooth + (1 - optitrack_smooth_alpha) * wristR_position

        recorded_shoulder_positions.append(shouR_position_smooth.copy())
        recorded_elbow_positions.append(elbowR_position_smooth.copy())
        recorded_wrist_positions.append(wristR_position_smooth.copy())
        recorded_timestamps.append(time.time() - start_time)

        p_elbowR, p_wristR = trans_global2shoulder(shouR_position_smooth, elbowR_position_smooth, wristR_position_smooth, arm='right')
        current_q = mos.inverse_kinematics(p_elbowR, p_wristR, d_uar, d_lar)

        endpoint = wristR_position_smooth.copy()
        goal_dist = np.linalg.norm(task_goal_global - endpoint)
        current_score = utils.calculate_upper_limb_score_with_joint_angles(current_q)
        score_history.append(current_score)
        joint_history.append(current_q.copy())
        trajectory_hand.append(endpoint.copy())

        if goal_dist < goal_threshold:
            print("Reached goal at step {}".format(step))
            optimized_robot_positions.append(endpoint.copy())
            break

        goal_vec = task_goal_global - endpoint
        if np.linalg.norm(goal_vec) > 1e-8:
            goal_vec = goal_vec / np.linalg.norm(goal_vec)
        else:
            goal_vec = np.zeros(3)

        ref_vec = np.zeros(3)
        dist_to_ref = float('inf')
        w_ref_effective = w_ref
        if ref_kdtree is not None and w_ref > 0:
            dist_to_ref, idx = ref_kdtree.query(endpoint, k=1)
            dist_to_ref = np.atleast_1d(dist_to_ref)[0]
            idx = np.atleast_1d(idx)[0]
            closest_point = reference_trajectory[idx]
            ref_vec_raw = closest_point - endpoint
            if np.linalg.norm(ref_vec_raw) > 1e-8:
                ref_vec = ref_vec_raw / np.linalg.norm(ref_vec_raw)
            # 接近 reference 时的过渡：距离越近越减弱 ref 权重，避免最近点跳变导致 cmd 突变
            if dist_to_ref < ref_transition_dist_zero:
                w_ref_effective = 0.0
            elif dist_to_ref < ref_transition_dist:
                w_ref_effective = w_ref * (dist_to_ref - ref_transition_dist_zero) / (ref_transition_dist - ref_transition_dist_zero)
            # ref_vec 时序平滑：与上一帧做 EMA，减轻最近点索引跳变带来的方向突变
            if ref_vec_smooth is None:
                ref_vec_smooth = ref_vec.copy()
            else:
                if np.linalg.norm(ref_vec) > 1e-8 and np.linalg.norm(ref_vec_smooth) > 1e-8:
                    ref_vec_smooth = ref_vec_smooth_alpha * ref_vec_smooth + (1 - ref_vec_smooth_alpha) * ref_vec
                    ref_norm = np.linalg.norm(ref_vec_smooth)
                    if ref_norm > 1e-8:
                        ref_vec_smooth = ref_vec_smooth / ref_norm
                else:
                    ref_vec_smooth = ref_vec.copy()
            ref_vec = ref_vec_smooth

        # Ergo vector: joint-space neighbors of current_q -> optimal_q -> FK to task space -> direction to that position
        ergo_vec = np.zeros(3)
        if w_ergo > 0:
            ergo_vec = cf_plan.compute_ergonomic_vector_task_space(
                endpoint, shouR_position_smooth, current_q, d_uar, d_lar, joint_angle_bounds,
                n_samples=125, joint_neighbor_radius=0.06)

        combined = w_goal * goal_vec + w_ref_effective * ref_vec + w_ergo * ergo_vec
        combined_norm = np.linalg.norm(combined)
        if combined_norm > 1e-8:
            combined = combined / combined_norm
        else:
            combined = goal_vec

        # 记录本步规划的 motion direction（平滑前）
        planned_motion_directions.append(combined.copy())

        # 运动方向滤波：method 2/3 时对 combined 做 EMA，接近 ref 时避免 cmd 方向跳变
        if PLANNING_METHOD in (2, 3):
            if motion_direction_smooth is None:
                motion_direction_smooth = combined.copy()
            else:
                motion_direction_smooth = motion_direction_smooth_alpha * motion_direction_smooth + (1 - motion_direction_smooth_alpha) * combined
                md_norm = np.linalg.norm(motion_direction_smooth)
                if md_norm > 1e-8:
                    motion_direction_smooth = motion_direction_smooth / md_norm
                else:
                    motion_direction_smooth = goal_vec.copy()
            combined = motion_direction_smooth

        adaptive_step = min(step_size, goal_dist * 0.4)
        next_waypoint = endpoint + adaptive_step * combined
        q_next, ik_error = cf_plan.ik_target_point(
            next_waypoint, shouR_position_smooth, current_q, d_uar, d_lar, joint_angle_bounds, maxiter=120, ftol=1e-8)

        # 机器人左臂：从 init 做增量叠加；对位移做率限平滑，避免力矩变化率过大
        raw_displacement = next_waypoint - wristR_position_init
        delta = np.clip(raw_displacement - displacement_cmd,
                       -max_displacement_delta_per_step, max_displacement_delta_per_step)
        displacement_cmd = displacement_cmd + delta
        robot_left_position = robot_left_position_init + displacement_cmd
        # Cmd position 平滑：method 3 时对下发的末端位置做 EMA，使轨迹更平滑、减轻接近 ref 时的突变
        # if PLANNING_METHOD == 3:
        #     if robot_left_position_smooth is None:
        #         robot_left_position_smooth = robot_left_position.copy()
        #     else:
        #         robot_left_position_smooth = robot_cmd_smooth_alpha * robot_left_position_smooth + (1 - robot_cmd_smooth_alpha) * robot_left_position
        #     robot_left_position = robot_left_position_smooth.copy()
        optimized_robot_positions.append(robot_left_position.copy())
        robot_right_position = robot_right_position_init

        robot_left_pose_matrix = np.r_[
            np.c_[robot_left_rotation_matrix_init, robot_left_position.T], np.array([[0, 0, 0, 1]])]
        robot_right_pose_matrix = np.r_[
            np.c_[robot_right_rotation_matrix_init, robot_right_position.T], np.array([[0, 0, 0, 1]])]
        robot_left_pose_matrix = base2torso_matrix @ robot_left_pose_matrix
        robot_right_pose_matrix = base2torso_matrix @ robot_right_pose_matrix
        curi.set_tcp_servo(robot_left_pose_matrix, robot_right_pose_matrix)

        if step % 50 == 0:
            print("Step {}: score={:.4f}, goal_dist={:.1f}mm, IK_err={:.2f}mm, displacement_cmd={}".format(
                step, current_score, goal_dist * 1000, ik_error * 1000, displacement_cmd.round(5)))
        time.sleep(control_dt)

    recorded_data = {
        'timestamps': recorded_timestamps,
        'shoulder_positions': recorded_shoulder_positions,
        'elbow_positions': recorded_elbow_positions,
        'wrist_positions': recorded_wrist_positions
    }
    np.save(os.path.join(folder, 'recorded_human_position.npy'), recorded_data)
    np.save(os.path.join(folder, 'optimized_robot_positions.npy'), np.array(optimized_robot_positions))
    np.save(os.path.join(folder, 'optimized_joint_angles.npy'), np.array(joint_history))
    np.save(os.path.join(folder, 'ergonomics_scores.npy'), np.array(score_history))
    np.save(os.path.join(folder, 'planned_motion_directions.npy'), np.array(planned_motion_directions))

    print(f"Recorded {len(recorded_timestamps)} position samples")
    print("轨迹执行完成！")

    # emg_processor.read_emg_flag = False
    # data_queue.join()
    # for t in threads:
    #     t.join()

    while 1:
        interrupt = False
        time.sleep(1)