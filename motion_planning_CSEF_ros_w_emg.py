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
from mpl_toolkits.mplot3d import Axes3D

from EMGProcessor import EMGProcessor

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


def update(frame):
    global current_q, global_positions, trajectory_hand, trajectory_elbow
    ax.clear()
    ax.set_xlim((0.7, 2.2))
    ax.set_ylim((-0.7, 0.8))
    ax.set_zlim((0.0, 1.5))
    # ax.set_xlim((-0.5, 0.1))
    # ax.set_ylim((0, 0.6))
    # ax.set_zlim((0.9, 1.5))

    ax.view_init(elev=30, azim=-30)

    new_bounds = compress_bounds(joint_angle_bounds, current_q, compression_factor=comp_factor)
    joint_angle_ranges = [np.linspace(lower, upper, num_samples_per_joint) for lower, upper in new_bounds]

    q_combinations = np.array(list(product(*joint_angle_ranges)))

    scores = []
    candidate_elbows = []
    candidate_hands = []

    for q in q_combinations:
        elbow_cand, hand_cand = mos.forward_kinematics(q, d_uar, d_lar)
        hand_cand = trans_shoulder2global(hand_cand, shoulder, arm='right')
        elbow_cand = trans_shoulder2global(elbow_cand, shoulder, arm='right')

        candidate_elbows.append(elbow_cand)
        candidate_hands.append(hand_cand)
        s = utils.calculate_upper_limb_score_with_joint_angles(q)
        scores.append(s)

    scores = np.array(scores)
    candidate_elbows = np.array(candidate_elbows)
    candidate_hands = np.array(candidate_hands)

    norm_obj = Normalize(vmin=scores.min(), vmax=scores.max())
    cmap = plt.get_cmap('coolwarm')
    colors = cmap(norm_obj(scores))
    # ax.scatter(candidate_hands[:, 0], candidate_hands[:, 1], candidate_hands[:, 2],
    #            c=colors, s=5, alpha=0.5)

    # ref_point = global_positions[8]
    ref_point = global_positions[5]

    ## Find the neighbor with the lowest ergo score

    target_idx = np.argmin(scores)
    candidate_q = q_combinations[target_idx]

    new_elbow = candidate_elbows[target_idx]
    new_hand = candidate_hands[target_idx]

    dist = np.linalg.norm(new_hand - ref_point)
    if dist > max_disp:
        ratio = max_disp / dist
    else:
        ratio = 1.0

    new_q = current_q + ratio * (candidate_q - current_q)

    ## Find the neighbor pointing to the optimal point

    # A = ref_point
    # B = optimal_position
    # expected_p = A + frame * (B - A) / 20
    #
    # candidate_expected_dists = np.linalg.norm(candidate_hands - expected_p, axis=1)
    # target_idx = np.argmin(candidate_expected_dists)
    # candidate_q = q_combinations[target_idx]
    # candidate_hand = candidate_hands[target_idx]
    # candidate_elbow = candidate_elbows[target_idx]
    #
    # # 计算从当前手腕到选择候选点的位移
    # disp = candidate_hand - ref_point
    # dist = np.linalg.norm(disp)
    # if dist > max_disp:
    #     ratio = max_disp / dist
    # else:
    #     ratio = 1.0
    #
    # # 在关节空间中按比例线性插值更新配置
    # new_q = current_q + ratio * (candidate_q - current_q)

    new_elbow, new_hand = mos.forward_kinematics(new_q, d_uar, d_lar)
    new_hand = trans_shoulder2global(new_hand, shoulder, arm='right')
    new_elbow = trans_shoulder2global(new_elbow, shoulder, arm='right')

    current_q = new_q

    s = utils.calculate_upper_limb_score_with_joint_angles(current_q)
    score_history.append(s)

    # global_positions[7] = new_elbow
    # global_positions[8] = new_hand
    global_positions[4] = new_elbow
    global_positions[5] = new_hand

    # 将新位置加入轨迹（便于绘制轨迹）
    trajectory_hand.append(new_hand.copy())
    trajectory_elbow.append(new_elbow.copy())
    joint_history.append(new_q.copy())

    # ---------------------- 绘制部分 ----------------------
    # 绘制肩部、肘部、手腕点（用不同颜色标记），以及轨迹
    ax.scatter(shoulder[0], shoulder[1], shoulder[2], c='black', s=50, label='Shoulder')
    ax.scatter(new_elbow[0], new_elbow[1], new_elbow[2], c='blue', s=50, label='Elbow')
    ax.scatter(new_hand[0], new_hand[1], new_hand[2], c='green', s=50, label='Hand')

    # 绘制轨迹（手腕轨迹）
    traj = np.array(trajectory_hand)
    ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], c='green', linestyle='--')

    # 绘制从肩部到手腕的连线，模拟手臂
    ax.plot([shoulder[0], new_elbow[0], new_hand[0]],
            [shoulder[1], new_elbow[1], new_hand[1]],
            [shoulder[2], new_elbow[2], new_hand[2]], c='red', linewidth=2)

    # 绘制参考最优位置（optimal_position，供对比）
    ax.scatter(optimal_position[0], optimal_position[1], optimal_position[2],
               c='magenta', s=50, label='Optimal Position')

    utils.plot_skeleton(ax, global_positions, skeleton_parent_indices, color='black')

    # 设定坐标轴标签与标题
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_zlabel('Z Position')
    ax.set_title(f'Iteration {frame + 1}')
    ax.legend()


def run_iterations(num_iterations):
    global current_q, global_positions, trajectory_hand, trajectory_elbow, score_history, joint_history

    # 定义CSEF场所需参数
    q_opt = optimal_q  # 最优关节角度配置
    weights = np.array([1.0, 1.0, 1.0, 2.0])  # 关节角度权重
    comfort_threshold = 0.1  # 舒适阈值
    alpha = 0.7  # 目标吸引力权重
    beta = 0.3  # 人体工学舒适度权重
    step_size = 0.05  # 步长

    # 用于保存结果的数组
    trajectory = []
    q_current = current_q.copy()
    q_goal = optimal_q.copy()  # 目标关节角度配置

    print("Planning trajectory using CSEF guidance...")

    # 计算SEF值 (Signed Ergonomics Field)
    def calculate_sef(q, q_opt, weights, comfort_threshold):
        ergo_score = utils.calculate_upper_limb_score_with_joint_angles(q)
        score_history.append(ergo_score)
        return ergo_score - comfort_threshold

    # 计算SEF梯度 (方向指向不舒适度增加的方向)
    def calculate_sef_gradient(q, q_opt, weights, comfort_threshold, delta=1e-6):
        grad = np.zeros_like(q)
        sef_q = calculate_sef(q, q_opt, weights, comfort_threshold)

        for i in range(len(q)):
            q_plus = q.copy()
            q_plus[i] += delta
            sef_plus = calculate_sef(q_plus, q_opt, weights, comfort_threshold)
            grad[i] = (sef_plus - sef_q) / delta

        return grad

    # 确保关节角度在限制范围内
    def enforce_joint_limits(q, bounds):
        q_limited = np.copy(q)
        for i in range(len(q)):
            q_limited[i] = max(bounds[i][0], min(bounds[i][1], q[i]))
        return q_limited

    # 添加初始位置到轨迹
    trajectory.append(q_current)

    # 主循环：使用CSEF指导的梯度下降
    for step in range(num_iterations):
        # 计算指向目标的向量
        goal_direction = q_goal - q_current
        goal_distance = np.linalg.norm(goal_direction)

        # 如果足够接近目标，结束轨迹
        if goal_distance < 0.05:
            trajectory.append(q_goal)
            break

        # 归一化目标方向
        if goal_distance > 0:
            goal_direction = goal_direction / goal_distance

        # 获取SEF梯度（不舒适度增加的方向）
        sef_gradient = calculate_sef_gradient(q_current, q_opt, weights, comfort_threshold)
        sef_gradient_norm = np.linalg.norm(sef_gradient)

        # 归一化SEF梯度（如果非零）
        if sef_gradient_norm > 0:
            sef_gradient = sef_gradient / sef_gradient_norm

        # 组合目标方向和负SEF梯度（朝向更舒适的方向）
        combined_direction = alpha * goal_direction - beta * sef_gradient
        combined_norm = np.linalg.norm(combined_direction)

        if combined_norm > 0:
            # 归一化并移动一步
            combined_direction = combined_direction / combined_norm
            q_next = q_current + step_size * combined_direction
        else:
            # 如果方向相互抵消，优先考虑目标方向
            q_next = q_current + step_size * goal_direction

        # 确保关节角度在限制范围内
        q_next = enforce_joint_limits(q_next, joint_angle_bounds)

        # 使用前向运动学计算新的手肘和手腕位置
        new_elbow, new_hand = mos.forward_kinematics(q_next, d_uar, d_lar)
        new_hand = trans_shoulder2global(new_hand, shoulder, arm='right')
        new_elbow = trans_shoulder2global(new_elbow, shoulder, arm='right')

        # 添加到轨迹并更新当前位置
        trajectory.append(q_next)
        q_current = q_next
        joint_history.append(q_next.copy())

        # 更新全局位置数组
        global_positions[4] = new_elbow
        global_positions[5] = new_hand

        # 将新位置添加到手腕和肘部轨迹
        trajectory_hand.append(new_hand.copy())
        trajectory_elbow.append(new_elbow.copy())

        # 自适应步长：接近目标时减小
        step_size = min(0.05, goal_distance * 0.2)

        print(f"Iteration {step + 1}/{num_iterations} completed. Current score: {score_history[-1]:.4f}")

    # 设置最终配置为当前配置
    current_q = q_current

    # 迭代完成后绘制最终结果
    ax.set_xlim((0.7, 2.2))
    ax.set_ylim((-0.7, 0.8))
    ax.set_zlim((0.0, 1.5))
    ax.view_init(elev=30, azim=-30)

    # 获取最终位置
    new_elbow = global_positions[4]
    new_hand = global_positions[5]

    # 绘制肩部、肘部、手腕点（用不同颜色标记），以及轨迹
    ax.scatter(shoulder[0], shoulder[1], shoulder[2], c='black', s=50, label='Shoulder')
    ax.scatter(new_elbow[0], new_elbow[1], new_elbow[2], c='blue', s=50, label='Elbow')
    ax.scatter(new_hand[0], new_hand[1], new_hand[2], c='green', s=50, label='Hand')

    # 绘制轨迹（手腕轨迹）
    traj = np.array(trajectory_hand)
    ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], c='green', linestyle='--')

    # 绘制从肩部到手腕的连线，模拟手臂
    ax.plot([shoulder[0], new_elbow[0], new_hand[0]],
            [shoulder[1], new_elbow[1], new_hand[1]],
            [shoulder[2], new_elbow[2], new_hand[2]], c='red', linewidth=2)

    # 绘制参考最优位置（optimal_position，供对比）
    ax.scatter(optimal_position[0], optimal_position[1], optimal_position[2],
               c='magenta', s=50, label='Optimal Position')

    utils.plot_skeleton(ax, global_positions, skeleton_parent_indices, color='black')

    # 设定坐标轴标签与标题
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_zlabel('Z Position')
    ax.set_title(f'Final Result after {num_iterations} Iterations (CSEF Method)')
    ax.legend()

    plt.show()

    # 绘制分数历史趋势图
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(score_history) + 1), score_history)
    plt.xlabel('Iteration')
    plt.ylabel('Ergonomic Score')
    plt.title('Ergonomic Score History (CSEF Method)')
    plt.grid(True)
    plt.show()

    # 可选：绘制SEF值历史趋势
    sef_values = [score - comfort_threshold for score in score_history]
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(sef_values) + 1), sef_values)
    plt.axhline(y=0, color='r', linestyle='--', label='Comfort Threshold')
    plt.xlabel('Iteration')
    plt.ylabel('SEF Value')
    plt.title('Signed Ergonomics Field Values (CSEF Method)')
    plt.grid(True)
    plt.legend()
    plt.show()

    return trajectory_hand, trajectory_elbow, joint_history, score_history


latest_shouR_msg = None
latest_elbowR_msg = None
latest_wristR_msg = None


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
    rospy.Subscriber('/vrpn_client_node/shouR/pose', PoseStamped, shouR_callback)
    rospy.Subscriber('/vrpn_client_node/elbowR/pose', PoseStamped, elbowR_callback)
    rospy.Subscriber('/vrpn_client_node/wristR/pose', PoseStamped, wristR_callback)


if __name__ == '__main__':
    rospy.init_node('vf_hrc')
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
    robot_left_position_init = np.array([0.85, 0.2, 1.25])
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
    print("collecting human data successfully!")

    sub_robot = transform_to_pose(subscriber_robot)
    # sub_shouL = transform_to_pose(subscriber_shouL)
    sub_shouR = transform_to_pose(subscriber_shouR)
    # sub_elbowL = transform_to_pose(subscriber_elbowL)
    sub_elbowR = transform_to_pose(subscriber_elbowR)
    # sub_wristL = transform_to_pose(subscriber_wristL)
    sub_wristR = transform_to_pose(subscriber_wristR)

    # sub_robot = np.array([-0.2195, 1.11462, 0, 0, 0, 0, 1])
    # sub_shouL = np.array([2, 1.5, 0.25, 0, 0, 0, 1])
    # sub_shouR = np.array([2, 1.5, -0.25, 0, 0, 0, 1])
    # sub_elbowL = np.array([1.9, 1.3, 0.3, 0, 0, 0, 1])
    # sub_elbowR = np.array([1.9, 1.3, -0.3, 0, 0, 0, 1])
    # sub_wristL = np.array([1.8, 1.2, 0.3, 0, 0, 0, 1])
    # sub_wristR = np.array([1.8, 1.4, -0.3, 0, 0, 0, 1])

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
        utils.read_skeleton_motion('/home/clover/Chenzui/Ergo-Manip/data/demo_2_test_chenzui_only_optitrack2hotu.npy')
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

    # 设置候选离散采样数、压缩系数和迭代次数
    num_samples_per_joint = 15
    comp_factor = 0.1
    num_iterations = 80
    max_disp = 0.02  # maximum allowed displacement per iteration in global (hand) space

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # ---------------------- 动画启动 ----------------------
    # anim = FuncAnimation(fig, update, frames=num_iterations, interval=800, repeat=False)
    # anim.save("/home/ubuntu/Ergo-Manip/vector_field/figs/animation_left_arm_straight.gif", writer=PillowWriter(fps=2))
    # plt.show()

    trajectory_hand, trajectory_elbow, joint_history, score_history = run_iterations(num_iterations)

    # 在这里添加你想在迭代完成后执行的代码
    print("Iterations completed. Continuing with next steps...")

    # 首先对轨迹进行平滑处理
    print("平滑处理轨迹...")
    smoothed_trajectory_hand = smooth_trajectory(np.array(trajectory_hand), smoothing_factor=0.8, iterations=5)

    waypoints_ergo = smoothed_trajectory_hand
    waypoints_straight = np.array([smoothed_trajectory_hand[0], smoothed_trajectory_hand[-1]])

    speed_limit = 0.05  # 最大速度限制
    t_total = 8  # 总时间
    t_sample = 0.0025  # 采样时间间隔

    # 生成平滑轨迹
    print("生成平滑轨迹...")
    trajectory_positions_ergo = generate_smooth_trajectory(waypoints_ergo, speed_limit, t_total, t_sample)
    trajectory_positions_straight = generate_smooth_trajectory(waypoints_straight, speed_limit, t_total, t_sample)

    # 可视化轨迹平滑度
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 3, 1)
    plt.plot(trajectory_positions_ergo[:, 0], label='X')
    plt.title('X Position')
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.plot(trajectory_positions_ergo[:, 1], label='Y')
    plt.title('Y Position')
    plt.grid(True)

    plt.subplot(1, 3, 3)
    plt.plot(trajectory_positions_ergo[:, 2], label='Z')
    plt.title('Z Position')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # 计算相对于初始位置的位移
    # position_ergo = trajectory_positions_ergo - trajectory_positions_ergo[0]
    position_ergo = trajectory_positions_straight - trajectory_positions_straight[0]

    print("left_arm_current", curi.get_tcp(0))
    print("right_arm_current", curi.get_tcp(1))


    # 等待一些初始消息到达
    print("Waiting for initial pose messages...")
    start_time = rospy.Time.now()
    while (latest_shouR_msg is None or latest_elbowR_msg is None or latest_wristR_msg is None) and \
            (rospy.Time.now() - start_time).to_sec() < 3.0:  # 5秒超时
        time.sleep(0.1)

    if latest_shouR_msg is None or latest_elbowR_msg is None or latest_wristR_msg is None:
        rospy.logwarn("Not all initial pose messages received. Proceeding anyway.")

    # 记录位置数据
    recorded_shoulder_positions = []
    recorded_elbow_positions = []
    recorded_wrist_positions = []
    recorded_timestamps = []

    setup_subscribers()

    folder = '/home/clover/Chenzui/Ergo-Manip/data/drilling/0921/wuxi/2.3'
    os.makedirs(folder, exist_ok=True)
    start_time = time.time()
    # emg_processor = EMGProcessor(channel_num=5, sample_fre=200, start_time=start_time, save=True, save_folder=folder)
    # data_queue = queue.Queue()
    # threads = [
    #     threading.Thread(
    #         target=emg_processor.read_emg,
    #         args=(data_queue,),
    #         name="EMG-Reader"
    #     ),
    #     threading.Thread(
    #         target=emg_processor.process_emg,
    #         args=(data_queue,),
    #         name="EMG-Processor"
    #     )
    # ]
    # for t in threads:
    #     t.daemon = True
    #     t.start()
    # time.sleep(10.0)
    # print("EMG processor initialized")

    print(f"开始执行轨迹，共{len(position_ergo)}个点...")
    for i in range(len(position_ergo)):

        if i % 100 == 0:
            print(f"执行到第{i}个点...")

        current_timestamp = rospy.Time.now()

        # 使用最新可用的消息
        if latest_shouR_msg and latest_elbowR_msg and latest_wristR_msg:
            # 转换为位置数据
            sub_shouR = transform_to_pose(latest_shouR_msg)
            sub_elbowR = transform_to_pose(latest_elbowR_msg)
            sub_wristR = transform_to_pose(latest_wristR_msg)

            # 转换到机器人基座标系
            shouR_position = T_optitrack2robotbase[:3, :3] @ sub_shouR[:3] + T_optitrack2robotbase[:3, 3]
            elbowR_position = T_optitrack2robotbase[:3, :3] @ sub_elbowR[:3] + T_optitrack2robotbase[:3, 3]
            wristR_position = T_optitrack2robotbase[:3, :3] @ sub_wristR[:3] + T_optitrack2robotbase[:3, 3]

            # 记录位置数据
            recorded_shoulder_positions.append(shouR_position.copy())
            recorded_elbow_positions.append(elbowR_position.copy())
            recorded_wrist_positions.append(wristR_position.copy())
            # recorded_timestamps.append(current_timestamp.to_sec())
            recorded_timestamps.append(time.time() - start_time)

        # 执行机器人控制代码
        robot_left_position = robot_left_position_init + position_ergo[i]
        robot_right_position = robot_right_position_init

        robot_left_pose_matrix = np.r_[
            np.c_[robot_left_rotation_matrix_init, robot_left_position.T], np.array([[0, 0, 0, 1]])]
        robot_right_pose_matrix = np.r_[
            np.c_[robot_right_rotation_matrix_init, robot_right_position.T], np.array([[0, 0, 0, 1]])]

        robot_left_pose_matrix = base2torso_matrix @ robot_left_pose_matrix
        robot_right_pose_matrix = base2torso_matrix @ robot_right_pose_matrix

        curi.set_tcp_servo(robot_left_pose_matrix, robot_right_pose_matrix)
        time.sleep(0.001)  # 使用轨迹采样时间间隔

    recorded_data = {
        'timestamps': recorded_timestamps,
        'shoulder_positions': recorded_shoulder_positions,
        'elbow_positions': recorded_elbow_positions,
        'wrist_positions': recorded_wrist_positions
    }
    np.save(f'{folder}/recorded_human_position.npy', recorded_data)
    np.save(f'{folder}/optimized_robot_positions.npy', position_ergo)
    np.save(f'{folder}/optimized_joint_angles.npy', joint_history)
    np.save(f'{folder}/ergonomics_scores.npy', score_history)

    print(f"Recorded {len(recorded_timestamps)} position samples")
    print("轨迹执行完成！")

    emg_processor.read_emg_flag = False
    data_queue.join()
    for t in threads:
        t.join()

    while 1:
        interrupt = False
        time.sleep(1)