#!/usr/bin/env python3
import numpy as np
import math
import matplotlib.pyplot as plt
import utils
import transformation as tsf
from iros2025_code import main_opt_static as mos
import message_filters

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
    command = f"roslaunch {launch_file} server:=192.168.10.10"
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
    joint_pos_new = joint_pos.copy()
    if arm == 'left':
        joint_pos_new[[0, 1]] = -joint_pos[[1, 0]]
        joint_pos_new[1] = -joint_pos_new[1]
        joint_pos_new = joint_pos_new + shoulder_pos
    if arm == 'right':
        joint_pos_new[[0, 1]] = -joint_pos[[1, 0]]
        joint_pos_new = joint_pos_new + shoulder_pos
    return joint_pos_new


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


def generate_trajectory_with_speed_limit(waypoints, speed_limit, t_total, t_sample):
    num_waypoints = len(waypoints)
    if num_waypoints < 2:
        raise ValueError("需要至少两个 waypoints 来生成轨迹")

    # 平均分配时间
    t_waypoints = np.linspace(0, t_total, num_waypoints)

    # 存储最终轨迹
    full_trajectory = []

    for i in range(num_waypoints - 1):
        start = [waypoints[i], 0, 0]  # 假设初始速度和加速度为 0
        end = [waypoints[i + 1], 0, 0]  # 假设目标速度和加速度为 0
        t_start = t_waypoints[i]
        t_end = t_waypoints[i + 1]

        # 生成 minimum jerk 轨迹
        segment_trajectory = minimum_jerk_trajectory(start, end, t_start, t_end, t_sample)

        # 合并轨迹段
        if i > 0:
            # 避免重复第一个点
            full_trajectory = np.vstack((full_trajectory, segment_trajectory[1:]))
        else:
            full_trajectory = segment_trajectory

    return full_trajectory


# 计算SEF值 (Signed Ergonomics Field)
def calculate_sef(q, comfort_threshold):
    ergo_score = utils.calculate_upper_limb_score_with_joint_angles(q)
    return ergo_score - comfort_threshold


# 计算SEF梯度 (方向指向不舒适度增加的方向)
def calculate_sef_gradient(q, comfort_threshold, delta=1e-6):
    grad = np.zeros_like(q)
    sef_q = calculate_sef(q, comfort_threshold)

    for i in range(len(q)):
        q_plus = q.copy()
        q_plus[i] += delta
        sef_plus = calculate_sef(q_plus, comfort_threshold)
        grad[i] = (sef_plus - sef_q) / delta

    return grad


# 确保关节角度在限制范围内
def enforce_joint_limits(q, bounds):
    q_limited = np.copy(q)
    for i in range(len(q)):
        q_limited[i] = max(bounds[i][0], min(bounds[i][1], q[i]))
    return q_limited


# 应用动量平滑运动方向
def apply_momentum(current_direction, previous_direction, momentum=0.7):
    """应用动量以平滑方向变化"""
    if previous_direction is None or np.linalg.norm(previous_direction) < 1e-10:
        return current_direction
    return momentum * previous_direction + (1 - momentum) * current_direction


# 优化函数 - 寻找全局最优配置
def find_global_optimal_configuration(
        initial_q_l, initial_q_r, d_ual, d_lal, d_uar, d_lar,
        shoulder_l, shoulder_r, initial_distance, comfort_threshold,
        joint_angle_bounds, num_samples=500, search_radius=0.5, fine_tuning_steps=10):
    """
    使用随机搜索+局部优化的方法寻找全局最优配置
    """
    print("开始寻找全局最优配置...")
    best_q_l = initial_q_l.copy()
    best_q_r = initial_q_r.copy()

    # 计算初始的CSEF值
    initial_csef_l = calculate_sef(initial_q_l, comfort_threshold)
    initial_csef_r = calculate_sef(initial_q_r, comfort_threshold)
    best_max_csef = max(initial_csef_l, initial_csef_r)

    print(f"初始状态: 左臂CSEF={initial_csef_l:.4f}, 右臂CSEF={initial_csef_r:.4f}, 最大值={best_max_csef:.4f}")

    start_time = time.time()

    # 第1阶段: 广泛随机搜索
    for i in range(num_samples):
        if i % 100 == 0:
            elapsed = time.time() - start_time
            print(f"搜索进度: {i}/{num_samples} 样本 (耗时: {elapsed:.2f}秒)")
            print(f"当前最佳: 最大CSEF={best_max_csef:.4f}")

        # 生成随机扰动
        perturbation_l = np.random.uniform(-search_radius, search_radius, size=len(initial_q_l))
        perturbation_r = np.random.uniform(-search_radius, search_radius, size=len(initial_q_r))

        # 应用扰动并确保在限制范围内
        candidate_q_l = enforce_joint_limits(initial_q_l + perturbation_l, joint_angle_bounds)
        candidate_q_r = enforce_joint_limits(initial_q_r + perturbation_r, joint_angle_bounds)

        # 计算对应的手部位置
        _, hand_l = mos.forward_kinematics(candidate_q_l, d_ual, d_lal)
        hand_l_global = trans_shoulder2global(hand_l, shoulder_l, arm='left')

        _, hand_r = mos.forward_kinematics(candidate_q_r, d_uar, d_lar)
        hand_r_global = trans_shoulder2global(hand_r, shoulder_r, arm='right')

        # 检查距离约束
        current_distance = np.linalg.norm(hand_r_global - hand_l_global)

        # 如果距离约束满足(允许一定的误差)
        if abs(current_distance - initial_distance) < 0.05:
            # 计算CSEF值
            csef_l = calculate_sef(candidate_q_l, comfort_threshold)
            csef_r = calculate_sef(candidate_q_r, comfort_threshold)
            max_csef = max(csef_l, csef_r)

            # 如果找到更好的解
            if max_csef < best_max_csef:
                best_q_l = candidate_q_l.copy()
                best_q_r = candidate_q_r.copy()
                best_max_csef = max_csef
                print(f"找到更好的解: 最大CSEF={best_max_csef:.4f}, 左={csef_l:.4f}, 右={csef_r:.4f}")

    # 第2阶段: 基于最佳样本进行细化优化
    print("\n开始细化优化...")

    for step in range(fine_tuning_steps):
        # 逐渐减小搜索范围
        local_radius = search_radius * (1 - step / fine_tuning_steps) * 0.2

        # 在当前最佳解附近搜索
        for _ in range(50):  # 每步细化进行50次尝试
            # 生成小范围随机扰动
            perturbation_l = np.random.uniform(-local_radius, local_radius, size=len(initial_q_l))
            perturbation_r = np.random.uniform(-local_radius, local_radius, size=len(initial_q_r))

            # 应用扰动并确保在限制范围内
            candidate_q_l = enforce_joint_limits(best_q_l + perturbation_l, joint_angle_bounds)
            candidate_q_r = enforce_joint_limits(best_q_r + perturbation_r, joint_angle_bounds)

            # 计算对应的手部位置
            _, hand_l = mos.forward_kinematics(candidate_q_l, d_ual, d_lal)
            hand_l_global = trans_shoulder2global(hand_l, shoulder_l, arm='left')

            _, hand_r = mos.forward_kinematics(candidate_q_r, d_uar, d_lar)
            hand_r_global = trans_shoulder2global(hand_r, shoulder_r, arm='right')

            # 检查距离约束
            current_distance = np.linalg.norm(hand_r_global - hand_l_global)

            # 如果距离约束满足(更严格的误差)
            if abs(current_distance - initial_distance) < 0.02:
                # 计算CSEF值
                csef_l = calculate_sef(candidate_q_l, comfort_threshold)
                csef_r = calculate_sef(candidate_q_r, comfort_threshold)
                max_csef = max(csef_l, csef_r)

                # 如果找到更好的解
                if max_csef < best_max_csef:
                    best_q_l = candidate_q_l.copy()
                    best_q_r = candidate_q_r.copy()
                    best_max_csef = max_csef
                    print(f"细化阶段找到更好的解: 最大CSEF={best_max_csef:.4f}, 左={csef_l:.4f}, 右={csef_r:.4f}")

    # 计算最终状态的CSEF值
    final_csef_l = calculate_sef(best_q_l, comfort_threshold)
    final_csef_r = calculate_sef(best_q_r, comfort_threshold)

    # 验证最终距离
    _, hand_l = mos.forward_kinematics(best_q_l, d_ual, d_lal)
    hand_l_global = trans_shoulder2global(hand_l, shoulder_l, arm='left')

    _, hand_r = mos.forward_kinematics(best_q_r, d_uar, d_lar)
    hand_r_global = trans_shoulder2global(hand_r, shoulder_r, arm='right')

    final_distance = np.linalg.norm(hand_r_global - hand_l_global)

    print("\n=================== 全局优化结果 ===================")
    print(
        f"初始状态: 左臂CSEF={initial_csef_l:.4f}, 右臂CSEF={initial_csef_r:.4f}, 最大值={max(initial_csef_l, initial_csef_r):.4f}")
    print(
        f"优化后: 左臂CSEF={final_csef_l:.4f}, 右臂CSEF={final_csef_r:.4f}, 最大值={max(final_csef_l, final_csef_r):.4f}")
    print(f"CSEF最大值改善: {max(initial_csef_l, initial_csef_r) - max(final_csef_l, final_csef_r):.4f}")
    print(
        f"距离约束: 初始={initial_distance:.4f}m, 最终={final_distance:.4f}m, 误差={abs(final_distance - initial_distance):.4f}m")

    return best_q_l, best_q_r


def run_csef_dual_arm_planning(
        current_q_l, current_q_r, optimal_q_l, optimal_q_r,
        d_ual, d_lal, d_uar, d_lar,
        shoulder_l, shoulder_r, initial_distance,
        joint_angle_bounds, num_iterations=30):
    """
    使用CSEF方法进行双臂轨迹规划
    """
    print("开始进行CSEF双臂轨迹规划...")

    # 为左右臂轨迹、关节角度和分数记录历史数据
    trajectory_hand_l = []
    trajectory_elbow_l = []
    joint_history_l = []
    score_history_l = []

    trajectory_hand_r = []
    trajectory_elbow_r = []
    joint_history_r = []
    score_history_r = []

    max_csef_history = []
    csef_diff_history = []

    # 计算初始位置
    elbow_current_l, hand_current_l = mos.forward_kinematics(current_q_l, d_ual, d_lal)
    hand_current_l_global = trans_shoulder2global(hand_current_l, shoulder_l, arm='left')
    elbow_current_l_global = trans_shoulder2global(elbow_current_l, shoulder_l, arm='left')

    elbow_current_r, hand_current_r = mos.forward_kinematics(current_q_r, d_uar, d_lar)
    hand_current_r_global = trans_shoulder2global(hand_current_r, shoulder_r, arm='right')
    elbow_current_r_global = trans_shoulder2global(elbow_current_r, shoulder_r, arm='right')

    # 记录初始状态
    trajectory_hand_l.append(hand_current_l_global.copy())
    trajectory_elbow_l.append(elbow_current_l_global.copy())
    joint_history_l.append(current_q_l.copy())

    trajectory_hand_r.append(hand_current_r_global.copy())
    trajectory_elbow_r.append(elbow_current_r_global.copy())
    joint_history_r.append(current_q_r.copy())

    # 计算并记录初始CSEF值
    comfort_threshold = 0  # 舒适阈值
    csef_l = calculate_sef(current_q_l, comfort_threshold)
    csef_r = calculate_sef(current_q_r, comfort_threshold)

    score_history_l.append(utils.calculate_upper_limb_score_with_joint_angles(current_q_l))
    score_history_r.append(utils.calculate_upper_limb_score_with_joint_angles(current_q_r))
    max_csef_history.append(max(csef_l, csef_r))
    csef_diff_history.append(abs(csef_l - csef_r))

    # CSEF参数设置
    alpha = 0.8  # 目标吸引力权重 - 增加以更快地接近全局最优
    beta = 0.2  # 人体工学舒适度权重 - 减少以优先考虑接近全局最优
    step_size = 0.05  # 步长
    distance_tolerance = 0.01  # 双臂距离约束的容差

    # 动量参数
    momentum = 0.7  # 动量因子

    # 保存上一次的方向
    prev_direction_l = None
    prev_direction_r = None

    # 主循环 - 双臂轨迹生成
    for i in range(num_iterations):
        # 计算当前CSEF值
        csef_l = calculate_sef(current_q_l, comfort_threshold)
        csef_r = calculate_sef(current_q_r, comfort_threshold)
        max_csef = max(csef_l, csef_r)
        csef_diff = abs(csef_l - csef_r)

        print(f"迭代 {i + 1}: CSEF左臂={csef_l:.4f}, 右臂={csef_r:.4f}, 最大值={max_csef:.4f}, 差值={csef_diff:.4f}")

        # ----- 左臂轨迹规划 -----
        # 使用全局最优点作为目标
        goal_direction_l = optimal_q_l - current_q_l
        goal_distance_l = np.linalg.norm(goal_direction_l)

        if goal_distance_l > 0:
            goal_direction_l = goal_direction_l / goal_distance_l

        # 计算SEF梯度
        sef_gradient_l = calculate_sef_gradient(current_q_l, comfort_threshold)
        if np.linalg.norm(sef_gradient_l) > 0:
            sef_gradient_l = sef_gradient_l / np.linalg.norm(sef_gradient_l)

        # 组合梯度 - 主要关注接近全局最优点
        combined_direction_l = alpha * goal_direction_l - beta * sef_gradient_l

        if np.linalg.norm(combined_direction_l) > 0:
            combined_direction_l = combined_direction_l / np.linalg.norm(combined_direction_l)

        # 应用动量平滑轨迹
        combined_direction_l = apply_momentum(combined_direction_l, prev_direction_l, momentum)
        prev_direction_l = combined_direction_l.copy()

        # 计算下一步位置
        q_next_l = current_q_l + step_size * combined_direction_l
        q_next_l = enforce_joint_limits(q_next_l, joint_angle_bounds)

        # ----- 右臂轨迹规划 -----
        goal_direction_r = optimal_q_r - current_q_r
        goal_distance_r = np.linalg.norm(goal_direction_r)

        if goal_distance_r > 0:
            goal_direction_r = goal_direction_r / goal_distance_r

        sef_gradient_r = calculate_sef_gradient(current_q_r, comfort_threshold)
        if np.linalg.norm(sef_gradient_r) > 0:
            sef_gradient_r = sef_gradient_r / np.linalg.norm(sef_gradient_r)

        combined_direction_r = alpha * goal_direction_r - beta * sef_gradient_r

        if np.linalg.norm(combined_direction_r) > 0:
            combined_direction_r = combined_direction_r / np.linalg.norm(combined_direction_r)

        combined_direction_r = apply_momentum(combined_direction_r, prev_direction_r, momentum)
        prev_direction_r = combined_direction_r.copy()

        q_next_r = current_q_r + step_size * combined_direction_r
        q_next_r = enforce_joint_limits(q_next_r, joint_angle_bounds)

        # 使用前向运动学计算新的位置
        new_elbow_l, new_hand_l = mos.forward_kinematics(q_next_l, d_ual, d_lal)
        new_hand_l_global = trans_shoulder2global(new_hand_l, shoulder_l, arm='left')
        new_elbow_l_global = trans_shoulder2global(new_elbow_l, shoulder_l, arm='left')

        new_elbow_r, new_hand_r = mos.forward_kinematics(q_next_r, d_uar, d_lar)
        new_hand_r_global = trans_shoulder2global(new_hand_r, shoulder_r, arm='right')
        new_elbow_r_global = trans_shoulder2global(new_elbow_r, shoulder_r, arm='right')

        # 检查双臂距离约束
        current_distance = np.linalg.norm(new_hand_r_global - new_hand_l_global)

        # 如果距离违反约束，调整位置
        if abs(current_distance - initial_distance) > distance_tolerance:
            # 计算从左手到右手的单位向量
            direction_vec = (new_hand_r_global - new_hand_l_global) / current_distance

            # 需要的调整量
            adjustment = (current_distance - initial_distance) / 2

            # 调整双手位置以满足约束
            adjusted_hand_l_global = new_hand_l_global + adjustment * direction_vec
            adjusted_hand_r_global = new_hand_r_global - adjustment * direction_vec

            # 将全局坐标转回肩膀坐标系
            adjusted_elbow_l, adjusted_hand_l = trans_global2shoulder(shoulder_l, new_elbow_l_global,
                                                                      adjusted_hand_l_global, arm='left')
            adjusted_elbow_r, adjusted_hand_r = trans_global2shoulder(shoulder_r, new_elbow_r_global,
                                                                      adjusted_hand_r_global, arm='right')

            # 使用逆运动学重新计算关节角度
            q_next_l = mos.inverse_kinematics(adjusted_elbow_l, adjusted_hand_l, d_ual, d_lal)
            q_next_r = mos.inverse_kinematics(adjusted_elbow_r, adjusted_hand_r, d_uar, d_lar)

            # 重新计算调整后的位置
            new_elbow_l, new_hand_l = mos.forward_kinematics(q_next_l, d_ual, d_lal)
            new_hand_l_global = trans_shoulder2global(new_hand_l, shoulder_l, arm='left')
            new_elbow_l_global = trans_shoulder2global(new_elbow_l, shoulder_l, arm='left')

            new_elbow_r, new_hand_r = mos.forward_kinematics(q_next_r, d_uar, d_lar)
            new_hand_r_global = trans_shoulder2global(new_hand_r, shoulder_r, arm='right')
            new_elbow_r_global = trans_shoulder2global(new_elbow_r, shoulder_r, arm='right')

            # 验证调整后的距离
            adjusted_distance = np.linalg.norm(new_hand_r_global - new_hand_l_global)
            print(f"距离调整: {current_distance:.4f} → {adjusted_distance:.4f} (目标: {initial_distance:.4f})")

        # 更新当前关节角度和位置
        current_q_l = q_next_l
        current_q_r = q_next_r

        # 记录轨迹
        trajectory_hand_l.append(new_hand_l_global.copy())
        trajectory_elbow_l.append(new_elbow_l_global.copy())
        joint_history_l.append(current_q_l.copy())

        trajectory_hand_r.append(new_hand_r_global.copy())
        trajectory_elbow_r.append(new_elbow_r_global.copy())
        joint_history_r.append(current_q_r.copy())

        # 记录分数
        score_l = utils.calculate_upper_limb_score_with_joint_angles(current_q_l)
        score_r = utils.calculate_upper_limb_score_with_joint_angles(current_q_r)
        score_history_l.append(score_l)
        score_history_r.append(score_r)

        # 更新CSEF相关历史记录
        csef_l = calculate_sef(current_q_l, comfort_threshold)
        csef_r = calculate_sef(current_q_r, comfort_threshold)
        max_csef_history.append(max(csef_l, csef_r))
        csef_diff_history.append(abs(csef_l - csef_r))

        # 自适应步长：接近目标时减小
        goal_distance = min(goal_distance_l, goal_distance_r)
        step_size = max(0.02, min(0.05, goal_distance * 0.2))

        print(
            f"迭代 {i + 1}/{num_iterations}. 左臂得分: {score_l:.4f}, 右臂得分: {score_r:.4f}, 最大CSEF: {max_csef_history[-1]:.4f}")

    results = {
        'trajectory_hand_l': trajectory_hand_l,
        'trajectory_elbow_l': trajectory_elbow_l,
        'joint_history_l': joint_history_l,
        'score_history_l': score_history_l,

        'trajectory_hand_r': trajectory_hand_r,
        'trajectory_elbow_r': trajectory_elbow_r,
        'joint_history_r': joint_history_r,
        'score_history_r': score_history_r,

        'max_csef_history': max_csef_history,
        'csef_diff_history': csef_diff_history
    }

    return results


def run_iterations(num_iterations):
    global current_q_l, current_q_r, global_positions
    global trajectory_hand_l, trajectory_elbow_l, trajectory_hand_r, trajectory_elbow_r
    global score_history_l, joint_history_l, score_history_r, joint_history_r
    global max_csef_history, csef_diff_history

    # 2. 使用CSEF方法生成从当前状态到全局最优状态的轨迹
    print("基于全局最优配置规划CSEF轨迹...")

    results = run_csef_dual_arm_planning(
        current_q_l, current_q_r, optimal_q_l, optimal_q_r,
        d_ual, d_lal, d_uar, d_lar,
        shoulder_l, shoulder_r, initial_distance,
        joint_angle_bounds, num_iterations)

    # 提取结果
    trajectory_hand_l = results['trajectory_hand_l']
    trajectory_elbow_l = results['trajectory_elbow_l']
    joint_history_l = results['joint_history_l']
    score_history_l = results['score_history_l']

    trajectory_hand_r = results['trajectory_hand_r']
    trajectory_elbow_r = results['trajectory_elbow_r']
    joint_history_r = results['joint_history_r']
    score_history_r = results['score_history_r']

    max_csef_history = results['max_csef_history']
    csef_diff_history = results['csef_diff_history']

    # 更新全局位置 - 取最终位置
    current_q_l = joint_history_l[-1].copy()
    current_q_r = joint_history_r[-1].copy()

    global_positions[8] = trajectory_hand_l[-1].copy()
    global_positions[7] = trajectory_elbow_l[-1].copy()
    global_positions[5] = trajectory_hand_r[-1].copy()
    global_positions[4] = trajectory_elbow_r[-1].copy()

    # 3. 绘制最终轨迹和结果
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    ax.set_xlim((0.7, 2.2))
    ax.set_ylim((-0.7, 0.8))
    ax.set_zlim((0.0, 1.5))
    ax.view_init(elev=30, azim=-30)

    # 绘制左臂
    ax.scatter(shoulder_l[0], shoulder_l[1], shoulder_l[2], c='black', s=50)
    ax.scatter(trajectory_elbow_l[-1][0], trajectory_elbow_l[-1][1], trajectory_elbow_l[-1][2], c='blue', s=50)
    ax.scatter(trajectory_hand_l[-1][0], trajectory_hand_l[-1][1], trajectory_hand_l[-1][2], c='green', s=50,
               label='Left Hand')
    ax.plot([shoulder_l[0], trajectory_elbow_l[-1][0], trajectory_hand_l[-1][0]],
            [shoulder_l[1], trajectory_elbow_l[-1][1], trajectory_hand_l[-1][1]],
            [shoulder_l[2], trajectory_elbow_l[-1][2], trajectory_hand_l[-1][2]], c='blue', linewidth=2)

    # 绘制右臂
    ax.scatter(shoulder_r[0], shoulder_r[1], shoulder_r[2], c='black', s=50)
    ax.scatter(trajectory_elbow_r[-1][0], trajectory_elbow_r[-1][1], trajectory_elbow_r[-1][2], c='red', s=50)
    ax.scatter(trajectory_hand_r[-1][0], trajectory_hand_r[-1][1], trajectory_hand_r[-1][2], c='magenta', s=50,
               label='Right Hand')
    ax.plot([shoulder_r[0], trajectory_elbow_r[-1][0], trajectory_hand_r[-1][0]],
            [shoulder_r[1], trajectory_elbow_r[-1][1], trajectory_hand_r[-1][1]],
            [shoulder_r[2], trajectory_elbow_r[-1][2], trajectory_hand_r[-1][2]], c='red', linewidth=2)

    # 绘制轨迹
    traj_l = np.array(trajectory_hand_l)
    ax.plot(traj_l[:, 0], traj_l[:, 1], traj_l[:, 2], c='cyan', linestyle=':')
    traj_r = np.array(trajectory_hand_r)
    ax.plot(traj_r[:, 0], traj_r[:, 1], traj_r[:, 2], c='pink', linestyle=':')

    # 绘制全局最优位置
    ax.scatter(optimal_hand_l_global[0], optimal_hand_l_global[1], optimal_hand_l_global[2],
               c='darkblue', s=50, marker='*', label='Left Optimal')
    ax.scatter(optimal_hand_r_global[0], optimal_hand_r_global[1], optimal_hand_r_global[2],
               c='darkred', s=50, marker='*', label='Right Optimal')

    # 绘制双手之间的连线，表示距离约束
    ax.plot([trajectory_hand_l[-1][0], trajectory_hand_r[-1][0]],
            [trajectory_hand_l[-1][1], trajectory_hand_r[-1][1]],
            [trajectory_hand_l[-1][2], trajectory_hand_r[-1][2]],
            c='green', linestyle='--', linewidth=1,
            label=f'Distance: {np.linalg.norm(trajectory_hand_r[-1] - trajectory_hand_l[-1]):.3f}m')

    # 绘制骨架
    utils.plot_skeleton(ax, global_positions, skeleton_parent_indices, color='black')

    # 设置标题和标签
    final_csef_l = calculate_sef(current_q_l, 0)
    final_csef_r = calculate_sef(current_q_r, 0)
    ax.set_title(
        f'Final Result: Left CSEF={final_csef_l:.3f}, Right CSEF={final_csef_r:.3f}, Max={max(final_csef_l, final_csef_r):.3f}')
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.set_zlabel('Z Position (m)')
    ax.legend(loc='upper left', fontsize='small')

    plt.show()

    # 绘制分数历史
    iterations = np.arange(0, len(score_history_l))
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, score_history_l, marker='o', linestyle='-', label='Left Arm Score')
    plt.plot(iterations, score_history_r, marker='x', linestyle='-', label='Right Arm Score')
    plt.plot(iterations, max_csef_history, marker='d', linestyle='-.', label='Max CSEF')
    plt.plot(iterations, csef_diff_history, marker='s', linestyle=':', label='CSEF Difference')

    # 添加全局最优参考线
    optimal_csef_l = calculate_sef(optimal_q_l, 0)
    optimal_csef_r = calculate_sef(optimal_q_r, 0)
    optimal_max = max(optimal_csef_l, optimal_csef_r)
    plt.axhline(y=optimal_max, color='r', linestyle='--', label=f'Global Optimal Max CSEF: {optimal_max:.3f}')

    plt.xlabel('Iteration')
    plt.ylabel('Ergonomic Score / CSEF Value')
    plt.title('Ergonomic Scores and CSEF Values over Iterations')
    plt.legend()
    plt.grid(True)
    plt.show()

    # 返回最终轨迹
    return trajectory_hand_l, trajectory_hand_r, joint_history_l, joint_history_r, score_history_l, score_history_r


if __name__ == '__main__':
    rospy.init_node('csef_hrc')
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
    robot_left_position_init = np.array([1.0, 0.15, 1.2])
    robot_right_position_init = np.array([0.9, -0.25, 0.7])

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
    subscriber_shouL = rospy.wait_for_message('/vrpn_client_node/shouL/pose', PoseStamped)
    subscriber_shouR = rospy.wait_for_message('/vrpn_client_node/shouR/pose', PoseStamped)
    subscriber_elbowL = rospy.wait_for_message('/vrpn_client_node/elbowL/pose', PoseStamped)
    subscriber_elbowR = rospy.wait_for_message('/vrpn_client_node/elbowR/pose', PoseStamped)
    subscriber_wristL = rospy.wait_for_message('/vrpn_client_node/wristL/pose', PoseStamped)
    subscriber_wristR = rospy.wait_for_message('/vrpn_client_node/wristR/pose', PoseStamped)
    print("collecting human data successfully!")

    sub_robot = transform_to_pose(subscriber_robot)
    sub_shouL = transform_to_pose(subscriber_shouL)
    sub_shouR = transform_to_pose(subscriber_shouR)
    sub_elbowL = transform_to_pose(subscriber_elbowL)
    sub_elbowR = transform_to_pose(subscriber_elbowR)
    sub_wristL = transform_to_pose(subscriber_wristL)
    sub_wristR = transform_to_pose(subscriber_wristR)

    T_optitrack2robotbase = np.linalg.inv(
        tsf.transform_optitrack_origin_to_optitrack_robot(
            sub_robot) @ tsf.transform_optitrack_robot_to_robot_base())
    shouL_position_init = T_optitrack2robotbase[:3, :3] @ sub_shouL[:3] + T_optitrack2robotbase[:3, 3]
    shouR_position_init = T_optitrack2robotbase[:3, :3] @ sub_shouR[:3] + T_optitrack2robotbase[:3, 3]
    elbowL_position_init = T_optitrack2robotbase[:3, :3] @ sub_elbowL[:3] + T_optitrack2robotbase[:3, 3]
    elbowR_position_init = T_optitrack2robotbase[:3, :3] @ sub_elbowR[:3] + T_optitrack2robotbase[:3, 3]
    wristL_position_init = T_optitrack2robotbase[:3, :3] @ sub_wristL[:3] + T_optitrack2robotbase[:3, 3]
    wristR_position_init = T_optitrack2robotbase[:3, :3] @ sub_wristR[:3] + T_optitrack2robotbase[:3, 3]

    joint_angle_bounds = [
        (-math.pi / 18, 17 * math.pi / 18),  # Joint 1
        (-math.pi / 18, 17 * math.pi / 18),  # Joint 2
        (-np.pi / 3, np.pi / 2),  # Joint 3
        (-np.pi / 2, np.pi / 3)  # Joint 4
    ]
    optimal_q = [0, 0, 0, -math.pi / 6]

    skeleton_joint_name, skeleton_joints, skeleton_parent_indices, skeleton_joint_local_translation = \
        utils.read_skeleton_motion('/home/clover/Chenzui/Ergo-Manip/data/demo_2_test_chenzui_only_optitrack2hotu.npy')
    skeleton_joint = skeleton_joints[500, :]
    global_positions, global_rotations = utils.forward_kinematics(skeleton_joint_local_translation,
                                                                  skeleton_joint, skeleton_parent_indices)
    global_positions[:, 2] = global_positions[:, 2] * 1.2

    global_positions[4] = global_positions[3] + (elbowR_position_init - shouR_position_init)
    global_positions[7] = global_positions[6] + (elbowL_position_init - shouL_position_init)
    global_positions[5] = global_positions[3] + (wristR_position_init - shouR_position_init)
    global_positions[8] = global_positions[6] + (wristL_position_init - shouL_position_init)

    shou_center = (shouL_position_init + shouR_position_init) / 2
    global_positions = global_positions + np.array([shou_center[0], shou_center[1], 0])

    # 初始位置
    initial_position_l = global_positions[8]
    initial_position_r = global_positions[5]

    # 计算初始双手距离
    initial_distance = np.linalg.norm(initial_position_r - initial_position_l)
    print(f"初始双手距离: {initial_distance:.4f} 米")

    # Body dimensions - 左右臂
    d_ual, d_uar, d_lal, d_lar = mos.calculate_arm_dimensions(
        shouL_position_init, elbowL_position_init, wristL_position_init,
        shouR_position_init, elbowR_position_init, wristR_position_init)

    # 提取肩部位置
    shoulder_l = global_positions[6].copy()
    shoulder_r = global_positions[3].copy()

    # 计算初始关节角度
    p_elbowL_init, p_wristL_init = trans_global2shoulder(
        shoulder_l, global_positions[7], global_positions[8], arm='left')
    p_elbowR_init, p_wristR_init = trans_global2shoulder(
        shoulder_r, global_positions[4], global_positions[5], arm='right')

    current_q_l = mos.inverse_kinematics(p_elbowL_init, p_wristL_init, d_ual, d_lal)
    current_q_r = mos.inverse_kinematics(p_elbowR_init, p_wristR_init, d_uar, d_lar)

    # 1. 首先找到全局最优配置
    optimal_q_l, optimal_q_r = find_global_optimal_configuration(
        current_q_l, current_q_r, d_ual, d_lal, d_uar, d_lar,
        shoulder_l, shoulder_r, initial_distance, comfort_threshold=0,
        joint_angle_bounds=joint_angle_bounds, num_samples=200, fine_tuning_steps=5)

    # 计算全局最优对应的位置（用于可视化）
    _, optimal_hand_l = mos.forward_kinematics(optimal_q_l, d_ual, d_lal)
    optimal_hand_l_global = trans_shoulder2global(optimal_hand_l, shoulder_l, arm='left')

    _, optimal_hand_r = mos.forward_kinematics(optimal_q_r, d_uar, d_lar)
    optimal_hand_r_global = trans_shoulder2global(optimal_hand_r, shoulder_r, arm='right')

    # 为两臂轨迹记录数据结构
    trajectory_hand_l = [global_positions[8].copy()]
    trajectory_elbow_l = [global_positions[7].copy()]
    joint_history_l = [current_q_l.copy()]
    score_history_l = [utils.calculate_upper_limb_score_with_joint_angles(current_q_l)]

    trajectory_hand_r = [global_positions[5].copy()]
    trajectory_elbow_r = [global_positions[4].copy()]
    joint_history_r = [current_q_r.copy()]
    score_history_r = [utils.calculate_upper_limb_score_with_joint_angles(current_q_r)]

    max_csef_history = [max(score_history_l[0], score_history_r[0])]
    csef_diff_history = [abs(score_history_l[0] - score_history_r[0])]

    # 设置优化参数
    num_samples_per_joint = 15
    comp_factor = 0.1
    num_iterations = 30
    max_disp = 0.03  # maximum allowed displacement per iteration in global (hand) space

    # 运行CSEF双臂规划
    trajectory_hand_l, trajectory_hand_r, joint_history_l, joint_history_r, score_history_l, score_history_r = run_iterations(
        num_iterations)

    # 在这里添加你想在迭代完成后执行的代码
    print("双臂CSEF规划完成. 继续执行下一步...")

    # 生成轨迹 - 左臂
    waypoints_ergo_l = trajectory_hand_l
    waypoints_straight_l = [trajectory_hand_l[0], trajectory_hand_l[-1]]

    # 生成轨迹 - 右臂
    waypoints_ergo_r = trajectory_hand_r
    waypoints_straight_r = [trajectory_hand_r[0], trajectory_hand_r[-1]]

    speed_limit = 0.05  # 最大速度限制
    t_total = 8  # 总时间
    t_sample = 0.0025  # 采样时间间隔 (1000 Hz)

    # 生成轨迹 - 这里使用人体工学轨迹而不是直线轨迹
    trajectory_ergo_l = generate_trajectory_with_speed_limit(waypoints_ergo_l, speed_limit, t_total, t_sample)
    trajectory_ergo_r = generate_trajectory_with_speed_limit(waypoints_ergo_r, speed_limit, t_total, t_sample)

    # 绘制轨迹
    plt.figure()
    plt.plot(trajectory_ergo_l[:, 0], label='Left Arm X')
    plt.plot(trajectory_ergo_r[:, 0], label='Right Arm X')
    plt.legend()
    plt.title('Trajectory Positions (X axis)')
    plt.show()

    # 标准化位置
    position_l = trajectory_ergo_l[:, 0] - trajectory_ergo_l[0, 0]
    position_r = trajectory_ergo_r[:, 0] - trajectory_ergo_r[0, 0]

    print("left_arm_current", curi.get_tcp(0))
    print("right_arm_current", curi.get_tcp(1))

    for i in range(len(position_l)):
        robot_left_position = robot_left_position_init + position_l[i]
        robot_right_position = robot_right_position_init + position_r[i]

        robot_left_pose_matrix = np.r_[
            np.c_[robot_left_rotation_matrix_init, robot_left_position.T], np.array([[0, 0, 0, 1]])]
        robot_right_pose_matrix = np.r_[
            np.c_[robot_right_rotation_matrix_init, robot_right_position.T], np.array([[0, 0, 0, 1]])]

        robot_left_pose_matrix = base2torso_matrix @ robot_left_pose_matrix
        robot_right_pose_matrix = base2torso_matrix @ robot_right_pose_matrix

        curi.set_tcp_servo(robot_left_pose_matrix, robot_right_pose_matrix)
        time.sleep(0.001)

    while 1:
        interrupt = False
        time.sleep(1)