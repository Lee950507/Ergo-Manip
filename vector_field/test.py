#!/usr/bin/env python3
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from itertools import product
from matplotlib.colors import Normalize
import time


# 模拟前向运动学函数
def forward_kinematics(q, d_ua, d_la):
    T1 = np.array([[math.cos(q[0]), 0, math.sin(q[0])], [0, 1, 0], [-math.sin(q[0]), 0, math.cos(q[0])]])
    T2 = np.array([[1, 0, 0], [0, math.cos(q[1]), -math.sin(q[1])], [0, math.sin(q[1]), math.cos(q[1])]])
    T3 = np.array([[math.cos(q[2]), -math.sin(q[2]), 0], [math.sin(q[2]), math.cos(q[2]), 0], [0, 0, 1]])
    T4 = np.array([[1, 0, 0], [0, math.cos(q[3]), -math.sin(q[3])], [0, math.sin(q[3]), math.cos(q[3])]])
    p_elbow = T1 @ T2 @ T3 @ d_ua
    p_hand = T1 @ T2 @ T3 @ (d_ua + T4 @ d_la)
    return p_elbow, p_hand


# 模拟逆向运动学函数
def inverse_kinematics(p_elbow, p_hand, d_ua, d_la):
    # 简化版本的逆向运动学求解
    # 实际应用中可能需要更复杂的算法
    return np.array([0.2, 0.3, 0.1, -0.5])  # 返回一个示例关节角配置


# 模拟人体工学评分函数
def calculate_upper_limb_score_with_joint_angles(q):
    # 简化的人体工学评分计算
    # 在实际应用中，这可能是一个更复杂的函数
    # 这里我们使用加权平方和作为示例
    weights = [1.2, 1.0, 0.8, 1.5]
    optimal = [0, 0, 0, -math.pi / 6]
    score = sum(w * (q[i] - optimal[i]) ** 2 for i, w in enumerate(weights))
    return score


# 模拟骨架绘制函数
def plot_skeleton(ax, positions, parent_indices, color='black'):
    for i, parent in enumerate(parent_indices):
        if parent >= 0:  # 如果有父节点
            ax.plot([positions[parent, 0], positions[i, 0]],
                    [positions[parent, 1], positions[i, 1]],
                    [positions[parent, 2], positions[i, 2]], color=color)


# 坐标转换函数
def trans_shoulder2global(joint_pos, shoulder_pos, arm='right'):
    if arm == 'left':
        joint_pos_new = joint_pos.copy()
        joint_pos_new[[0, 1]] = -joint_pos[[1, 0]]
        joint_pos_new[1] = -joint_pos_new[1]
        joint_pos_new = joint_pos_new + shoulder_pos
        return joint_pos_new
    if arm == 'right':
        joint_pos_new = joint_pos.copy()
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


# 压缩关节角度边界函数
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


# CSEF算法实现的run_iterations函数
def run_iterations(num_iterations):
    global current_q, global_positions, shoulder, d_uar, d_lar, optimal_q, joint_angle_bounds
    global trajectory_hand, trajectory_elbow, score_history, joint_history, ax

    # 定义CSEF场所需参数
    q_opt = optimal_q  # 最优关节角度配置
    weights = np.array([1.0, 1.5, 1.0, 2.0])  # 关节角度权重
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
        ergo_score = calculate_upper_limb_score_with_joint_angles(q)
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
        new_elbow, new_hand = forward_kinematics(q_next, d_uar, d_lar)
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

    plot_skeleton(ax, global_positions, skeleton_parent_indices, color='black')

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


def main():
    global current_q, global_positions, shoulder, d_uar, d_lar, optimal_q, joint_angle_bounds
    global trajectory_hand, trajectory_elbow, score_history, joint_history, ax, optimal_position
    global skeleton_parent_indices

    # 创建测试数据
    print("初始化测试数据...")

    # 定义关节角度边界
    joint_angle_bounds = [
        (-math.pi / 18, 17 * math.pi / 18),  # Joint 1
        (-math.pi / 18, 17 * math.pi / 18),  # Joint 2
        (-np.pi / 3, np.pi / 2),  # Joint 3
        (-np.pi / 2, np.pi / 3)  # Joint 4
    ]

    # 定义最优关节角度配置
    optimal_q = np.array([0, 0, 0, -math.pi / 6])

    # 定义当前关节角度配置（初始值）
    current_q = np.array([1.2, 0.3, 1.1, -1.5])

    # 定义手臂尺寸参数
    d_uar = np.array([0, 0, -0.3])  # Upper arm vector (shoulder to elbow)
    d_lar = np.array([0, 0.25, 0])  # Lower arm vector (elbow to hand)

    # 定义肩部位置
    shoulder = np.array([1.0, 0.0, 1.0])

    # 计算初始的肘部和手部位置
    elbow_init, hand_init = forward_kinematics(current_q, d_uar, d_lar)
    elbow_init = trans_shoulder2global(elbow_init, shoulder, arm='right')
    hand_init = trans_shoulder2global(hand_init, shoulder, arm='right')

    # 计算最优位置
    _, optimal_hand = forward_kinematics(optimal_q, d_uar, d_lar)
    optimal_position = trans_shoulder2global(optimal_hand, shoulder, arm='right')

    # 创建全局位置数组（模拟骨架）
    global_positions = np.zeros((10, 3))  # 假设有10个关节点
    global_positions[3] = shoulder  # 肩部
    global_positions[4] = elbow_init  # 肘部
    global_positions[5] = hand_init  # 手部

    # 定义骨架的父子关系（用于绘图）
    skeleton_parent_indices = [-1, 0, 1, 2, 3, 4, 2, 6, 7, 2]  # 示例父索引

    # 初始化轨迹和历史记录
    trajectory_hand = [hand_init.copy()]
    trajectory_elbow = [elbow_init.copy()]
    score_history = []
    joint_history = [current_q.copy()]

    # 创建图形对象
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 执行CSEF算法
    print("执行CSEF算法优化轨迹...")
    num_iterations = 40  # 迭代次数
    trajectory_hand, trajectory_elbow, joint_history, score_history = run_iterations(num_iterations)

    print("算法执行完毕!")
    print(f"最终人体工学得分: {score_history[-1]:.4f}")

    # 分析结果
    print("\n轨迹分析:")
    print(f"- 起始位置: {trajectory_hand[0]}")
    print(f"- 终止位置: {trajectory_hand[-1]}")
    print(f"- 轨迹长度: {len(trajectory_hand)}")

    print("\n关节角度历史:")
    for i, q in enumerate(joint_history):
        if i % 5 == 0 or i == len(joint_history) - 1:  # 每5步打印一次
            print(f"Step {i}: {q}")


if __name__ == "__main__":
    main()