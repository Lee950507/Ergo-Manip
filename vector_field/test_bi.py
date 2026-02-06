#!/usr/bin/env python3
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import utils
from iros2025_code import main_opt_static as mos
import time


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
        joint_angle_bounds, num_samples=1000, search_radius=0.5, fine_tuning_steps=20):
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


def main():
    # 定义关节角度界限
    joint_angle_bounds = [
        (-math.pi / 18, 17 * math.pi / 18),  # Joint 1
        (-math.pi / 18, 17 * math.pi / 18),  # Joint 2
        (-np.pi / 3, np.pi / 2),  # Joint 3
        (-np.pi / 2, np.pi / 3)  # Joint 4
    ]

    # 读取骨架数据
    skeleton_joint_name, skeleton_joints, skeleton_parent_indices, skeleton_joint_local_translation = \
        utils.read_skeleton_motion('/home/ubuntu/Ergo-Manip/data/demo_2_test_chenzui_only_optitrack2hotu.npy')
    skeleton_joint = skeleton_joints[400, :]
    global_positions, global_rotations = utils.forward_kinematics(skeleton_joint_local_translation,
                                                                  skeleton_joint, skeleton_parent_indices)
    # 计算上肢长度（用于正向运动学）
    d_ual, d_uar, d_lal, d_lar = mos.calculate_arm_dimensions(global_positions[6], global_positions[7],
                                                              global_positions[8], global_positions[3],
                                                              global_positions[4], global_positions[5])

    # 计算初始关节角度
    p_elbowL_init, p_wristL_init = trans_global2shoulder(global_positions[6], global_positions[7], global_positions[8],
                                                         arm='left')
    p_elbowR_init, p_wristR_init = trans_global2shoulder(global_positions[3], global_positions[4], global_positions[5],
                                                         arm='right')

    current_q_l = mos.inverse_kinematics(p_elbowL_init, p_wristL_init, d_ual, d_lal)
    current_q_r = mos.inverse_kinematics(p_elbowR_init, p_wristR_init, d_uar, d_lar)

    # 记录初始位置
    hand_current_l = global_positions[8].copy()
    elbow_current_l = global_positions[7].copy()
    hand_current_r = global_positions[5].copy()
    elbow_current_r = global_positions[4].copy()

    shoulder_l = global_positions[6].copy()
    shoulder_r = global_positions[3].copy()

    # 计算并记录初始双手距离约束
    initial_distance = np.linalg.norm(hand_current_r - hand_current_l)
    print(f"初始双手距离: {initial_distance:.4f} 米")

    # 为动画记录历史轨迹
    trajectory_hand_l = [hand_current_l.copy()]
    trajectory_elbow_l = [elbow_current_l.copy()]
    trajectory_hand_r = [hand_current_r.copy()]
    trajectory_elbow_r = [elbow_current_r.copy()]

    # 计算初始CSEF值
    comfort_threshold = 0  # 舒适阈值
    csef_l = calculate_sef(current_q_l, comfort_threshold)
    csef_r = calculate_sef(current_q_r, comfort_threshold)

    score_history_l = [utils.calculate_upper_limb_score_with_joint_angles(current_q_l)]
    joint_history_l = [current_q_l.copy()]
    score_history_r = [utils.calculate_upper_limb_score_with_joint_angles(current_q_r)]
    joint_history_r = [current_q_r.copy()]
    max_csef_history = [max(csef_l, csef_r)]
    csef_diff_history = [abs(csef_l - csef_r)]

    # 执行全局优化，寻找最优配置
    optimal_q_l, optimal_q_r = find_global_optimal_configuration(
        current_q_l, current_q_r, d_ual, d_lal, d_uar, d_lar,
        shoulder_l, shoulder_r, initial_distance, comfort_threshold,
        joint_angle_bounds, num_samples=500, search_radius=0.5, fine_tuning_steps=10)

    # 计算最优配置对应的位置（用于可视化）
    _, optimal_hand_l = mos.forward_kinematics(optimal_q_l, d_ual, d_lal)
    optimal_hand_l_global = trans_shoulder2global(optimal_hand_l, shoulder_l, arm='left')

    _, optimal_hand_r = mos.forward_kinematics(optimal_q_r, d_uar, d_lar)
    optimal_hand_r_global = trans_shoulder2global(optimal_hand_r, shoulder_r, arm='right')

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

    # 迭代次数
    num_iterations = 30

    # 创建图形和三维坐标轴
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # ---------------------- 迭代更新函数 ----------------------
    def update(frame):
        nonlocal current_q_l, current_q_r, global_positions, step_size
        nonlocal trajectory_hand_l, trajectory_elbow_l, trajectory_hand_r, trajectory_elbow_r
        nonlocal score_history_l, joint_history_l, score_history_r, joint_history_r, max_csef_history, csef_diff_history
        nonlocal prev_direction_l, prev_direction_r

        ax.clear()
        ax.set_xlim((-1.2, 0.3))
        ax.set_ylim((-0.7, 0.8))
        ax.set_zlim((0.0, 1.5))
        ax.view_init(elev=30, azim=-30)

        # 计算当前左右臂的CSEF值
        csef_l = calculate_sef(current_q_l, comfort_threshold)
        csef_r = calculate_sef(current_q_r, comfort_threshold)
        max_csef = max(csef_l, csef_r)
        csef_diff = abs(csef_l - csef_r)

        print(
            f"迭代 {frame + 1}: CSEF左臂={csef_l:.4f}, 右臂={csef_r:.4f}, 最大值={max_csef:.4f}, 差值={csef_diff:.4f}")

        # ----- 左臂轨迹规划 -----
        # 使用全局最优点作为目标，而不是预定义的最优点
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

        global_positions[8] = new_hand_l_global
        global_positions[7] = new_elbow_l_global
        global_positions[5] = new_hand_r_global
        global_positions[4] = new_elbow_r_global

        # 添加到轨迹
        trajectory_hand_l.append(new_hand_l_global.copy())
        trajectory_elbow_l.append(new_elbow_l_global.copy())
        trajectory_hand_r.append(new_hand_r_global.copy())
        trajectory_elbow_r.append(new_elbow_r_global.copy())

        # 记录关节角度和得分
        joint_history_l.append(current_q_l.copy())
        joint_history_r.append(current_q_r.copy())
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
            f"迭代 {frame + 1}/{num_iterations}. 左臂得分: {score_l:.4f}, 右臂得分: {score_r:.4f}, 最大CSEF: {max_csef_history[-1]:.4f}")

        # ---------------------- 绘制部分 ----------------------
        # 绘制左臂
        ax.scatter(shoulder_l[0], shoulder_l[1], shoulder_l[2], c='black', s=50)
        ax.scatter(new_elbow_l_global[0], new_elbow_l_global[1], new_elbow_l_global[2], c='blue', s=50)
        ax.scatter(new_hand_l_global[0], new_hand_l_global[1], new_hand_l_global[2], c='green', s=50, label='Left Hand')
        ax.plot([shoulder_l[0], new_elbow_l_global[0], new_hand_l_global[0]],
                [shoulder_l[1], new_elbow_l_global[1], new_hand_l_global[1]],
                [shoulder_l[2], new_elbow_l_global[2], new_hand_l_global[2]], c='blue', linewidth=2)

        # 绘制右臂
        ax.scatter(shoulder_r[0], shoulder_r[1], shoulder_r[2], c='black', s=50)
        ax.scatter(new_elbow_r_global[0], new_elbow_r_global[1], new_elbow_r_global[2], c='red', s=50)
        ax.scatter(new_hand_r_global[0], new_hand_r_global[1], new_hand_r_global[2], c='magenta', s=50,
                   label='Right Hand')
        ax.plot([shoulder_r[0], new_elbow_r_global[0], new_hand_r_global[0]],
                [shoulder_r[1], new_elbow_r_global[1], new_hand_r_global[1]],
                [shoulder_r[2], new_elbow_r_global[2], new_hand_r_global[2]], c='red', linewidth=2)

        # 绘制双手之间的连线，表示距离约束
        ax.plot([new_hand_l_global[0], new_hand_r_global[0]],
                [new_hand_l_global[1], new_hand_r_global[1]],
                [new_hand_l_global[2], new_hand_r_global[2]],
                c='green', linestyle='--', linewidth=1,
                label=f'Distance: {np.linalg.norm(new_hand_r_global - new_hand_l_global):.3f}m')

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

        # 绘制骨架
        utils.plot_skeleton(ax, global_positions, skeleton_parent_indices, color='black')

        # 在图中添加当前CSEF信息
        title_text = f"CSEF Dual-Arm Planning - Iteration {frame + 1}\n"
        title_text += f"Left CSEF: {csef_l:.3f}, Right CSEF: {csef_r:.3f}, Max: {max(csef_l, csef_r):.3f}, Diff: {abs(csef_l - csef_r):.3f}"

        # 计算接近全局最优的程度
        optimal_csef_l = calculate_sef(optimal_q_l, comfort_threshold)
        optimal_csef_r = calculate_sef(optimal_q_r, comfort_threshold)
        optimal_max = max(optimal_csef_l, optimal_csef_r)
        progress = 1.0 - (max_csef - optimal_max) / (max_csef_history[0] - optimal_max) if max_csef_history[
                                                                                               0] > optimal_max else 1.0
        title_text += f"\nProgress to Global Optimum: {progress * 100:.1f}%"

        ax.set_title(title_text)
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.set_zlabel('Z Position (m)')
        ax.legend(loc='upper left', fontsize='small')

    # 制作动画
    anim = FuncAnimation(fig, update, frames=num_iterations, interval=800, repeat=False)
    writer = PillowWriter(fps=2)
    anim.save("1.gif", writer=writer)
    plt.show()

    # 绘制得分历史
    iterations = np.arange(0, len(score_history_l))
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, score_history_l, marker='o', linestyle='-', label='Left Arm Score')
    plt.plot(iterations, score_history_r, marker='x', linestyle='-', label='Right Arm Score')
    plt.plot(iterations, max_csef_history, marker='d', linestyle='-.', label='Max CSEF')
    plt.plot(iterations, csef_diff_history, marker='s', linestyle=':', label='CSEF Difference')

    # 添加全局最优参考线
    optimal_csef_l = calculate_sef(optimal_q_l, comfort_threshold)
    optimal_csef_r = calculate_sef(optimal_q_r, comfort_threshold)
    optimal_max = max(optimal_csef_l, optimal_csef_r)
    plt.axhline(y=optimal_max, color='r', linestyle='--', label=f'Global Optimal Max CSEF: {optimal_max:.3f}')

    plt.xlabel('Iteration')
    plt.ylabel('Ergonomic Score / CSEF Value')
    plt.title('Ergonomic Scores and CSEF Values over Iterations')
    plt.legend()
    plt.grid(True)
    plt.savefig('2.png', dpi=300)
    plt.show()

    # 绘制关节角度历史
    joint_history_l_arr = np.array(joint_history_l)
    joint_history_r_arr = np.array(joint_history_r)
    joint_names = ['Joint 1', 'Joint 2', 'Joint 3', 'Joint 4']

    # 左臂关节角度历史
    plt.figure(figsize=(12, 6))
    for j in range(4):
        plt.plot(iterations, joint_history_l_arr[:, j],
                 marker='o', linestyle='-', label=f"{joint_names[j]}")
        # 添加全局最优参考线
        plt.axhline(y=optimal_q_l[j], color=f'C{j}', linestyle='--', alpha=0.5)
    plt.xlabel('Iteration')
    plt.ylabel('Joint Angle (rad)')
    plt.title('Left Arm Joint Angles over Iterations (CSEF Method)')
    plt.legend()
    plt.grid(True)
    plt.savefig('3.png', dpi=300)
    plt.show()

    # 右臂关节角度历史
    plt.figure(figsize=(12, 6))
    for j in range(4):
        plt.plot(iterations, joint_history_r_arr[:, j],
                 marker='o', linestyle='-', label=f"{joint_names[j]}")
        # 添加全局最优参考线
        plt.axhline(y=optimal_q_r[j], color=f'C{j}', linestyle='--', alpha=0.5)
    plt.xlabel('Iteration')
    plt.ylabel('Joint Angle (rad)')
    plt.title('Right Arm Joint Angles over Iterations (CSEF Method)')
    plt.legend()
    plt.grid(True)
    plt.savefig('4.png', dpi=300)
    plt.show()

    # 输出最终结果统计
    print("\n=================== 最终结果 ===================")
    print(f"左臂初始分数: {score_history_l[0]:.4f}, 最终分数: {score_history_l[-1]:.4f}")
    print(f"右臂初始分数: {score_history_r[0]:.4f}, 最终分数: {score_history_r[-1]:.4f}")
    print(f"初始最大CSEF: {max_csef_history[0]:.4f}, 最终最大CSEF: {max_csef_history[-1]:.4f}")
    print(f"全局最优最大CSEF: {optimal_max:.4f}")
    print(
        f"到达全局最优的程度: {(1.0 - (max_csef_history[-1] - optimal_max) / (max_csef_history[0] - optimal_max)) * 100:.1f}%")
    print(f"初始CSEF差异: {csef_diff_history[0]:.4f}, 最终CSEF差异: {csef_diff_history[-1]:.4f}")

    # 验证最终距离约束
    initial_hand_distance = np.linalg.norm(trajectory_hand_l[0] - trajectory_hand_r[0])
    final_hand_distance = np.linalg.norm(trajectory_hand_l[-1] - trajectory_hand_r[-1])
    print(f"初始双手距离: {initial_hand_distance:.4f}m")
    print(f"最终双手距离: {final_hand_distance:.4f}m")
    print(
        f"距离误差: {abs(final_hand_distance - initial_hand_distance):.4f}m ({abs(final_hand_distance - initial_hand_distance) / initial_hand_distance * 100:.2f}%)")


if __name__ == '__main__':
    main()