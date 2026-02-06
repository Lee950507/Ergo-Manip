#!/usr/bin/env python3
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import utils
from iros2025_code import main_opt_static as mos
from itertools import product


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
    if arm=='left':
        joint_pos[[0, 1]] = -joint_pos[[1, 0]]
        joint_pos[1] = -joint_pos[1]
        joint_pos = joint_pos + shoulder_pos
    if arm=='right':
        joint_pos[[0, 1]] = -joint_pos[[1, 0]]
        joint_pos = joint_pos + shoulder_pos
    return joint_pos


def trans_global2shoulder(shoulder, elbow, wrist, arm='left'):
    if arm=='left':
        elbow_new = elbow - shoulder
        elbow_new = np.array([elbow_new[1], -elbow_new[0], elbow_new[2]])
        wrist_new = wrist - shoulder
        wrist_new = np.array([wrist_new[1], -wrist_new[0], wrist_new[2]])
    if arm=='right':
        elbow_new = elbow - shoulder
        elbow_new = np.array([-elbow_new[1], -elbow_new[0], elbow_new[2]])
        wrist_new = wrist - shoulder
        wrist_new = np.array([-wrist_new[1], -wrist_new[0], wrist_new[2]])
    return elbow_new, wrist_new


def main():
    joint_angle_bounds = [
        (-math.pi / 18, 17 * math.pi / 18),  # Joint 1
        (-math.pi / 18, 17 * math.pi / 18),  # Joint 2
        (-np.pi / 3, np.pi / 2),  # Joint 3
        (-np.pi / 2, np.pi / 3)  # Joint 4
    ]
    optimal_q_l = [0.6, 0, 0, -0.5]
    optimal_q_r = [0.55, -0.05, 0.05, -0.55]

    skeleton_joint_name, skeleton_joints, skeleton_parent_indices, skeleton_joint_local_translation = \
             utils.read_skeleton_motion('/home/ubuntu/Ergo-Manip/data/demo_2_test_chenzui_only_optitrack2hotu.npy')
    skeleton_joint = skeleton_joints[400, :]
    global_positions, global_rotations = utils.forward_kinematics(skeleton_joint_local_translation,
                                                                  skeleton_joint, skeleton_parent_indices)
    # 计算上肢长度（用于正向运动学）
    d_ual, d_uar, d_lal, d_lar = mos.calculate_arm_dimensions(global_positions[6], global_positions[7],
                                                              global_positions[8], global_positions[3],
                                                              global_positions[4], global_positions[5])
    # 计算初始“最优”位置（仅用于可视化对比），这里采用 optimal_q 得到的手腕位置
    _, optimal_position_l = mos.forward_kinematics(optimal_q_l, d_ual, d_lal)
    optimal_position_l = trans_shoulder2global(optimal_position_l, global_positions[6], arm='left')
    _, optimal_position_r = mos.forward_kinematics(optimal_q_r, d_uar, d_lar)
    optimal_position_r = trans_shoulder2global(optimal_position_r, global_positions[3], arm='right')

    p_elbowL_init, p_wristL_init = trans_global2shoulder(global_positions[6], global_positions[7], global_positions[8], arm='left')
    p_elbowR_init, p_wristR_init = trans_global2shoulder(global_positions[3], global_positions[4], global_positions[5],
                                                         arm='right')

    current_q_l = mos.inverse_kinematics(p_elbowL_init, p_wristL_init, d_ual, d_lal)
    current_q_r = mos.inverse_kinematics(p_elbowR_init, p_wristR_init, d_uar, d_lar)
    current_score_l = utils.calculate_upper_limb_score_with_joint_angles(current_q_l)
    current_score_r = utils.calculate_upper_limb_score_with_joint_angles(current_q_r)

    hand_current_l = global_positions[8]
    elbow_current_l = global_positions[7]
    hand_current_r = global_positions[5]
    elbow_current_r = global_positions[4]

    shoulder_l = global_positions[6].copy()
    shoulder_r = global_positions[3].copy()

    # 为动画记录历史轨迹（可选）
    trajectory_hand_l = [hand_current_l.copy()]
    trajectory_elbow_l = [elbow_current_l.copy()]
    trajectory_hand_r = [hand_current_r.copy()]
    trajectory_elbow_r = [elbow_current_r.copy()]

    score_history_l = []
    joint_history_l = []
    score_history_r = []
    joint_history_r = []

    # 设置候选离散采样数、压缩系数和迭代次数
    num_samples_per_joint = 5
    comp_factor = 0.1
    num_iterations = 20
    max_disp = 0.05  # maximum allowed displacement per iteration in global (hand) space
    distance_weight = 10.0

    # 创建图形和三维坐标轴
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

# ---------------------- 迭代更新函数 ----------------------
    def update(frame):
        nonlocal current_q_l, current_q_r, global_positions, trajectory_hand_l, trajectory_elbow_l, trajectory_hand_r, trajectory_elbow_r
        nonlocal score_history_l, joint_history_l, score_history_r, joint_history_r

        ax.clear()
        ax.set_xlim((-1.2, 0.3))
        ax.set_ylim((-0.7, 0.8))
        ax.set_zlim((0.0, 1.5))
        # ax.set_xlim((-0.5, 0.1))
        # ax.set_ylim((0, 0.6))
        # ax.set_zlim((0.9, 1.5))

        ax.view_init(elev=30, azim=-30)

        # 以当前配置为中心，压缩关节角范围（仅对左侧手臂使用）
        new_bounds_l = compress_bounds(joint_angle_bounds, current_q_l, compression_factor=comp_factor)
        new_bounds_r = compress_bounds(joint_angle_bounds, current_q_r, compression_factor=comp_factor)
        joint_angle_ranges_l = [np.linspace(lower, upper, num_samples_per_joint) for lower, upper in new_bounds_l]
        joint_angle_ranges_r = [np.linspace(lower, upper, num_samples_per_joint) for lower, upper in new_bounds_r]

        q_combinations_l = np.array(list(product(*joint_angle_ranges_l)))
        q_combinations_r = np.array(list(product(*joint_angle_ranges_r)))

        scores_l = []
        candidate_elbows_l = []
        candidate_hands_l = []
        for q in q_combinations_l:
            elbow_cand, hand_cand = mos.forward_kinematics(q, d_ual, d_lal)
            hand_cand = trans_shoulder2global(hand_cand, shoulder_l, arm='left')
            elbow_cand = trans_shoulder2global(elbow_cand, shoulder_l, arm='left')

            candidate_elbows_l.append(elbow_cand)
            candidate_hands_l.append(hand_cand)
            s = utils.calculate_upper_limb_score_with_joint_angles(q)
            scores_l.append(s)

        scores_r = []
        candidate_elbows_r = []
        candidate_hands_r = []
        for q in q_combinations_r:
            elbow_cand, hand_cand = mos.forward_kinematics(q, d_uar, d_lar)
            hand_cand = trans_shoulder2global(hand_cand, shoulder_r, arm='right')
            elbow_cand = trans_shoulder2global(elbow_cand, shoulder_r, arm='right')

            candidate_elbows_r.append(elbow_cand)
            candidate_hands_r.append(hand_cand)
            s = utils.calculate_upper_limb_score_with_joint_angles(q)
            scores_r.append(s)

        scores_l = np.array(scores_l)
        candidate_elbows_l = np.array(candidate_elbows_l)
        candidate_hands_l = np.array(candidate_hands_l)
        scores_r = np.array(scores_r)
        candidate_elbows_r = np.array(candidate_elbows_r)
        candidate_hands_r = np.array(candidate_hands_r)

        ref_point_l = global_positions[8]
        ref_point_r = global_positions[5]
        d0 = np.linalg.norm(ref_point_r - ref_point_l)

        ## Bimanual update with relative distance constraint
        best_cost = None
        best_idx_l = None
        best_idx_r = None
        for i in range(len(q_combinations_l)):
            for j in range(len(q_combinations_r)):
                # 两候选点对应的左右手位置
                hand_l = candidate_hands_l[i]
                hand_r = candidate_hands_r[j]
                pair_distance = np.linalg.norm(hand_r - hand_l)
                cost = max(scores_l[i], scores_r[j]) + distance_weight * abs(pair_distance - d0)
                if best_cost is None or cost < best_cost:
                    best_cost = cost
                    best_idx_l = i
                    best_idx_r = j

        # 如果找到了最佳候选组合，则更新左右臂状态
        candidate_q_l = q_combinations_l[best_idx_l]
        candidate_q_r = q_combinations_r[best_idx_r]
        candidate_hand_l = candidate_hands_l[best_idx_l]
        candidate_hand_r = candidate_hands_r[best_idx_r]

        ## Bimanual update with straight lines
        # expected_l = global_positions[8] + frame * (optimal_position_l - global_positions[8]) / 20
        # expected_r = global_positions[5] + frame * (optimal_position_r - global_positions[5]) / 20
        #
        # candidate_expected_dists_l = np.linalg.norm(candidate_hands_l - expected_l, axis=1)
        # target_idx = np.argmin(candidate_expected_dists_l)
        # candidate_q_l = q_combinations_l[target_idx]
        # candidate_hand_l = candidate_hands_l[target_idx]
        # candidate_elbow_l = candidate_elbows_l[target_idx]
        #
        # candidate_expected_dists_r = np.linalg.norm(candidate_hands_r - expected_r, axis=1)
        # target_idx = np.argmin(candidate_expected_dists_r)
        # candidate_q_r = q_combinations_r[target_idx]
        # candidate_hand_r = candidate_hands_r[target_idx]
        # candidate_elbow_r = candidate_elbows_r[target_idx]

        current_hand_l = global_positions[8]
        disp_l = candidate_hand_l - current_hand_l
        dist_l = np.linalg.norm(disp_l)
        ratio_l = max_disp / dist_l if dist_l > max_disp else 1.0

        current_hand_r = global_positions[5]
        disp_r = candidate_hand_r - current_hand_r
        dist_r = np.linalg.norm(disp_r)
        ratio_r = max_disp / dist_r if dist_r > max_disp else 1.0

        # 线性插值更新关节角
        new_q_l = current_q_l + ratio_l * (candidate_q_l - current_q_l)
        new_q_r = current_q_r + ratio_r * (candidate_q_r - current_q_r)

        new_elbow_l, new_hand_l = mos.forward_kinematics(new_q_l, d_ual, d_lal)
        new_hand_l = trans_shoulder2global(new_hand_l, shoulder_l, arm='left')
        new_elbow_l = trans_shoulder2global(new_elbow_l, shoulder_l, arm='left')

        new_elbow_r, new_hand_r = mos.forward_kinematics(new_q_r, d_uar, d_lar)
        new_hand_r = trans_shoulder2global(new_hand_r, shoulder_r, arm='right')
        new_elbow_r = trans_shoulder2global(new_elbow_r, shoulder_r, arm='right')

        current_q_l = new_q_l
        current_q_r = new_q_r
        global_positions[8] = new_hand_l
        global_positions[7] = new_elbow_l
        global_positions[5] = new_hand_r
        global_positions[4] = new_elbow_r

        trajectory_hand_l.append(new_hand_l.copy())
        trajectory_elbow_l.append(new_elbow_l.copy())
        trajectory_hand_r.append(new_hand_r.copy())
        trajectory_elbow_r.append(new_elbow_r.copy())
        joint_history_l.append(new_q_l.copy())
        joint_history_r.append(new_q_r.copy())
        score_history_l.append(utils.calculate_upper_limb_score_with_joint_angles(new_q_l))
        score_history_r.append(utils.calculate_upper_limb_score_with_joint_angles(new_q_r))

        # ---------------------- 绘制部分 ----------------------
        ax.scatter(shoulder_l[0], shoulder_l[1], shoulder_l[2], c='black', s=50)
        ax.scatter(new_elbow_l[0], new_elbow_l[1], new_elbow_l[2], c='blue', s=50)
        ax.scatter(new_hand_l[0], new_hand_l[1], new_hand_l[2], c='green', s=50, label='Left Hand')
        ax.plot([shoulder_l[0], new_elbow_l[0], new_hand_l[0]],
                [shoulder_l[1], new_elbow_l[1], new_hand_l[1]],
                [shoulder_l[2], new_elbow_l[2], new_hand_l[2]], c='red', linewidth=2)

        ax.scatter(shoulder_r[0], shoulder_r[1], shoulder_r[2], c='black', s=50)
        ax.scatter(new_elbow_r[0], new_elbow_r[1], new_elbow_r[2], c='cyan', s=50,)
        ax.scatter(new_hand_r[0], new_hand_r[1], new_hand_r[2], c='magenta', s=50, label='Right Hand')
        ax.plot([shoulder_r[0], new_elbow_r[0], new_hand_r[0]],
                [shoulder_r[1], new_elbow_r[1], new_hand_r[1]],
                [shoulder_r[2], new_elbow_r[2], new_hand_r[2]], c='orange', linewidth=2)

        # 同时显示左右候选点的颜色条（分别）
        # sm_l = ScalarMappable(cmap=cmap_l, norm=norm_obj_l)
        # sm_l.set_array(scores_l)
        # cbar_l = plt.colorbar(sm_l, ax=ax, pad=0.1)
        # cbar_l.set_label('Left Score')
        # sm_r = ScalarMappable(cmap=cmap_r, norm=norm_obj_r)
        # sm_r.set_array(scores_r)
        # cbar_r = plt.colorbar(sm_r, ax=ax, pad=0.05)
        # cbar_r.set_label('Right Score')

        traj_l = np.array(trajectory_hand_l)
        ax.plot(traj_l[:, 0], traj_l[:, 1], traj_l[:, 2], c='green', linestyle='--')
        traj_r = np.array(trajectory_hand_r)
        ax.plot(traj_r[:, 0], traj_r[:, 1], traj_r[:, 2], c='green', linestyle='--')

        # ax.scatter(optimal_position_l[0], optimal_position_l[1], optimal_position_l[2],
        #            c='red', s=50)
        # ax.scatter(optimal_position_r[0], optimal_position_r[1], optimal_position_r[2],
        #            c='red', s=50)

        utils.plot_skeleton(ax, global_positions, skeleton_parent_indices, color='black')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f"Iteration {frame + 1}")
        ax.legend(loc='upper left', fontsize='small')

    # 制作动画并保存为 GIF
    anim = FuncAnimation(fig, update, frames=num_iterations, interval=800, repeat=False)
    writer = PillowWriter(fps=2)
    anim.save("/home/ubuntu/Ergo-Manip/vector_field/figs/bi/double_arm_animation.gif", writer=writer)
    plt.show()

    # 动画结束后分别绘制并保存分数与关节角变化图（左右臂）
    iterations = np.arange(1, len(score_history_l) + 1)
    plt.figure()
    plt.plot(iterations, score_history_l, marker='o', linestyle='-', label='Left Score')
    plt.plot(iterations, score_history_r, marker='o', linestyle='-', label='Right Score')
    plt.xlabel('Iteration')
    plt.ylabel('Score')
    plt.title('Score over Iterations')
    plt.legend()
    plt.grid(True)
    plt.savefig("/home/ubuntu/Ergo-Manip/vector_field/figs/bi/double_arm_score_plot.png")
    plt.show()

    # 绘制左右臂的关节角变化
    joint_history_l_arr = np.array(joint_history_l)  # shape: (n, num_joints)
    joint_history_r_arr = np.array(joint_history_r)
    num_joints = joint_history_l_arr.shape[1]
    plt.figure()
    for j in range(num_joints):
        plt.plot(np.arange(1, joint_history_l_arr.shape[0] + 1), joint_history_l_arr[:, j],
                 marker='o', linestyle='-', label=f"Left Joint {j + 1}")
    plt.xlabel('Iteration')
    plt.ylabel('Joint Angle (rad)')
    plt.title('Left Arm Joint Angles over Iterations')
    plt.legend()
    plt.grid(True)
    plt.savefig("/home/ubuntu/Ergo-Manip/vector_field/figs/bi/left_arm_joint_angles_plot.png")
    plt.show()

    plt.figure()
    for j in range(num_joints):
        plt.plot(np.arange(1, joint_history_r_arr.shape[0] + 1), joint_history_r_arr[:, j],
                 marker='o', linestyle='-', label=f"Right Joint {j + 1}")
    plt.xlabel('Iteration')
    plt.ylabel('Joint Angle (rad)')
    plt.title('Right Arm Joint Angles over Iterations')
    plt.legend()
    plt.grid(True)
    plt.savefig("/home/ubuntu/Ergo-Manip/vector_field/figs/bi/right_arm_joint_angles_plot.png")
    plt.show()

if __name__ == '__main__':
    main()