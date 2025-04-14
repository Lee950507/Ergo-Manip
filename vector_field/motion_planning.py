#!/usr/bin/env python3
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, PillowWriter
import utils
import main_opt_static as mos
from itertools import product
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable, get_cmap


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
    optimal_q = [0, 0, 0, -math.pi / 6]

    skeleton_joint_name, skeleton_joints, skeleton_parent_indices, skeleton_joint_local_translation = \
             utils.read_skeleton_motion('/home/curi/Chenzui/Ergo-Manip/data/demo_2_test_chenzui_only_optitrack2hotu.npy')
    skeleton_joint = skeleton_joints[400, :]
    global_positions, global_rotations = utils.forward_kinematics(skeleton_joint_local_translation,
                                                                  skeleton_joint, skeleton_parent_indices)
    # 计算上肢长度（用于正向运动学）
    d_ual, d_uar, d_lal, d_lar = mos.calculate_arm_dimensions(global_positions[6], global_positions[7],
                                                              global_positions[8], global_positions[3],
                                                              global_positions[4], global_positions[5])
    # 计算初始“最优”位置（仅用于可视化对比），这里采用 optimal_q 得到的手腕位置
    _, optimal_position = mos.forward_kinematics(optimal_q, d_ual, d_lal)
    optimal_position = trans_shoulder2global(optimal_position, global_positions[6], arm='left')
    # optimal_position = trans_shoulder2global(optimal_position, global_positions[3], arm='right')

    p_elbowL_init, p_wristL_init = trans_global2shoulder(global_positions[6], global_positions[7], global_positions[8], arm='left')
    # p_elbowR_init, p_wristR_init = trans_global2shoulder(global_positions[3], global_positions[4], global_positions[5],
    #                                                      arm='right')

    current_q = mos.inverse_kinematics(p_elbowL_init, p_wristL_init, d_ual, d_lal)
    current_score = utils.calculate_upper_limb_score_with_joint_angles(current_q)

    hand_current = global_positions[8]
    elbow_current = global_positions[7]
    # hand_current = global_positions[5]
    # elbow_current = global_positions[4]

    shoulder = global_positions[6].copy()
    # shoulder = global_positions[3].copy()

    # 为动画记录历史轨迹（可选）
    trajectory_hand = [hand_current.copy()]
    trajectory_elbow = [elbow_current.copy()]

    score_history = []
    joint_history = []

    # 设置候选离散采样数、压缩系数和迭代次数
    num_samples_per_joint = 8
    comp_factor = 0.1
    num_iterations = 20
    max_disp = 0.05  # maximum allowed displacement per iteration in global (hand) space

    # 创建图形和三维坐标轴
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

# ---------------------- 迭代更新函数 ----------------------
    def update(frame):
        nonlocal current_q, global_positions, trajectory_hand, trajectory_elbow
        ax.clear()
        # ax.set_xlim((-0.5, 0.1))
        # ax.set_ylim((-0.6, 0))
        # ax.set_zlim((0.9, 1.5))
        # ax.set_xlim((-0.5, 0.1))
        # ax.set_ylim((0, 0.6))
        # ax.set_zlim((0.9, 1.5))

        ax.view_init(elev=30, azim=-30)

        # 以当前配置为中心，压缩关节角范围（仅对左侧手臂使用）
        new_bounds = compress_bounds(joint_angle_bounds, current_q, compression_factor=comp_factor)
        joint_angle_ranges = [np.linspace(lower, upper, num_samples_per_joint) for lower, upper in new_bounds]

        q_combinations = np.array(list(product(*joint_angle_ranges)))

        scores = []
        candidate_elbows = []
        candidate_hands = []

        for q in q_combinations:
            elbow_cand, hand_cand = mos.forward_kinematics(q, d_ual, d_lal)
            hand_cand = trans_shoulder2global(hand_cand, shoulder, arm='left')
            elbow_cand = trans_shoulder2global(elbow_cand, shoulder, arm='left')

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

        ref_point = global_positions[8]
        # ref_point = global_positions[5]

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

        new_elbow, new_hand = mos.forward_kinematics(new_q, d_ual, d_lal)
        new_hand = trans_shoulder2global(new_hand, shoulder, arm='left')
        new_elbow = trans_shoulder2global(new_elbow, shoulder, arm='left')

        current_q = new_q

        s = utils.calculate_upper_limb_score_with_joint_angles(current_q)
        score_history.append(s)

        global_positions[7] = new_elbow
        global_positions[8] = new_hand
        # global_positions[4] = new_elbow
        # global_positions[5] = new_hand

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
        ax.plot(traj[:,0], traj[:,1], traj[:,2], c='green', linestyle='--')

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
        ax.set_title(f'Iteration {frame+1}')
        ax.legend()

    # ---------------------- 动画启动 ----------------------
    anim = FuncAnimation(fig, update, frames=num_iterations, interval=800, repeat=False)
    # anim.save("/home/ubuntu/Ergo-Manip/vector_field/figs/animation_left_arm_straight.gif", writer=PillowWriter(fps=2))
    plt.show()

    plt.figure()
    iterations = np.arange(1, len(score_history) + 1)
    plt.plot(iterations, score_history, marker='o', linestyle='-')
    plt.xlabel("Iteration")
    plt.ylabel("Score")
    plt.title("Score over Iterations")
    plt.grid(True)
    # plt.savefig("/home/ubuntu/Ergo-Manip/vector_field/figs/score_variation_left_ergo_based.png")
    plt.show()

    joint_history = np.array(joint_history)  # shape: (num_iterations, num_joints)
    num_joints = joint_history.shape[1]
    plt.figure()
    for j in range(num_joints):
        plt.plot(np.arange(1, joint_history.shape[0] + 1), joint_history[:, j],
                 marker='o', linestyle='-', label=f"Joint {j + 1}")
    plt.xlabel("Iteration")
    plt.ylabel("Joint Angle (rad)")
    plt.title("Joint Angles over Iterations")
    plt.legend()
    plt.grid(True)
    # plt.savefig("/home/ubuntu/Ergo-Manip/vector_field/figs/joint_angles_plot.png")
    plt.show()

if __name__ == '__main__':
    main()