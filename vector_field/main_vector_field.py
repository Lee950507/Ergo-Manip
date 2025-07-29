import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, PillowWriter
import utils
import main_opt_static as mos
from itertools import product
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable


def compress_bounds(joint_angle_bounds, q, optimal_q, compression_factor=0.5):
    new_bounds = []
    joint_center = (q + optimal_q) / 2

    for i, (lower, upper) in enumerate(joint_angle_bounds):
        range_half = (upper - lower) * compression_factor / 2
        center = joint_center[i]
        # 计算新的边界
        new_lower = center - range_half
        new_upper = center + range_half

        # 确保新边界包含最优关节角组合
        if new_lower > optimal_q[i]:
            new_lower = max(new_lower, optimal_q[i])
        if new_upper < optimal_q[i]:
            new_upper = min(new_upper, optimal_q[i])

        new_bounds.append((new_lower, new_upper))

    return new_bounds


def generate_trajectory(target_x, target_y, target_ori, num_timestamps):
    # 从 (0, 0) 到 (target_x, target_y) 的匀速直线运动
    x_coords = np.linspace(0, target_x, num_timestamps)
    y_coords = np.linspace(0, target_y, num_timestamps)
    orientation = np.linspace(0, target_ori, num_timestamps)

    # 组合为三维数组 [x, y, orientation]
    trajectory = np.column_stack((x_coords, y_coords, orientation))

    return trajectory


def update(frame):
    if frame % 10 != 0:  # 每隔10帧更新一次
        return

    ax.cla()  # 清空当前坐标轴
    # 更新骨骼模型和分布图
    skeleton_joint = skeleton_joints[frame, :]
    global_positions, global_rotations = utils.forward_kinematics(skeleton_joint_local_translation,
                                                                  skeleton_joint, skeleton_parent_indices)

    rotation_matrix = np.array([[np.cos(human_trajectory[frame, 2]), -np.sin(human_trajectory[frame, 2])],
                                [np.sin(human_trajectory[frame, 2]), np.cos(human_trajectory[frame, 2])]])
    global_positions[:, :2] = global_positions[:, :2] @ rotation_matrix.T  # 应用旋转

    global_positions = global_positions + np.array([human_trajectory[frame, 0], human_trajectory[frame, 1], 0])
    # 旋转骨架模型以匹配当前朝向



    # Upper limb length
    d_ual, d_uar, d_lal, d_lar = mos.calculate_arm_dimensions(global_positions[6], global_positions[7], global_positions[8], global_positions[3], global_positions[4], global_positions[5])

    # 计算并更新右肘和右手位置
    p_elbowR_init = global_positions[4] - global_positions[3]
    p_wristR_init = global_positions[5] - global_positions[3]
    q_r = mos.inverse_kinematics(p_elbowR_init, p_wristR_init, d_uar, d_lar)

    new_joint_angle_bounds = compress_bounds(joint_angle_bounds, q_r, optimal_q, compression_factor=0.3)

    # 为每个关节生成离散的角度
    joint_angle_ranges = [np.linspace(lower, upper, num_samples_per_joint) for lower, upper in new_joint_angle_bounds]
    q_combinations = np.array(list(product(*joint_angle_ranges)))

    scores = []
    hand_positions = []
    elbow_positions = []

    for q in q_combinations:
        elbow_right, hand_right = mos.forward_kinematics(q, d_uar, d_lar)
        overall_arm_score_right = utils.calculate_upper_limb_score_with_joint_angles(q)
        scores.append(overall_arm_score_right)
        hand_right[[0, 1]] = - hand_right[[1, 0]]
        hand_right[:2] = hand_right[:2] @ rotation_matrix.T
        hand_right = hand_right + global_positions[3]
        elbow_right[[0, 1]] = - elbow_right[[1, 0]]
        elbow_right[:2] = elbow_right[:2] @ rotation_matrix.T
        elbow_right = elbow_right + global_positions[3]
        hand_positions.append(hand_right)
        elbow_positions.append(elbow_right)

    scores = np.array(scores)
    hand_positions = np.array(hand_positions)
    elbow_positions = np.array(elbow_positions)

    # 根据得分的高低选择颜色
    norm = Normalize(vmin=np.min(scores), vmax=np.max(scores))
    cmap = plt.get_cmap('coolwarm')
    colors = cmap(norm(scores))

    # 绘制散点
    ax.scatter(hand_positions[:, 0], hand_positions[:, 1], hand_positions[:, 2], c=colors, s=5)

    # 绘制轨迹
    if frame < len(human_trajectory):
        ax.plot(human_trajectory[:frame + 1, 0], human_trajectory[:frame + 1, 1], label='Trajectory', color='blue')

        # 绘制朝向箭头
        # for i in range(frame):
        #     x = human_trajectory[i, 0]
        #     y = human_trajectory[i, 1]
        #     orientation = human_trajectory[i, 2]
        #     # 计算箭头在 x 和 y 方向上的分量
        #     V = np.cos(orientation)  # x 方向分量
        #     W = np.sin(orientation)  # y 方向分量
        #     ax.quiver(x, y, V, W, angles='xy', scale_units='xy', scale=0.5, color='red')

    # 设定刻度
    x_ticks = np.arange(np.floor(hand_positions[:, 0].min()), np.ceil(hand_positions[:, 0].max()) + 1, 0.2)
    y_ticks = np.arange(np.floor(hand_positions[:, 1].min()), np.ceil(hand_positions[:, 1].max()) + 1, 0.2)
    z_ticks = np.arange(np.floor(hand_positions[:, 2].min()), np.ceil(hand_positions[:, 2].max()) + 0.5, 0.5)

    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    ax.set_zticks(z_ticks)

    # 绘制骨架
    utils.plot_skeleton(ax, global_positions, skeleton_parent_indices, color='black')

    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_zlabel('Z Position')
    ax.set_title(f'Timestep: {frame}')


if __name__ == '__main__':
    # 设置关节角度的范围
    joint_angle_bounds = [
        (-math.pi / 18, 17 * math.pi / 18),  # Joint 1
        (-math.pi / 18, 17 * math.pi / 18),  # Joint 2
        (-np.pi / 3, np.pi / 2),  # Joint 3
        (-np.pi / 2, np.pi / 3)  # Joint 4
    ]
    optimal_q = [0, 0, 0, -math.pi / 6]

    # Skeleton Model
    skeleton_joint_name, skeleton_joints, skeleton_parent_indices, skeleton_joint_local_translation = utils.read_skeleton_motion(
        '/home/ubuntu/Ergo-Manip/data/demo_2_test_chenzui_only_optitrack2hotu.npy')

    skeleton_joints = skeleton_joints[:200]
    human_trajectory = generate_trajectory(0, 0, math.pi / 6, len(skeleton_joints))

    num_samples_per_joint = 8  # 每个关节的采样数
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 动画生成
    ani = FuncAnimation(fig, update, frames=len(skeleton_joints), repeat=True)
    ani.save("/home/ubuntu/Ergo-Manip/vector_field/figs/animation_ergo_distribution.gif", writer=PillowWriter(fps=2))
    plt.show()


#     skeleton_joint = skeleton_joint[0, :]
#     global_positions, global_rotations = utils.forward_kinematics(skeleton_joint_local_translation,
#                                                                   skeleton_joint, skeleton_parent_indices)
#
#     # Upper limb length
#     d_ual, d_uar, d_lal, d_lar = mos.calculate_arm_dimensions(global_positions[6], global_positions[7],
#                                                               global_positions[8], global_positions[3],
#                                                               global_positions[4], global_positions[5])
#
#     # Transform from robot frame to each shoulder frame
#     p_elbowL_init = global_positions[7] - global_positions[6]
#     p_elbowL_init = np.array([p_elbowL_init[1], -p_elbowL_init[0], p_elbowL_init[2]])
#     p_wristL_init = global_positions[8] - global_positions[6]
#     p_wristL_init = np.array([p_wristL_init[1], -p_wristL_init[0], p_wristL_init[2]])
#
#     p_elbowR_init = global_positions[4] - global_positions[3]
#     p_elbowR_init = np.array([-p_elbowR_init[1], -p_elbowR_init[0], p_elbowR_init[2]])
#     p_wristR_init = global_positions[5] - global_positions[3]
#     p_wristR_init = np.array([-p_wristR_init[1], -p_wristR_init[0], p_wristR_init[2]])
#
#     # Inverse kinematics for joint angles
#     q_l = mos.inverse_kinematics(p_elbowL_init, p_wristL_init, d_ual, d_lal)
#     q_r = mos.inverse_kinematics(p_elbowR_init, p_wristR_init, d_uar, d_lar)
#
#     new_joint_angle_bounds = compress_bounds(joint_angle_bounds, q_r, optimal_q)
#
#     # 为每个关节生成离散的角度
#     num_samples_per_joint = 8  # 每个关节的采样数
#     joint_angle_ranges = [np.linspace(lower, upper, num_samples_per_joint) for lower, upper in new_joint_angle_bounds]
#
#     # 生成所有可能的角度组合
#     q_combinations = np.array(list(product(*joint_angle_ranges)))
#
#     # 遍历每一组关节角度组合
#     for q in q_combinations:
#         elbow_right, hand_right = mos.forward_kinematics(q, d_uar, d_lar)
#         overall_arm_score_right = utils.calculate_upper_limb_score_with_joint_angles(q)
#         scores.append(overall_arm_score_right)
#
#         # 保存手腕和肘部的位置
#         hand_right[[0, 1]] = - hand_right[[1, 0]]
#         hand_right = hand_right + global_positions[3]
#         elbow_right[[0, 1]] = - elbow_right[[1, 0]]
#         elbow_right = elbow_right + global_positions[3]
#         hand_positions.append(hand_right)
#         elbow_positions.append(elbow_right)
#
#     # 转换得分和位置为 numpy 数组
#     scores = np.array(scores)
#     hand_positions = np.array(hand_positions)
#     elbow_positions = np.array(elbow_positions)
#
#     # 进行坐标变换
#     # hand_positions[:, [0, 1]] = - hand_positions[:, [1, 0]]
#     # hand_positions = global_positions[3] + hand_positions
#     # elbow_positions[:, [0, 1]] = - elbow_positions[:, [1, 0]]
#     # elbow_positions = global_positions[3] + elbow_positions
#
#     # 检查维度
#     print("Scores shape:", scores.shape)
#     print("Hand positions shape:", hand_positions.shape)
#
#     # 绘制三维散点图
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#
#     # 根据得分的高低选择颜色
#     norm = Normalize(vmin=np.min(scores), vmax=np.max(scores))
#     cmap = plt.get_cmap('coolwarm')  # 从蓝色到红色的渐变色
#     colors = cmap(norm(scores))  # 应用颜色映射
#
#     # 绘制散点
#     sc = ax.scatter(hand_positions[:, 0], hand_positions[:, 1], hand_positions[:, 2], c=colors, s=5)
#
#     x_ticks = np.arange(np.floor(hand_positions[:, 0].min()), np.ceil(hand_positions[:, 0].max()) + 1, 0.2)
#     y_ticks = np.arange(np.floor(hand_positions[:, 1].min()), np.ceil(hand_positions[:, 1].max()) + 1, 0.2)
#     z_ticks = np.arange(np.floor(hand_positions[:, 2].min()), np.ceil(hand_positions[:, 2].max()) + 0.5, 0.5)
#
#     ax.set_xticks(x_ticks)
#     ax.set_yticks(y_ticks)
#     ax.set_zticks(z_ticks)
#
#     # 绘制骨架
#     utils.plot_skeleton(ax, global_positions, skeleton_parent_indices, color='black')
#
#     ax.set_xlabel('X Position')
#     ax.set_ylabel('Y Position')
#     ax.set_zlabel('Z Position')
#
#     # 添加颜色条
#     cbar = plt.colorbar(ScalarMappable(cmap=cmap, norm=norm), ax=ax, label='Overall Arm Score')
#
#     plt.title('3D Score Distribution of Human Upper Limb')
#     plt.show()