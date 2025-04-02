import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import utils
import main_opt_static as mos
from itertools import product
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable


def compress_bounds(joint_angle_bounds, q, compression_factor=0.5):
    new_bounds = []
    joint_center = q

    for i, (lower, upper) in enumerate(joint_angle_bounds):
        range_half = (upper - lower) * compression_factor / 2
        center = joint_center[i]
        # 计算新的边界
        new_lower = center - range_half
        new_upper = center + range_half

        # 确保新边界包含最优关节角组合
        # if new_lower > optimal_q[i]:
        #     new_lower = max(new_lower, optimal_q[i])
        # if new_upper < optimal_q[i]:
        #     new_upper = min(new_upper, optimal_q[i])

        new_bounds.append((new_lower, new_upper))

    return new_bounds


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
    skeleton_joint = skeleton_joints[410, :]
    global_positions, global_rotations = utils.forward_kinematics(skeleton_joint_local_translation,
                                                                  skeleton_joint, skeleton_parent_indices)

    # Upper limb length
    d_ual, d_uar, d_lal, d_lar = mos.calculate_arm_dimensions(global_positions[6], global_positions[7],
                                                              global_positions[8], global_positions[3],
                                                              global_positions[4], global_positions[5])

    _, optimal_position = mos.forward_kinematics(optimal_q, d_ual, d_lal)
    optimal_position[[0, 1]] = - optimal_position[[1, 0]]
    optimal_position[1] = - optimal_position[1]
    optimal_position = optimal_position + global_positions[6]

    # Transform from robot frame to each shoulder frame
    p_elbowL_init = global_positions[7] - global_positions[6]
    p_elbowL_init = np.array([p_elbowL_init[1], -p_elbowL_init[0], p_elbowL_init[2]])
    p_wristL_init = global_positions[8] - global_positions[6]
    p_wristL_init = np.array([p_wristL_init[1], -p_wristL_init[0], p_wristL_init[2]])

    p_elbowR_init = global_positions[4] - global_positions[3]
    p_elbowR_init = np.array([-p_elbowR_init[1], -p_elbowR_init[0], p_elbowR_init[2]])
    p_wristR_init = global_positions[5] - global_positions[3]
    p_wristR_init = np.array([-p_wristR_init[1], -p_wristR_init[0], p_wristR_init[2]])

    # Inverse kinematics for joint angles
    current_q = mos.inverse_kinematics(p_elbowL_init, p_wristL_init, d_ual, d_lal)
    q_r = mos.inverse_kinematics(p_elbowR_init, p_wristR_init, d_uar, d_lar)

    current_score = utils.calculate_upper_limb_score_with_joint_angles(current_q)

    new_joint_angle_bounds = compress_bounds(joint_angle_bounds, current_q, compression_factor=0.05)

    # 为每个关节生成离散的角度
    num_samples_per_joint = 8  # 每个关节的采样数
    joint_angle_ranges = [np.linspace(lower, upper, num_samples_per_joint) for lower, upper in new_joint_angle_bounds]

    # 生成所有可能的角度组合
    q_combinations = np.array(list(product(*joint_angle_ranges)))
    scores = []
    hand_positions = []
    elbow_positions = []

    # 遍历每一组关节角度组合
    for q in q_combinations:
        elbow_right, hand_right = mos.forward_kinematics(q, d_ual, d_lal)
        overall_arm_score_right = utils.calculate_upper_limb_score_with_joint_angles(q)
        scores.append(overall_arm_score_right)

        # 保存手腕和肘部的位置
        hand_right[[0, 1]] = - hand_right[[1, 0]]
        hand_right[1] = - hand_right[1]
        hand_right = hand_right + global_positions[6]
        elbow_right[[0, 1]] = - elbow_right[[1, 0]]
        elbow_right[1] = - elbow_right[1]
        elbow_right = elbow_right + global_positions[6]
        hand_positions.append(hand_right)
        elbow_positions.append(elbow_right)

    # 转换得分和位置为 numpy 数组
    scores = np.array(scores)
    hand_positions = np.array(hand_positions)
    elbow_positions = np.array(elbow_positions)

    # 检查维度
    print("Scores shape:", scores.shape)
    print("Hand positions shape:", hand_positions.shape)

    ref_point = global_positions[8]  # 参考点
    # 计算 hand_positions 中每个点与参考点之间的欧氏距离
    distances = np.linalg.norm(hand_positions - ref_point, axis=1)
    sorted_indices = np.argsort(distances)
    neighbors = sorted_indices[:]
    sorted_scores = scores[neighbors]
    sorted_hand_positions = hand_positions[neighbors]
    target_idx = neighbors[np.argmin(sorted_scores[neighbors])]
    # 形成向量：从当前点（ref_point）指向最低分数的点
    vector = sorted_hand_positions[target_idx] - ref_point
    norm_vector = np.linalg.norm(vector)
    if norm_vector > 1e-6:
        vector_normalized = vector / norm_vector
    else:
        vector_normalized = vector

    print("参考点 global_positions[8]:", ref_point)
    print("在20个邻域中，分数最低的点索引:", target_idx)
    print("该点分数:", sorted_scores[target_idx])
    print("计算得到的向量（归一化后）:", vector_normalized)

    # 计算向量场
    # 如果该点分数高于邻居点，则将邻居的方向累加，权重为分数差，并归一化计算单位方向

    # num_points = hand_positions.shape[0]
    # k_nearest = 25  # 定义邻居数量
    # vectors = np.zeros_like(hand_positions)  # 存储每个点的向量
    #
    # # 计算全部点两两之间的距离矩阵可以加快查询速度，但这里为简单起见，直接循环求最近邻
    # for i in range(num_points):
    #     pos_i = hand_positions[i]
    #     score_i = scores[i]
    #     # 计算与其他所有点的欧氏距离
    #     diff = hand_positions - pos_i  # (n,3)
    #     dist = np.linalg.norm(diff, axis=1)
    #     # 排序获取索引，排除自身（注意：自身距离为0）
    #     sorted_idx = np.argsort(dist)
    #     neighbor_indices = []
    #     for idx in sorted_idx:
    #         if idx == i:
    #             continue
    #         neighbor_indices.append(idx)
    #         if len(neighbor_indices) >= k_nearest:
    #             break
    #     vec = np.zeros(3)
    #     for j in neighbor_indices:
    #         # 当该点分数 > 邻居分数时，将指向分数低的邻居（hand_position[j] - pos_i）的单位向量乘以分数差相加
    #         if score_i > scores[j]:
    #             delta = hand_positions[j] - pos_i
    #             norm_delta = np.linalg.norm(delta)
    #             if norm_delta > 1e-6:
    #                 vec += (score_i - scores[j]) * (delta / norm_delta)
    #     vectors[i] = vec
    # # 可选择对所有向量进行归一化，以便于可视化（若向量差异较大）
    # vec_norm = np.linalg.norm(vectors, axis=1)
    # nonzero = vec_norm > 1e-6
    # # 这里没有对零向量做归一化
    # vectors[nonzero] = (vectors[nonzero].T / vec_norm[nonzero]).T

    # 绘制三维散点图
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 根据得分的高低选择颜色
    norm = Normalize(vmin=np.min(scores), vmax=np.max(scores))
    cmap = plt.get_cmap('coolwarm')  # 从蓝色到红色的渐变色
    colors = cmap(norm(scores))  # 应用颜色映射

    # 绘制散点
    # ax.scatter(hand_positions[:, 0], hand_positions[:, 1], hand_positions[:, 2], c=colors, s=5)

    ax.scatter(optimal_position[0], optimal_position[1], optimal_position[2], c='red', s=50, label='optimal_position')

    # 绘制从参考点出发的向量（箭头颜色为绿色，长度 scale 可根据需要调整）
    ax.quiver(ref_point[0], ref_point[1], ref_point[2],
              vector_normalized[0], vector_normalized[1], vector_normalized[2],
              length=0.1, normalize=True, color='green', alpha=0.8, label='Computed Vector')

    # 绘制每个点的向量（箭头），这里设置箭头长度scale为0.1，可根据实际情况调整
    # 注意：quiver 的 length 参数会自动调整箭头的长度

    # ax.quiver(hand_positions[::50, 0], hand_positions[::50, 1], hand_positions[::50, 2],
    #           vectors[::50, 0], vectors[::50, 1], vectors[::50, 2],
    #           length=0.08, normalize=True, color='blue', alpha=0.5)

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

    # 添加颜色条
    cbar = plt.colorbar(ScalarMappable(cmap=cmap, norm=norm), ax=ax, label='Overall Arm Score')

    plt.title('3D Score Distribution of Human Upper Limb')
    plt.show()