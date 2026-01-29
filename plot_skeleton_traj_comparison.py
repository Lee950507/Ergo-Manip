import numpy as np
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D
import utils
import main_opt_static as mos
from matplotlib.colors import LinearSegmentedColormap


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


def plot_trajectory_comparison(experiment_num, method_num, output_path=None):
    """
    绘制轨迹对比图

    参数：
    experiment_num: 实验编号 (1, 2, ...)
    method_num: 方法编号 (1 表示 CSEF, 2 表示 Point-to-Point)
    output_path: 输出图像路径
    """

    # 设置数据路径
    base_path = f"/home/ubuntu/Ergo-Manip/data/drilling/0921/wuxi/2/{method_num}.{experiment_num}"

    # 加载数据
    human_positions = np.load(os.path.join(base_path, "recorded_human_position.npy"), allow_pickle=True).item()
    robot_positions = np.load(os.path.join(base_path, "optimized_robot_positions.npy"), allow_pickle=True)
    joint_angles = np.load(os.path.join(base_path, "optimized_joint_angles.npy"), allow_pickle=True)

    # 读取骨架数据
    skeleton_joint_name, skeleton_joints, skeleton_parent_indices, skeleton_joint_local_translation = \
        utils.read_skeleton_motion('/home/ubuntu/Ergo-Manip/data/demo_2_test_chenzui_only_optitrack2hotu.npy')
    skeleton_joint = skeleton_joints[500, :]

    # 提取人体位置数据
    shoulder_positions = np.array(human_positions['shoulder_positions'])
    elbow_positions = np.array(human_positions['elbow_positions'])
    wrist_positions = np.array(human_positions['wrist_positions'])


    # 计算骨架初始和最终构型
    # 首先获取初始和最终肩部、肘部、手腕位置
    shoulder_start = shoulder_positions[0]
    elbow_start = elbow_positions[0]
    wrist_start = wrist_positions[0]

    shoulder_end = shoulder_positions[-1]
    elbow_end = elbow_positions[-1]
    wrist_end = wrist_positions[-1]

    # 绘制初始骨架
    global_positions_start, _ = utils.forward_kinematics(skeleton_joint_local_translation,
                                                         skeleton_joint, skeleton_parent_indices)
    global_positions_start[:, 2] = global_positions_start[:, 2] * 1.2

    # 更新骨架中的上肢位置（假设上肢关节索引与之前代码一致）
    global_positions_start[4] = global_positions_start[3] + (elbow_start - shoulder_start)
    global_positions_start[5] = global_positions_start[3] + (wrist_start - shoulder_start)
    global_positions_start[7] = global_positions_start[6] + [0, 0, -0.25]
    global_positions_start[8] = global_positions_start[6] + [-0.05, 0, -0.5]

    robot_increments = robot_positions - robot_positions[0]  # 计算机器人位置的增量
    robot_positions_relative =  global_positions_start[5] + robot_increments  # 将增量添加到人类手腕初始位置
    wrist_positions = wrist_positions - wrist_positions[0] + global_positions_start[5]
    # 绘制终止骨架
    global_positions_end = global_positions_start.copy()
    global_positions_end[3] = global_positions_end[3] + (shoulder_end - shoulder_start)
    global_positions_end[4] = global_positions_end[3] + (elbow_end - shoulder_end)
    global_positions_end[5] = global_positions_end[3] + (wrist_end - shoulder_end)

    # 创建图形
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # 设置视角
    ax.view_init(elev=15, azim=60)

    # 使用自定义颜色映射，以便轨迹逐渐变化颜色以表示时间流动
    n_points = len(robot_positions_relative)

    # 为CSEF和Point-to-Point方法使用不同颜色
    if method_num == 1:  # CSEF
        human_cmap = LinearSegmentedColormap.from_list('human_cmap', ['blue', 'cyan'])
        robot_cmap = LinearSegmentedColormap.from_list('robot_cmap', ['red', 'yellow'])
        method_name = 'CSEF-based Motion Planning'
    else:  # Point-to-Point
        human_cmap = LinearSegmentedColormap.from_list('human_cmap', ['darkblue', 'lightblue'])
        robot_cmap = LinearSegmentedColormap.from_list('robot_cmap', ['darkred', 'lightcoral'])
        method_name = 'Point-to-Point Motion Planning'

    # 绘制轨迹（带有颜色渐变）
    for i in range(1, n_points):
        # 人体轨迹
        human_color = human_cmap(i / n_points)
        ax.plot([wrist_positions[i - 1, 0], wrist_positions[i, 0]],
                [wrist_positions[i - 1, 1], wrist_positions[i, 1]],
                [wrist_positions[i - 1, 2], wrist_positions[i, 2]],
                color=human_color, linewidth=2, alpha=0.7)

        # 机器人轨迹 - 使用相对位置
        robot_color = robot_cmap(i / n_points)
        ax.plot([robot_positions_relative[i - 1, 0], robot_positions_relative[i, 0]],
                [robot_positions_relative[i - 1, 1], robot_positions_relative[i, 1]],
                [robot_positions_relative[i - 1, 2], robot_positions_relative[i, 2]],
                color=robot_color, linewidth=2, alpha=0.7)

    # 绘制起点和终点标记
    ax.scatter(wrist_positions[0, 0], wrist_positions[0, 1], wrist_positions[0, 2],
               color='red', s=100, marker='o', label='Start')
    ax.scatter(wrist_positions[-1, 0], wrist_positions[-1, 1], wrist_positions[-1, 2],
               color='cyan', s=100, marker='o', label='Actual End')

    # 机器人起点现在与人类手腕起点相同
    # ax.scatter(robot_positions_relative[0, 0], robot_positions_relative[0, 1], robot_positions_relative[0, 2],
    #            color='red', s=100, marker='o', label='Start')
    ax.scatter(robot_positions_relative[-1, 0], robot_positions_relative[-1, 1], robot_positions_relative[-1, 2],
               color='yellow', s=100, marker='o', label='Optimized End')

    # 绘制初始骨架（半透明）
    utils.plot_skeleton(ax, global_positions_start, skeleton_parent_indices, color='black')

    # 绘制终止骨架（半透明）
    utils.plot_skeleton(ax, global_positions_end, skeleton_parent_indices, color='darkblue')

    # 突出显示上肢骨架连接
    ax.plot([global_positions_start[3, 0], global_positions_start[4, 0], global_positions_start[5, 0]],
            [global_positions_start[3, 1], global_positions_start[4, 1], global_positions_start[5, 1]],
            [global_positions_start[3, 2], global_positions_start[4, 2], global_positions_start[5, 2]],
            color='blue', linewidth=3, alpha=0.8, label='Initial Upper Limb')

    ax.plot([global_positions_end[3, 0], global_positions_end[4, 0], global_positions_end[5, 0]],
            [global_positions_end[3, 1], global_positions_end[4, 1], global_positions_end[5, 1]],
            [global_positions_end[3, 2], global_positions_end[4, 2], global_positions_end[5, 2]],
            color='cyan', linewidth=3, alpha=0.8, label='Final Upper Limb')

    # 设置轴标签和标题
    ax.set_xlabel('X Position (m)', fontsize=16, labelpad=16)
    ax.set_ylabel('Y Position (m)', fontsize=16, labelpad=16)
    ax.set_zlabel('Z Position (m)', fontsize=16, labelpad=16)

    title = f"Experiment {experiment_num}: {method_name}\nTrajectory Comparison in Task Space"
    ax.set_title(title, fontsize=14, pad=20)

    # 添加图例
    ax.legend(loc='upper left', fontsize=12, bbox_to_anchor=(0.0, 1.0))

    # 设置视图范围以包含所有数据点
    all_points = np.vstack([wrist_positions, robot_positions_relative])

    # 计算边界并稍微扩展以便更好地可视化
    # x_min, y_min, z_min = np.min(all_points, axis=0) - 0.1
    # x_max, y_max, z_max = np.max(all_points, axis=0) + 0.1

    ax.set_xlim([-0.9, 0.9])
    ax.set_ylim([-0.9, 0.9])
    ax.set_zlim([0, 1.8])
    ax.tick_params(labelsize=14)

    # 添加网格线以增强3D感知
    ax.grid(False)

    # 添加轨迹长度和时间信息
    human_path_length = np.sum(np.sqrt(np.sum(np.diff(wrist_positions, axis=0) ** 2, axis=1)))
    robot_path_length = np.sum(np.sqrt(np.sum(np.diff(robot_positions_relative, axis=0) ** 2, axis=1)))

    info_text = f"Path Length - Human: {human_path_length:.3f}m, Robot: {robot_path_length:.3f}m"
    ax.text2D(0.02, 0.02, info_text, transform=ax.transAxes, fontsize=10,
              bbox=dict(facecolor='white', alpha=0.7))

    # 设置背景色为浅灰色以增强可视效果
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    ax.set_facecolor('none')

    # 保存图像（如果指定了输出路径）
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', transparent=True)
        print(f"图像已保存至: {output_path}")

    plt.tight_layout()
    return fig, ax


def main():
    # 指定要处理的实验和方法
    experiments = [1, 2]  # 实验编号
    methods = [1, 2]  # 1: CSEF, 2: Point-to-Point

    # 创建保存图像的目录
    output_dir = "/home/ubuntu/Ergo-Manip/figures/wuxi/2"
    os.makedirs(output_dir, exist_ok=True)

    # 生成每个实验和方法的轨迹对比图
    for exp in experiments:
        for method in methods:
            output_path = os.path.join(output_dir, f"trajectory_exp{exp}_method{method}.png")
            fig, ax = plot_trajectory_comparison(exp, method, output_path)
            plt.close(fig)

    print("所有图像已生成完成！")


if __name__ == "__main__":
    main()