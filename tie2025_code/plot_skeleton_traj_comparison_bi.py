import numpy as np
import matplotlib.pyplot as plt
import os
import utils
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
    绘制双臂轨迹对比图

    参数：
    experiment_num: 实验编号 (1, 2, ...)
    method_num: 方法编号 (1 表示 CSEF, 2 表示 Point-to-Point)
    output_path: 输出图像路径
    """

    # 设置数据路径
    base_path = f"/home/ubuntu/Ergo-Manip/data/box_carrying/0920/chenzui/3/{method_num}.{experiment_num}"

    # 加载数据
    human_positions = np.load(os.path.join(base_path, "recorded_human_position.npy"), allow_pickle=True).item()
    robot_positions_l = np.load(os.path.join(base_path, "optimized_robot_positions_l.npy"), allow_pickle=True)
    robot_positions_r = np.load(os.path.join(base_path, "optimized_robot_positions_r.npy"), allow_pickle=True)
    joint_angles_l = np.load(os.path.join(base_path, "optimized_joint_angles_l.npy"), allow_pickle=True)
    joint_angles_r = np.load(os.path.join(base_path, "optimized_joint_angles_r.npy"), allow_pickle=True)

    # 读取骨架数据
    skeleton_joint_name, skeleton_joints, skeleton_parent_indices, skeleton_joint_local_translation = \
        utils.read_skeleton_motion('/home/ubuntu/Ergo-Manip/data/demo_2_test_chenzui_only_optitrack2hotu.npy')
    skeleton_joint = skeleton_joints[500, :]

    # 提取人体位置数据 - 左臂
    shoulder_positions_l = np.array(human_positions['shoulder_positions_l'])
    elbow_positions_l = np.array(human_positions['elbow_positions_l'])
    wrist_positions_l = np.array(human_positions['wrist_positions_l'])

    # 提取人体位置数据 - 右臂
    shoulder_positions_r = np.array(human_positions['shoulder_positions'])
    elbow_positions_r = np.array(human_positions['elbow_positions'])
    wrist_positions_r = np.array(human_positions['wrist_positions'])

    # 计算骨架初始和最终构型
    # 首先获取初始和最终肩部、肘部、手腕位置 - 左臂
    shoulder_start_l = shoulder_positions_l[0]
    elbow_start_l = elbow_positions_l[0]
    wrist_start_l = wrist_positions_l[0]

    shoulder_end_l = shoulder_positions_l[-1]
    elbow_end_l = elbow_positions_l[-1]
    wrist_end_l = wrist_positions_l[-1]

    # 首先获取初始和最终肩部、肘部、手腕位置 - 右臂
    shoulder_start_r = shoulder_positions_r[0]
    elbow_start_r = elbow_positions_r[0]
    wrist_start_r = wrist_positions_r[0]

    shoulder_end_r = shoulder_positions_r[-1]
    elbow_end_r = elbow_positions_r[-1]
    wrist_end_r = wrist_positions_r[-1]

    # 绘制初始骨架
    global_positions_start, _ = utils.forward_kinematics(skeleton_joint_local_translation,
                                                         skeleton_joint, skeleton_parent_indices)
    global_positions_start[:, 2] = global_positions_start[:, 2] * 1.2

    # 更新骨架中的上肢位置 - 左臂
    # global_positions_start[6] = shoulder_start_l  # 左肩
    global_positions_start[7] = global_positions_start[6] + (elbow_start_l - shoulder_start_l)  # 左肘
    global_positions_start[8] = global_positions_start[6] + (wrist_start_l - shoulder_start_l)  # 左手腕

    # 更新骨架中的上肢位置 - 右臂
    # global_positions_start[3] = shoulder_start_r  # 右肩
    global_positions_start[4] = global_positions_start[3] + (elbow_start_r - shoulder_start_r)  # 右肘
    global_positions_start[5] = global_positions_start[3] + (wrist_start_r - shoulder_start_r)  # 右手腕

    # 计算机器人轨迹相对位置 - 左臂
    robot_increments_l = robot_positions_l - robot_positions_l[0]  # 计算机器人位置的增量
    robot_positions_relative_l = global_positions_start[8] + robot_increments_l  # 将增量添加到人类手腕初始位置
    wrist_positions_l = wrist_positions_l - wrist_positions_l[0] + global_positions_start[8]

    # 计算机器人轨迹相对位置 - 右臂
    robot_increments_r = robot_positions_r - robot_positions_r[0]  # 计算机器人位置的增量
    robot_positions_relative_r = global_positions_start[5] + robot_increments_r  # 将增量添加到人类手腕初始位置
    wrist_positions_r = wrist_positions_r - wrist_positions_r[0] + global_positions_start[5]

    # 绘制终止骨架
    global_positions_end = global_positions_start.copy()

    # 更新终止骨架 - 左臂
    global_positions_end[6] = global_positions_end[6] + (shoulder_end_l - shoulder_start_l)
    global_positions_end[7] = global_positions_end[6] + (elbow_end_l - shoulder_end_l)
    global_positions_end[8] = global_positions_end[6] + (wrist_end_l - shoulder_end_l)

    # 更新终止骨架 - 右臂
    global_positions_end[3] = global_positions_end[3] + (shoulder_end_r - shoulder_start_r)
    global_positions_end[4] = global_positions_end[3] + (elbow_end_r - shoulder_end_r)
    global_positions_end[5] = global_positions_end[3] + (wrist_end_r - shoulder_end_r)

    # 创建图形
    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(111, projection='3d')

    # 设置视角
    ax.view_init(elev=30, azim=105)

    # 为CSEF和Point-to-Point方法使用不同颜色
    if method_num == 1:  # CSEF
        human_cmap_l = LinearSegmentedColormap.from_list('human_cmap_l', ['blue', 'cyan'])
        robot_cmap_l = LinearSegmentedColormap.from_list('robot_cmap_l', ['red', 'yellow'])
        human_cmap_r = LinearSegmentedColormap.from_list('human_cmap_r', ['blue', 'cyan'])
        robot_cmap_r = LinearSegmentedColormap.from_list('robot_cmap_r', ['red', 'yellow'])
        method_name = 'CSEF-based Motion Planning'
    else:  # Point-to-Point
        human_cmap_l = LinearSegmentedColormap.from_list('human_cmap_l', ['darkblue', 'lightblue'])
        robot_cmap_l = LinearSegmentedColormap.from_list('robot_cmap_l', ['darkred', 'lightcoral'])
        human_cmap_r = LinearSegmentedColormap.from_list('human_cmap_r', ['darkblue', 'lightblue'])
        robot_cmap_r = LinearSegmentedColormap.from_list('robot_cmap_r', ['darkred', 'lightcoral'])
        method_name = 'Point-to-Point Motion Planning'

    # 绘制轨迹 - 左臂
    n_points_l = len(robot_positions_relative_l)
    for i in range(1, n_points_l):
        # 人体轨迹 - 左臂
        human_color_l = human_cmap_l(i / n_points_l)
        ax.plot([wrist_positions_l[i - 1, 0], wrist_positions_l[i, 0]],
                [wrist_positions_l[i - 1, 1], wrist_positions_l[i, 1]],
                [wrist_positions_l[i - 1, 2], wrist_positions_l[i, 2]],
                color=human_color_l, linewidth=2, alpha=0.7)

        # 机器人轨迹 - 左臂
        robot_color_l = robot_cmap_l(i / n_points_l)
        ax.plot([robot_positions_relative_l[i - 1, 0], robot_positions_relative_l[i, 0]],
                [robot_positions_relative_l[i - 1, 1], robot_positions_relative_l[i, 1]],
                [robot_positions_relative_l[i - 1, 2], robot_positions_relative_l[i, 2]],
                color=robot_color_l, linewidth=2, alpha=0.7)

    # 绘制轨迹 - 右臂
    n_points_r = len(robot_positions_relative_r)
    for i in range(1, n_points_r):
        # 人体轨迹 - 右臂
        human_color_r = human_cmap_r(i / n_points_r)
        ax.plot([wrist_positions_r[i - 1, 0], wrist_positions_r[i, 0]],
                [wrist_positions_r[i - 1, 1], wrist_positions_r[i, 1]],
                [wrist_positions_r[i - 1, 2], wrist_positions_r[i, 2]],
                color=human_color_r, linewidth=2, alpha=0.7)

        # 机器人轨迹 - 右臂
        robot_color_r = robot_cmap_r(i / n_points_r)
        ax.plot([robot_positions_relative_r[i - 1, 0], robot_positions_relative_r[i, 0]],
                [robot_positions_relative_r[i - 1, 1], robot_positions_relative_r[i, 1]],
                [robot_positions_relative_r[i - 1, 2], robot_positions_relative_r[i, 2]],
                color=robot_color_r, linewidth=2, alpha=0.7)

    # 绘制起点和终点标记 - 左臂
    ax.scatter(wrist_positions_l[0, 0], wrist_positions_l[0, 1], wrist_positions_l[0, 2],
               color='red', s=100, marker='o', label='Left Start')
    ax.scatter(wrist_positions_l[-1, 0], wrist_positions_l[-1, 1], wrist_positions_l[-1, 2],
               color='cyan', s=100, marker='o', label='Left Actual End')
    ax.scatter(robot_positions_relative_l[-1, 0], robot_positions_relative_l[-1, 1], robot_positions_relative_l[-1, 2],
               color='yellow', s=100, marker='o', label='Left Optimized End')

    # 绘制起点和终点标记 - 右臂
    ax.scatter(wrist_positions_r[0, 0], wrist_positions_r[0, 1], wrist_positions_r[0, 2],
               color='red', s=100, marker='o', label='Right Start')
    ax.scatter(wrist_positions_r[-1, 0], wrist_positions_r[-1, 1], wrist_positions_r[-1, 2],
               color='cyan', s=100, marker='o', label='Right Actual End')
    ax.scatter(robot_positions_relative_r[-1, 0], robot_positions_relative_r[-1, 1], robot_positions_relative_r[-1, 2],
               color='yellow', s=100, marker='o', label='Right Optimized End')

    # 绘制初始骨架
    utils.plot_skeleton(ax, global_positions_start, skeleton_parent_indices, color='darkblue')

    # 绘制终止骨架
    utils.plot_skeleton(ax, global_positions_end, skeleton_parent_indices, color='darkblue')

    # 突出显示上肢骨架连接 - 左臂
    ax.plot([global_positions_start[6, 0], global_positions_start[7, 0], global_positions_start[8, 0]],
            [global_positions_start[6, 1], global_positions_start[7, 1], global_positions_start[8, 1]],
            [global_positions_start[6, 2], global_positions_start[7, 2], global_positions_start[8, 2]],
            color='blue', linewidth=3, alpha=0.8, label='Initial Left Arm')

    ax.plot([global_positions_end[6, 0], global_positions_end[7, 0], global_positions_end[8, 0]],
            [global_positions_end[6, 1], global_positions_end[7, 1], global_positions_end[8, 1]],
            [global_positions_end[6, 2], global_positions_end[7, 2], global_positions_end[8, 2]],
            color='cyan', linewidth=3, alpha=0.8, label='Final Left Arm')

    # 突出显示上肢骨架连接 - 右臂
    ax.plot([global_positions_start[3, 0], global_positions_start[4, 0], global_positions_start[5, 0]],
            [global_positions_start[3, 1], global_positions_start[4, 1], global_positions_start[5, 1]],
            [global_positions_start[3, 2], global_positions_start[4, 2], global_positions_start[5, 2]],
            color='blue', linewidth=3, alpha=0.8, label='Initial Right Arm')

    ax.plot([global_positions_end[3, 0], global_positions_end[4, 0], global_positions_end[5, 0]],
            [global_positions_end[3, 1], global_positions_end[4, 1], global_positions_end[5, 1]],
            [global_positions_end[3, 2], global_positions_end[4, 2], global_positions_end[5, 2]],
            color='cyan', linewidth=3, alpha=0.8, label='Final Right Arm')

    # 设置轴标签和标题
    ax.set_xlabel('X Position (m)', fontsize=16, labelpad=16)
    ax.set_ylabel('Y Position (m)', fontsize=16, labelpad=16)
    ax.set_zlabel('Z Position (m)', fontsize=16, labelpad=16)

    title = f"Experiment {experiment_num}: {method_name}\nDual-Arm Trajectory Comparison in Task Space"
    ax.set_title(title, fontsize=14, pad=20)

    # 添加图例
    ax.legend(loc='upper left', fontsize=12, bbox_to_anchor=(0.0, 1.0))

    # 设置视图范围
    ax.set_xlim([-0.9, 0.9])
    ax.set_ylim([-0.9, 0.9])
    ax.set_zlim([0, 1.8])
    ax.tick_params(labelsize=14)

    # 添加网格线以增强3D感知
    ax.grid(False)

    # 添加轨迹长度信息
    human_path_length_l = np.sum(np.sqrt(np.sum(np.diff(wrist_positions_l, axis=0) ** 2, axis=1)))
    robot_path_length_l = np.sum(np.sqrt(np.sum(np.diff(robot_positions_relative_l, axis=0) ** 2, axis=1)))

    human_path_length_r = np.sum(np.sqrt(np.sum(np.diff(wrist_positions_r, axis=0) ** 2, axis=1)))
    robot_path_length_r = np.sum(np.sqrt(np.sum(np.diff(robot_positions_relative_r, axis=0) ** 2, axis=1)))

    info_text = (f"Path Length - Left Arm: Human {human_path_length_l:.3f}m, Robot {robot_path_length_l:.3f}m\n"
                 f"Path Length - Right Arm: Human {human_path_length_r:.3f}m, Robot {robot_path_length_r:.3f}m")
    ax.text2D(0.02, 0.02, info_text, transform=ax.transAxes, fontsize=10,
              bbox=dict(facecolor='white', alpha=0.7))

    # 设置透明背景
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
    output_dir = "/home/ubuntu/Ergo-Manip/figures/box_carrying/chenzui/3"
    os.makedirs(output_dir, exist_ok=True)

    # 生成每个实验和方法的轨迹对比图
    for exp in experiments:
        for method in methods:
            output_path = os.path.join(output_dir, f"dual_arm_trajectory_exp{exp}_method{method}.png")
            fig, ax = plot_trajectory_comparison(exp, method, output_path)
            plt.close(fig)

    print("所有图像已生成完成！")


if __name__ == "__main__":
    main()