import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.interpolate import interp1d
import matplotlib.gridspec as gridspec
import transformation as tsf
from iros2025_code import main_opt_static as mos


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


def calculate_upper_limb_score_with_joint_angles(q):
    shoulder_abduction, shoulder_flextion, shoulder_rotation, elbow_flextion = q
    shoulder_score = 0
    shoulder_score += 1 + 2 * abs(shoulder_abduction) / np.pi
    shoulder_score += 4.5 * abs(shoulder_flextion) / np.pi
    shoulder_score += 2 * abs(shoulder_rotation) / np.pi
    elbow_score = 0
    elbow_score += 7 * abs(elbow_flextion + np.pi / 4) / np.pi

    return shoulder_score + elbow_score


def analyze_joint_angles_and_ergonomics_dual_arm(experiment_num, method_num, output_dir=None):
    """
    分析双臂关节角误差和人体工程学分数

    参数：
    experiment_num: 实验编号
    method_num: 方法编号 (1: CSEF, 2: Point-to-Point)
    output_dir: 输出目录
    """
    # 设置数据路径
    base_path = f"/home/ubuntu/Ergo-Manip/data/box_carrying/0921/yiming/4/{method_num}.{experiment_num}"

    # 加载数据
    human_positions = np.load(os.path.join(base_path, "recorded_human_position.npy"), allow_pickle=True).item()

    # 加载左右臂优化的关节角度
    optimized_joint_angles_l = np.load(os.path.join(base_path, "optimized_joint_angles_l.npy"), allow_pickle=True)
    optimized_joint_angles_r = np.load(os.path.join(base_path, "optimized_joint_angles_r.npy"), allow_pickle=True)

    # 如果存在，加载优化的人体工程学分数
    try:
        optimized_ergonomics_scores_l = np.load(os.path.join(base_path, "ergonomics_scores_l.npy"), allow_pickle=True)
        optimized_ergonomics_scores_r = np.load(os.path.join(base_path, "ergonomics_scores_r.npy"), allow_pickle=True)
        has_optimized_scores = True
    except:
        has_optimized_scores = False

    # 获取人体关节位置 - 左臂
    shoulder_positions_l = np.array(human_positions['shoulder_positions_l'])
    elbow_positions_l = np.array(human_positions['elbow_positions_l'])
    wrist_positions_l = np.array(human_positions['wrist_positions_l'])

    # 获取人体关节位置 - 右臂
    shoulder_positions_r = np.array(human_positions['shoulder_positions'])
    elbow_positions_r = np.array(human_positions['elbow_positions'])
    wrist_positions_r = np.array(human_positions['wrist_positions'])

    # 设置坐标变换 (如果需要)
    try:
        sub_robot = np.array([-0.2195, 1.11462, 0, 0, 0, 0, 1])
        T_optitrack2robotbase = np.linalg.inv(
            tsf.transform_optitrack_origin_to_optitrack_robot(
                sub_robot) @ tsf.transform_optitrack_robot_to_robot_base())
    except:
        T_optitrack2robotbase = np.eye(4)  # 如果无法计算变换，使用单位矩阵

    # 计算实际关节角 - 左臂
    actual_joint_angles_l = []
    for i in range(len(shoulder_positions_l)):
        shoulder_pos = shoulder_positions_l[i]
        elbow_pos = elbow_positions_l[i]
        wrist_pos = wrist_positions_l[i]

        # 转换到肩部坐标系
        p_elbow, p_wrist = trans_global2shoulder(shoulder_pos, elbow_pos, wrist_pos, arm='left')

        # 计算手臂尺寸
        d_ual, d_uar, d_lal, d_lar = mos.calculate_arm_dimensions(
            shoulder_pos, elbow_pos, wrist_pos,
            shoulder_pos, elbow_pos, wrist_pos  # 假设左右臂对称
        )

        # 使用逆运动学计算关节角度
        try:
            q = mos.inverse_kinematics(p_elbow, p_wrist, d_ual, d_lal)
            actual_joint_angles_l.append(q)
        except Exception as e:
            print(f"左臂帧 {i} 的逆运动学求解失败: {e}")
            # 如果失败，使用前一帧的角度或默认值
            if len(actual_joint_angles_l) > 0:
                actual_joint_angles_l.append(actual_joint_angles_l[-1])
            else:
                actual_joint_angles_l.append(np.zeros(4))

    # 计算实际关节角 - 右臂
    actual_joint_angles_r = []
    for i in range(len(shoulder_positions_r)):
        shoulder_pos = shoulder_positions_r[i]
        elbow_pos = elbow_positions_r[i]
        wrist_pos = wrist_positions_r[i]

        # 转换到肩部坐标系
        p_elbow, p_wrist = trans_global2shoulder(shoulder_pos, elbow_pos, wrist_pos, arm='right')

        # 计算手臂尺寸
        d_ual, d_uar, d_lal, d_lar = mos.calculate_arm_dimensions(
            shoulder_pos, elbow_pos, wrist_pos,
            shoulder_pos, elbow_pos, wrist_pos  # 假设左右臂对称
        )

        # 使用逆运动学计算关节角度
        try:
            q = mos.inverse_kinematics(p_elbow, p_wrist, d_uar, d_lar)
            actual_joint_angles_r.append(q)
        except Exception as e:
            print(f"右臂帧 {i} 的逆运动学求解失败: {e}")
            # 如果失败，使用前一帧的角度或默认值
            if len(actual_joint_angles_r) > 0:
                actual_joint_angles_r.append(actual_joint_angles_r[-1])
            else:
                actual_joint_angles_r.append(np.zeros(4))

    actual_joint_angles_l = np.array(actual_joint_angles_l)
    actual_joint_angles_r = np.array(actual_joint_angles_r)

    # 计算实际关节角的人体工程学分数 - 左臂
    actual_ergonomics_scores_l = []
    for q in actual_joint_angles_l:
        score = calculate_upper_limb_score_with_joint_angles(q)
        actual_ergonomics_scores_l.append(score)

    # 计算实际关节角的人体工程学分数 - 右臂
    actual_ergonomics_scores_r = []
    for q in actual_joint_angles_r:
        score = calculate_upper_limb_score_with_joint_angles(q)
        actual_ergonomics_scores_r.append(score)

    actual_ergonomics_scores_l = np.array(actual_ergonomics_scores_l)
    actual_ergonomics_scores_r = np.array(actual_ergonomics_scores_r)

    # 计算优化关节角的人体工程学分数 - 左臂
    optimized_ergo_calculated_l = []
    for q in optimized_joint_angles_l:
        score = calculate_upper_limb_score_with_joint_angles(q)
        optimized_ergo_calculated_l.append(score)

    # 计算优化关节角的人体工程学分数 - 右臂
    optimized_ergo_calculated_r = []
    for q in optimized_joint_angles_r:
        score = calculate_upper_limb_score_with_joint_angles(q)
        optimized_ergo_calculated_r.append(score)

    optimized_ergo_calculated_l = np.array(optimized_ergo_calculated_l)
    optimized_ergo_calculated_r = np.array(optimized_ergo_calculated_r)

    # 对齐数据长度（通过时间归一化）- 左臂
    t_actual_l = np.linspace(0, 1, len(actual_joint_angles_l))
    t_optimized_l = np.linspace(0, 1, len(optimized_joint_angles_l))

    # 对齐数据长度（通过时间归一化）- 右臂
    t_actual_r = np.linspace(0, 1, len(actual_joint_angles_r))
    t_optimized_r = np.linspace(0, 1, len(optimized_joint_angles_r))

    # 创建插值函数 - 左臂
    interpolated_optimized_angles_l = []
    for j in range(4):  # 对每个关节角度
        interp_func = interp1d(t_optimized_l, optimized_joint_angles_l[:, j], kind='linear')
        interpolated_angle = interp_func(t_actual_l)
        interpolated_optimized_angles_l.append(interpolated_angle)

    # 创建插值函数 - 右臂
    interpolated_optimized_angles_r = []
    for j in range(4):  # 对每个关节角度
        interp_func = interp1d(t_optimized_r, optimized_joint_angles_r[:, j], kind='linear')
        interpolated_angle = interp_func(t_actual_r)
        interpolated_optimized_angles_r.append(interpolated_angle)

    interpolated_optimized_angles_l = np.array(interpolated_optimized_angles_l).T
    interpolated_optimized_angles_r = np.array(interpolated_optimized_angles_r).T

    # 插值优化的人体工程学分数 - 左臂
    t_opt_ergo_l = np.linspace(0, 1, len(optimized_ergo_calculated_l))
    interp_func = interp1d(t_opt_ergo_l, optimized_ergo_calculated_l, kind='linear')
    interpolated_optimized_ergo_l = interp_func(t_actual_l)

    # 插值优化的人体工程学分数 - 右臂
    t_opt_ergo_r = np.linspace(0, 1, len(optimized_ergo_calculated_r))
    interp_func = interp1d(t_opt_ergo_r, optimized_ergo_calculated_r, kind='linear')
    interpolated_optimized_ergo_r = interp_func(t_actual_r)

    # 计算关节角误差（RMSE）- 左臂
    angle_errors_l = np.sqrt(np.mean((actual_joint_angles_l - interpolated_optimized_angles_l) ** 2, axis=0))

    # 计算关节角误差（RMSE）- 右臂
    angle_errors_r = np.sqrt(np.mean((actual_joint_angles_r - interpolated_optimized_angles_r) ** 2, axis=0))

    # 创建图形 - 左臂
    fig_l = plt.figure(figsize=(16, 12))
    gs_l = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 1])

    # 设置标题 - 左臂
    method_name = "CSEF-based Motion Planning" if method_num == 1 else "Point-to-Point Motion Planning"
    fig_l.suptitle(f"Experiment {experiment_num}: {method_name}\nLeft Arm Joint Angle Analysis and Ergonomics Scores",
                   fontsize=16)

    # 绘制关节角度对比 - 左臂
    joint_names = [
        "Shoulder Abduction",
        "Shoulder Flexion",
        "Shoulder Rotation",
        "Elbow Flexion"
    ]

    # 创建关节角度子图 - 左臂
    axs_angles_l = []
    for i in range(4):
        ax = plt.subplot(gs_l[i // 2, i % 2])
        axs_angles_l.append(ax)

        # 绘制实际和优化的关节角度
        ax.plot(t_actual_l, actual_joint_angles_l[:, i], 'b-', label='Actual', linewidth=2)
        ax.plot(t_actual_l, interpolated_optimized_angles_l[:, i], 'r-', label='Optimized', linewidth=2)

        # 添加标题和标签
        ax.set_title(f"{joint_names[i]} (RMSE: {angle_errors_l[i]:.3f} rad)")
        ax.set_xlabel('Normalized Time')
        ax.set_ylabel('Angle (rad)')
        ax.grid(False)
        ax.legend()

    # 绘制人体工程学分数 - 左臂
    ax_ergo_l = plt.subplot(gs_l[2, :])
    ax_ergo_l.plot(t_actual_l, actual_ergonomics_scores_l, 'b-', label='Actual Ergonomics Score', linewidth=2)
    ax_ergo_l.plot(t_actual_l, interpolated_optimized_ergo_l, 'r-', label='Optimized Ergonomics Score', linewidth=2)

    ax_ergo_l.set_title('Left Arm Ergonomics Scores Comparison')
    ax_ergo_l.set_xlabel('Normalized Time')
    ax_ergo_l.set_ylabel('Ergonomics Score')
    ax_ergo_l.grid(False)
    ax_ergo_l.legend()

    # 计算平均人体工程学分数 - 左臂
    avg_actual_score_l = np.mean(actual_ergonomics_scores_l)
    avg_optimized_score_l = np.mean(interpolated_optimized_ergo_l)
    score_improvement_l = ((avg_optimized_score_l - avg_actual_score_l) / avg_actual_score_l) * 100

    score_text_l = f"Average Scores - Actual: {avg_actual_score_l:.2f}, Optimized: {avg_optimized_score_l:.2f}\n"
    if score_improvement_l < 0:
        score_text_l += f"Improvement: {abs(score_improvement_l):.2f}% (lower score is better)"
    else:
        score_text_l += f"Degradation: {score_improvement_l:.2f}% (lower score is better)"

    ax_ergo_l.text(0.02, 0.02, score_text_l, transform=ax_ergo_l.transAxes, fontsize=10,
                   bbox=dict(facecolor='white', alpha=0.7))

    # 设置透明背景 - 左臂
    fig_l.patch.set_alpha(0.0)
    for ax in axs_angles_l:
        ax.patch.set_alpha(0.0)
        ax.set_facecolor('none')
    ax_ergo_l.patch.set_alpha(0.0)
    ax_ergo_l.set_facecolor('none')

    # 保存图像 - 左臂
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if output_dir:
        output_path_l = os.path.join(output_dir, f"joint_analysis_left_exp{experiment_num}_method{method_num}.png")
        plt.savefig(output_path_l, dpi=300, bbox_inches='tight', transparent=True)
        print(f"左臂关节分析图像已保存至: {output_path_l}")

    plt.close(fig_l)

    # 创建图形 - 右臂
    fig_r = plt.figure(figsize=(16, 12))
    gs_r = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 1])

    # 设置标题 - 右臂
    fig_r.suptitle(f"Experiment {experiment_num}: {method_name}\nRight Arm Joint Angle Analysis and Ergonomics Scores",
                   fontsize=16)

    # 创建关节角度子图 - 右臂
    axs_angles_r = []
    for i in range(4):
        ax = plt.subplot(gs_r[i // 2, i % 2])
        axs_angles_r.append(ax)

        # 绘制实际和优化的关节角度
        ax.plot(t_actual_r, actual_joint_angles_r[:, i], 'b-', label='Actual', linewidth=2)
        ax.plot(t_actual_r, interpolated_optimized_angles_r[:, i], 'r-', label='Optimized', linewidth=2)

        # 添加标题和标签
        ax.set_title(f"{joint_names[i]} (RMSE: {angle_errors_r[i]:.3f} rad)")
        ax.set_xlabel('Normalized Time')
        ax.set_ylabel('Angle (rad)')
        ax.grid(False)
        ax.legend()

    # 绘制人体工程学分数 - 右臂
    ax_ergo_r = plt.subplot(gs_r[2, :])
    ax_ergo_r.plot(t_actual_r, actual_ergonomics_scores_r, 'b-', label='Actual Ergonomics Score', linewidth=2)
    ax_ergo_r.plot(t_actual_r, interpolated_optimized_ergo_r, 'r-', label='Optimized Ergonomics Score', linewidth=2)

    ax_ergo_r.set_title('Right Arm Ergonomics Scores Comparison')
    ax_ergo_r.set_xlabel('Normalized Time')
    ax_ergo_r.set_ylabel('Ergonomics Score')
    ax_ergo_r.grid(False)
    ax_ergo_r.legend()

    # 计算平均人体工程学分数 - 右臂
    avg_actual_score_r = np.mean(actual_ergonomics_scores_r)
    avg_optimized_score_r = np.mean(interpolated_optimized_ergo_r)
    score_improvement_r = ((avg_optimized_score_r - avg_actual_score_r) / avg_actual_score_r) * 100

    score_text_r = f"Average Scores - Actual: {avg_actual_score_r:.2f}, Optimized: {avg_optimized_score_r:.2f}\n"
    if score_improvement_r < 0:
        score_text_r += f"Improvement: {abs(score_improvement_r):.2f}% (lower score is better)"
    else:
        score_text_r += f"Degradation: {score_improvement_r:.2f}% (lower score is better)"

    ax_ergo_r.text(0.02, 0.02, score_text_r, transform=ax_ergo_r.transAxes, fontsize=10,
                   bbox=dict(facecolor='white', alpha=0.7))

    # 设置透明背景 - 右臂
    fig_r.patch.set_alpha(0.0)
    for ax in axs_angles_r:
        ax.patch.set_alpha(0.0)
        ax.set_facecolor('none')
    ax_ergo_r.patch.set_alpha(0.0)
    ax_ergo_r.set_facecolor('none')

    # 保存图像 - 右臂
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if output_dir:
        output_path_r = os.path.join(output_dir, f"joint_analysis_right_exp{experiment_num}_method{method_num}.png")
        plt.savefig(output_path_r, dpi=300, bbox_inches='tight', transparent=True)
        print(f"右臂关节分析图像已保存至: {output_path_r}")

    plt.close(fig_r)

    # 创建综合图 - 左右臂人体工程学分数对比
    fig_both = plt.figure(figsize=(14, 10))
    ax_both = fig_both.add_subplot(111)

    # 绘制左右臂的人体工程学分数
    ax_both.plot(t_actual_l, actual_ergonomics_scores_l, 'b-', label='Left Arm - Actual', linewidth=2)
    ax_both.plot(t_actual_l, interpolated_optimized_ergo_l, 'b--', label='Left Arm - Optimized', linewidth=2)
    ax_both.plot(t_actual_r, actual_ergonomics_scores_r, 'r-', label='Right Arm - Actual', linewidth=2)
    ax_both.plot(t_actual_r, interpolated_optimized_ergo_r, 'r--', label='Right Arm - Optimized', linewidth=2)

    # 添加标题和标签
    ax_both.set_title(f'Experiment {experiment_num}: {method_name}\nDual-Arm Ergonomics Scores Comparison', fontsize=16)
    ax_both.set_xlabel('Normalized Time', fontsize=14)
    ax_both.set_ylabel('Ergonomics Score', fontsize=14)
    ax_both.grid(False)
    ax_both.legend(fontsize=12)

    # 添加平均分数信息
    combined_text = (f"Left Arm - Actual: {avg_actual_score_l:.2f}, Optimized: {avg_optimized_score_l:.2f} "
                     f"({'-' if score_improvement_l < 0 else '+'}{'Imp: ' if score_improvement_l < 0 else 'Deg: '}{abs(score_improvement_l):.2f}%)\n"
                     f"Right Arm - Actual: {avg_actual_score_r:.2f}, Optimized: {avg_optimized_score_r:.2f} "
                     f"({'-' if score_improvement_r < 0 else '+'}{'Imp: ' if score_improvement_r < 0 else 'Deg: '}{abs(score_improvement_r):.2f}%)\n"
                     f"Dual-Arm Average - Actual: {(avg_actual_score_l + avg_actual_score_r) / 2:.2f}, "
                     f"Optimized: {(avg_optimized_score_l + avg_optimized_score_r) / 2:.2f}")

    ax_both.text(0.02, 0.02, combined_text, transform=ax_both.transAxes, fontsize=12,
                 bbox=dict(facecolor='white', alpha=0.7))

    # 设置透明背景
    fig_both.patch.set_alpha(0.0)
    ax_both.patch.set_alpha(0.0)
    ax_both.set_facecolor('none')

    # 保存图像
    plt.tight_layout()

    if output_dir:
        output_path_both = os.path.join(output_dir, f"dual_arm_ergo_exp{experiment_num}_method{method_num}.png")
        plt.savefig(output_path_both, dpi=300, bbox_inches='tight', transparent=True)
        print(f"双臂人体工程学对比图像已保存至: {output_path_both}")

    plt.close(fig_both)

    return angle_errors_l, angle_errors_r, actual_ergonomics_scores_l, actual_ergonomics_scores_r


def compare_methods_dual_arm(experiment_num, output_dir=None):
    """
    比较双臂两种方法的人体工程学分数

    参数：
    experiment_num: 实验编号
    output_dir: 输出目录
    """
    # 获取两种方法的人体工程学分数
    _, _, actual_scores_csef_l, actual_scores_csef_r = analyze_joint_angles_and_ergonomics_dual_arm(experiment_num, 1)
    _, _, actual_scores_ptp_l, actual_scores_ptp_r = analyze_joint_angles_and_ergonomics_dual_arm(experiment_num, 2)

    # 创建归一化时间轴
    t_csef_l = np.linspace(0, 1, len(actual_scores_csef_l))
    t_csef_r = np.linspace(0, 1, len(actual_scores_csef_r))
    t_ptp_l = np.linspace(0, 1, len(actual_scores_ptp_l))
    t_ptp_r = np.linspace(0, 1, len(actual_scores_ptp_r))

    # 插值到相同长度进行比较
    t_common = np.linspace(0, 1, 100)  # 统一的时间轴

    # 创建插值函数
    interp_csef_l = interp1d(t_csef_l, actual_scores_csef_l, kind='linear')
    interp_csef_r = interp1d(t_csef_r, actual_scores_csef_r, kind='linear')
    interp_ptp_l = interp1d(t_ptp_l, actual_scores_ptp_l, kind='linear')
    interp_ptp_r = interp1d(t_ptp_r, actual_scores_ptp_r, kind='linear')

    # 插值数据
    scores_csef_l_interp = interp_csef_l(t_common)
    scores_csef_r_interp = interp_csef_r(t_common)
    scores_ptp_l_interp = interp_ptp_l(t_common)
    scores_ptp_r_interp = interp_ptp_r(t_common)

    # 创建左臂对比图
    fig_l, ax_l = plt.subplots(figsize=(12, 8))
    ax_l.plot(t_common, scores_csef_l_interp, 'b-', label='CSEF-based Method', linewidth=2)
    ax_l.plot(t_common, scores_ptp_l_interp, 'r-', label='Point-to-Point Method', linewidth=2)

    # 计算左臂平均分数和改进百分比
    avg_csef_l = np.mean(scores_csef_l_interp)
    avg_ptp_l = np.mean(scores_ptp_l_interp)
    improvement_l = ((avg_ptp_l - avg_csef_l) / avg_ptp_l) * 100  # 注意：较低的分数更好

    # 添加左臂标题和标签
    ax_l.set_title(f'Experiment {experiment_num}: Left Arm Ergonomics Scores Comparison Between Methods', fontsize=16)
    ax_l.set_xlabel('Normalized Time', fontsize=14)
    ax_l.set_ylabel('Ergonomics Score', fontsize=14)
    ax_l.grid(False)
    ax_l.legend(fontsize=12)

    # 添加左臂平均分数和改进信息
    score_text_l = f"Average Scores - CSEF: {avg_csef_l:.2f}, Point-to-Point: {avg_ptp_l:.2f}\n"
    if improvement_l > 0:
        score_text_l += f"CSEF is {abs(improvement_l):.2f}% better than Point-to-Point (lower score is better)"
    else:
        score_text_l += f"CSEF is {abs(improvement_l):.2f}% worse than Point-to-Point (lower score is better)"

    ax_l.text(0.02, 0.02, score_text_l, transform=ax_l.transAxes, fontsize=12,
              bbox=dict(facecolor='white', alpha=0.7))

    # 设置左臂图透明背景
    fig_l.patch.set_alpha(0.0)
    ax_l.patch.set_alpha(0.0)
    ax_l.set_facecolor('none')

    # 保存左臂图像
    if output_dir:
        output_path_l = os.path.join(output_dir, f"left_arm_methods_comparison_exp{experiment_num}.png")
        plt.savefig(output_path_l, dpi=300, bbox_inches='tight', transparent=True)
        print(f"左臂方法对比图像已保存至: {output_path_l}")

    plt.close(fig_l)

    # 创建右臂对比图
    fig_r, ax_r = plt.subplots(figsize=(12, 8))
    ax_r.plot(t_common, scores_csef_r_interp, 'b-', label='CSEF-based Method', linewidth=2)
    ax_r.plot(t_common, scores_ptp_r_interp, 'r-', label='Point-to-Point Method', linewidth=2)

    # 计算右臂平均分数和改进百分比
    avg_csef_r = np.mean(scores_csef_r_interp)
    avg_ptp_r = np.mean(scores_ptp_r_interp)
    improvement_r = ((avg_ptp_r - avg_csef_r) / avg_ptp_r) * 100  # 注意：较低的分数更好

    # 添加右臂标题和标签
    ax_r.set_title(f'Experiment {experiment_num}: Right Arm Ergonomics Scores Comparison Between Methods', fontsize=16)
    ax_r.set_xlabel('Normalized Time', fontsize=14)
    ax_r.set_ylabel('Ergonomics Score', fontsize=14)
    ax_r.grid(False)
    ax_r.legend(fontsize=12)

    # 添加右臂平均分数和改进信息
    score_text_r = f"Average Scores - CSEF: {avg_csef_r:.2f}, Point-to-Point: {avg_ptp_r:.2f}\n"
    if improvement_r > 0:
        score_text_r += f"CSEF is {abs(improvement_r):.2f}% better than Point-to-Point (lower score is better)"
    else:
        score_text_r += f"CSEF is {abs(improvement_r):.2f}% worse than Point-to-Point (lower score is better)"

    ax_r.text(0.02, 0.02, score_text_r, transform=ax_r.transAxes, fontsize=12,
              bbox=dict(facecolor='white', alpha=0.7))

    # 设置右臂图透明背景
    fig_r.patch.set_alpha(0.0)
    ax_r.patch.set_alpha(0.0)
    ax_r.set_facecolor('none')

    # 保存右臂图像
    if output_dir:
        output_path_r = os.path.join(output_dir, f"right_arm_methods_comparison_exp{experiment_num}.png")
        plt.savefig(output_path_r, dpi=300, bbox_inches='tight', transparent=True)
        print(f"右臂方法对比图像已保存至: {output_path_r}")

    plt.close(fig_r)

    # 创建双臂综合对比图
    fig_combined, ax_combined = plt.subplots(figsize=(14, 10))

    # 计算双臂平均分数
    scores_csef_avg = (scores_csef_l_interp + scores_csef_r_interp) / 2
    scores_ptp_avg = (scores_ptp_l_interp + scores_ptp_r_interp) / 2

    # 绘制双臂平均分数
    ax_combined.plot(t_common, scores_csef_avg, 'b-', label='CSEF-based Method (Dual-Arm Avg)', linewidth=3)
    ax_combined.plot(t_common, scores_ptp_avg, 'r-', label='Point-to-Point Method (Dual-Arm Avg)', linewidth=3)

    # 绘制左右臂单独分数（虚线）
    ax_combined.plot(t_common, scores_csef_l_interp, 'b--', label='CSEF - Left Arm', linewidth=1.5, alpha=0.7)
    ax_combined.plot(t_common, scores_csef_r_interp, 'b-.', label='CSEF - Right Arm', linewidth=1.5, alpha=0.7)
    ax_combined.plot(t_common, scores_ptp_l_interp, 'r--', label='Point-to-Point - Left Arm', linewidth=1.5, alpha=0.7)
    ax_combined.plot(t_common, scores_ptp_r_interp, 'r-.', label='Point-to-Point - Right Arm', linewidth=1.5, alpha=0.7)

    # 计算双臂平均改进
    avg_csef_combined = np.mean(scores_csef_avg)
    avg_ptp_combined = np.mean(scores_ptp_avg)
    improvement_combined = ((avg_ptp_combined - avg_csef_combined) / avg_ptp_combined) * 100

    # 添加标题和标签
    ax_combined.set_title(f'Experiment {experiment_num}: Dual-Arm Ergonomics Scores Comparison Between Methods',
                          fontsize=16)
    ax_combined.set_xlabel('Normalized Time', fontsize=14)
    ax_combined.set_ylabel('Ergonomics Score', fontsize=14)
    ax_combined.grid(False)
    ax_combined.legend(fontsize=12)

    # 添加双臂综合信息
    combined_text = (f"Left Arm Average - CSEF: {avg_csef_l:.2f}, Point-to-Point: {avg_ptp_l:.2f} "
                     f"({'Better' if improvement_l > 0 else 'Worse'} by {abs(improvement_l):.2f}%)\n"
                     f"Right Arm Average - CSEF: {avg_csef_r:.2f}, Point-to-Point: {avg_ptp_r:.2f} "
                     f"({'Better' if improvement_r > 0 else 'Worse'} by {abs(improvement_r):.2f}%)\n"
                     f"Dual-Arm Average - CSEF: {avg_csef_combined:.2f}, Point-to-Point: {avg_ptp_combined:.2f} "
                     f"({'Better' if improvement_combined > 0 else 'Worse'} by {abs(improvement_combined):.2f}%)")

    ax_combined.text(0.02, 0.02, combined_text, transform=ax_combined.transAxes, fontsize=12,
                     bbox=dict(facecolor='white', alpha=0.7))

    # 设置透明背景
    fig_combined.patch.set_alpha(0.0)
    ax_combined.patch.set_alpha(0.0)
    ax_combined.set_facecolor('none')

    # 保存图像
    if output_dir:
        output_path_combined = os.path.join(output_dir, f"dual_arm_methods_comparison_exp{experiment_num}.png")
        plt.savefig(output_path_combined, dpi=300, bbox_inches='tight', transparent=True)
        print(f"双臂方法对比综合图像已保存至: {output_path_combined}")

    plt.close(fig_combined)

    return scores_csef_l_interp, scores_csef_r_interp, scores_ptp_l_interp, scores_ptp_r_interp


def plot_dual_arm_ergonomics_summary(experiments, methods, output_dir=None):
    """
    绘制双臂不同方法的人体工程学评分时间序列对比图，包括平均值和变化范围

    参数：
    experiments: 实验编号列表
    methods: 方法编号列表
    output_dir: 输出目录
    """
    # 存储每种方法的所有分数
    all_scores_l = {method: [] for method in methods}  # 左臂
    all_scores_r = {method: [] for method in methods}  # 右臂
    method_names = {1: "CSEF-based Method", 2: "Point-to-Point Method"}
    method_colors = {1: "blue", 2: "red"}

    # 收集所有实验的分数
    for exp in experiments:
        for method in methods:
            _, _, scores_l, scores_r = analyze_joint_angles_and_ergonomics_dual_arm(exp, method)

            # 归一化到0-1时间范围 - 左臂
            t_norm = np.linspace(0, 1, 100)
            t_orig_l = np.linspace(0, 1, len(scores_l))
            interp_func_l = interp1d(t_orig_l, scores_l, kind='linear')
            norm_scores_l = interp_func_l(t_norm)
            all_scores_l[method].append(norm_scores_l)

            # 归一化到0-1时间范围 - 右臂
            t_orig_r = np.linspace(0, 1, len(scores_r))
            interp_func_r = interp1d(t_orig_r, scores_r, kind='linear')
            norm_scores_r = interp_func_r(t_norm)
            all_scores_r[method].append(norm_scores_r)

    # 转换为numpy数组以便计算
    for method in methods:
        all_scores_l[method] = np.array(all_scores_l[method])
        all_scores_r[method] = np.array(all_scores_r[method])

    # 创建图形 - 左臂
    fig_l, ax_l = plt.subplots(figsize=(12, 8))

    # 创建图形 - 右臂
    fig_r, ax_r = plt.subplots(figsize=(12, 8))

    # 创建图形 - 双臂平均
    fig_combined, ax_combined = plt.subplots(figsize=(14, 10))

    # 时间轴
    time_points = np.linspace(0, 1, 100)

    # 左臂和右臂的组合数据
    all_scores_combined = {method: [] for method in methods}

    # 对每种方法绘制平均曲线和置信区间
    for method in methods:
        # 左臂处理
        method_scores_l = all_scores_l[method]
        mean_curve_l = np.mean(method_scores_l, axis=0)  # 每个时间点的平均值
        std_curve_l = np.std(method_scores_l, axis=0)  # 每个时间点的标准差

        # 计算左臂95%置信区间
        lower_bound_l = mean_curve_l - 1.96 * std_curve_l / np.sqrt(len(experiments))
        upper_bound_l = mean_curve_l + 1.96 * std_curve_l / np.sqrt(len(experiments))

        # 绘制左臂平均曲线
        ax_l.plot(time_points, mean_curve_l, '-', color=method_colors[method],
                  linewidth=2, label=f"{method_names[method]} (Mean)")

        # 绘制左臂置信区间
        ax_l.fill_between(time_points, lower_bound_l, upper_bound_l,
                          color=method_colors[method], alpha=0.2,
                          label=f"{method_names[method]} (95% CI)")

        # 右臂处理
        method_scores_r = all_scores_r[method]
        mean_curve_r = np.mean(method_scores_r, axis=0)  # 每个时间点的平均值
        std_curve_r = np.std(method_scores_r, axis=0)  # 每个时间点的标准差

        # 计算右臂95%置信区间
        lower_bound_r = mean_curve_r - 1.96 * std_curve_r / np.sqrt(len(experiments))
        upper_bound_r = mean_curve_r + 1.96 * std_curve_r / np.sqrt(len(experiments))

        # 绘制右臂平均曲线
        ax_r.plot(time_points, mean_curve_r, '-', color=method_colors[method],
                  linewidth=2, label=f"{method_names[method]} (Mean)")

        # 绘制右臂置信区间
        ax_r.fill_between(time_points, lower_bound_r, upper_bound_r,
                          color=method_colors[method], alpha=0.2,
                          label=f"{method_names[method]} (95% CI)")

        # 计算双臂平均值
        all_scores_combined[method] = (method_scores_l + method_scores_r) / 2
        mean_curve_combined = np.mean(all_scores_combined[method], axis=0)  # 每个时间点的平均值
        std_curve_combined = np.std(all_scores_combined[method], axis=0)  # 每个时间点的标准差

        # 计算双臂95%置信区间
        lower_bound_combined = mean_curve_combined - 1.96 * std_curve_combined / np.sqrt(len(experiments))
        upper_bound_combined = mean_curve_combined + 1.96 * std_curve_combined / np.sqrt(len(experiments))

        # 绘制双臂平均曲线
        ax_combined.plot(time_points, mean_curve_combined, '-', color=method_colors[method],
                         linewidth=2, label=f"{method_names[method]} (Mean)")

        # 绘制双臂置信区间
        ax_combined.fill_between(time_points, lower_bound_combined, upper_bound_combined,
                                 color=method_colors[method], alpha=0.2,
                                 label=f"{method_names[method]} (95% CI)")

    # 计算每种方法的总平均分数 - 左臂
    method_means_l = {}
    method_stds_l = {}
    for method in methods:
        method_scores_l = all_scores_l[method]
        method_means_l[method] = np.mean(method_scores_l)  # 所有时间点和实验的平均值
        method_stds_l[method] = np.std(np.mean(method_scores_l, axis=1))  # 实验间的标准差

    # 计算每种方法的总平均分数 - 右臂
    method_means_r = {}
    method_stds_r = {}
    for method in methods:
        method_scores_r = all_scores_r[method]
        method_means_r[method] = np.mean(method_scores_r)  # 所有时间点和实验的平均值
        method_stds_r[method] = np.std(np.mean(method_scores_r, axis=1))  # 实验间的标准差

    # 计算每种方法的总平均分数 - 双臂平均
    method_means_combined = {}
    method_stds_combined = {}
    for method in methods:
        method_scores_combined = all_scores_combined[method]
        method_means_combined[method] = np.mean(method_scores_combined)  # 所有时间点和实验的平均值
        method_stds_combined[method] = np.std(np.mean(method_scores_combined, axis=1))  # 实验间的标准差

    # 添加左臂比较文本
    comparison_text_l = "Left Arm Average Scores:\n"
    for method in methods:
        comparison_text_l += f"{method_names[method]}: {method_means_l[method]:.2f} ± {method_stds_l[method]:.2f}\n"

    if len(methods) > 1:
        improvement_l = ((method_means_l[2] - method_means_l[1]) / method_means_l[2]) * 100
        if improvement_l > 0:
            comparison_text_l += f"\nCSEF is {improvement_l:.2f}% better than Point-to-Point"
        else:
            comparison_text_l += f"\nCSEF is {abs(improvement_l):.2f}% worse than Point-to-Point"
        comparison_text_l += " (lower score is better)"

    ax_l.text(0.02, 0.02, comparison_text_l, transform=ax_l.transAxes, fontsize=12,
              bbox=dict(facecolor='white', alpha=0.7))

    # 添加右臂比较文本
    comparison_text_r = "Right Arm Average Scores:\n"
    for method in methods:
        comparison_text_r += f"{method_names[method]}: {method_means_r[method]:.2f} ± {method_stds_r[method]:.2f}\n"

    if len(methods) > 1:
        improvement_r = ((method_means_r[2] - method_means_r[1]) / method_means_r[2]) * 100
        if improvement_r > 0:
            comparison_text_r += f"\nCSEF is {improvement_r:.2f}% better than Point-to-Point"
        else:
            comparison_text_r += f"\nCSEF is {abs(improvement_r):.2f}% worse than Point-to-Point"
        comparison_text_r += " (lower score is better)"

    ax_r.text(0.02, 0.02, comparison_text_r, transform=ax_r.transAxes, fontsize=12,
              bbox=dict(facecolor='white', alpha=0.7))

    # 添加双臂平均比较文本
    comparison_text_combined = "Dual-Arm Average Scores:\n"
    for method in methods:
        comparison_text_combined += f"{method_names[method]}: {method_means_combined[method]:.2f} ± {method_stds_combined[method]:.2f}\n"

    if len(methods) > 1:
        improvement_combined = ((method_means_combined[2] - method_means_combined[1]) / method_means_combined[2]) * 100
        if improvement_combined > 0:
            comparison_text_combined += f"\nCSEF is {improvement_combined:.2f}% better than Point-to-Point"
        else:
            comparison_text_combined += f"\nCSEF is {abs(improvement_combined):.2f}% worse than Point-to-Point"
        comparison_text_combined += " (lower score is better)"

        # 添加左右臂比较
        comparison_text_combined += f"\n\nLeft Arm - CSEF: {method_means_l[1]:.2f}, Point-to-Point: {method_means_l[2]:.2f} "
        comparison_text_combined += f"({'Better' if improvement_l > 0 else 'Worse'} by {abs(improvement_l):.2f}%)"
        comparison_text_combined += f"\nRight Arm - CSEF: {method_means_r[1]:.2f}, Point-to-Point: {method_means_r[2]:.2f} "
        comparison_text_combined += f"({'Better' if improvement_r > 0 else 'Worse'} by {abs(improvement_r):.2f}%)"

    ax_combined.text(0.02, 0.02, comparison_text_combined, transform=ax_combined.transAxes, fontsize=12,
                     bbox=dict(facecolor='white', alpha=0.7))

    # 设置左臂轴标签和标题
    ax_l.set_xlabel('Normalized Time', fontsize=14)
    ax_l.set_ylabel('Ergonomics Score', fontsize=14)
    ax_l.set_title('Comparison of Left Arm Ergonomics Scores Across Methods (with 95% CI)', fontsize=16)
    ax_l.grid(False)
    ax_l.legend(fontsize=12)

    # 设置右臂轴标签和标题
    ax_r.set_xlabel('Normalized Time', fontsize=14)
    ax_r.set_ylabel('Ergonomics Score', fontsize=14)
    ax_r.set_title('Comparison of Right Arm Ergonomics Scores Across Methods (with 95% CI)', fontsize=16)
    ax_r.grid(False)
    ax_r.legend(fontsize=12)

    # 设置双臂轴标签和标题
    ax_combined.set_xlabel('Normalized Time', fontsize=14)
    ax_combined.set_ylabel('Ergonomics Score', fontsize=14)
    ax_combined.set_title('Comparison of Dual-Arm Ergonomics Scores Across Methods (with 95% CI)', fontsize=16)
    ax_combined.grid(False)
    ax_combined.legend(fontsize=12)

    # 设置透明背景 - 左臂
    fig_l.patch.set_alpha(0.0)
    ax_l.patch.set_alpha(0.0)
    ax_l.set_facecolor('none')

    # 设置透明背景 - 右臂
    fig_r.patch.set_alpha(0.0)
    ax_r.patch.set_alpha(0.0)
    ax_r.set_facecolor('none')

    # 设置透明背景 - 双臂
    fig_combined.patch.set_alpha(0.0)
    ax_combined.patch.set_alpha(0.0)
    ax_combined.set_facecolor('none')

    # 保存图像 - 左臂
    plt.figure(fig_l.number)
    plt.tight_layout()
    if output_dir:
        output_path_l = os.path.join(output_dir, "left_arm_ergonomics_summary.png")
        plt.savefig(output_path_l, dpi=300, bbox_inches='tight', transparent=True)
        print(f"左臂人体工程学总结图像已保存至: {output_path_l}")

    # 保存图像 - 右臂
    plt.figure(fig_r.number)
    plt.tight_layout()
    if output_dir:
        output_path_r = os.path.join(output_dir, "right_arm_ergonomics_summary.png")
        plt.savefig(output_path_r, dpi=300, bbox_inches='tight', transparent=True)
        print(f"右臂人体工程学总结图像已保存至: {output_path_r}")

    # 保存图像 - 双臂平均
    plt.figure(fig_combined.number)
    plt.tight_layout()
    if output_dir:
        output_path_combined = os.path.join(output_dir, "dual_arm_ergonomics_summary.png")
        plt.savefig(output_path_combined, dpi=300, bbox_inches='tight', transparent=True)
        print(f"双臂人体工程学总结图像已保存至: {output_path_combined}")

    plt.close(fig_l)
    plt.close(fig_r)
    plt.close(fig_combined)

    return fig_l, fig_r, fig_combined


def main():
    # 指定要处理的实验和方法
    experiments = [1, 2, 3]  # 实验编号
    methods = [1, 2]  # 1: CSEF, 2: Point-to-Point

    # 创建保存图像的目录
    output_dir = "/home/ubuntu/Ergo-Manip/figures/box_carrying/yiming/4"
    os.makedirs(output_dir, exist_ok=True)

    # 分析每个实验和方法的关节角度和人体工程学分数
    for exp in experiments:
        for method in methods:
            angle_errors_l, angle_errors_r, _, _ = analyze_joint_angles_and_ergonomics_dual_arm(exp, method, output_dir)
            print(f"实验 {exp}, 方法 {method}:")
            print(f"  左臂关节角度RMSE: {angle_errors_l}")
            print(f"  右臂关节角度RMSE: {angle_errors_r}")

        # 比较两种方法的人体工程学分数
        compare_methods_dual_arm(exp, output_dir)

    # 绘制所有实验的人体工程学分数总结对比图
    plot_dual_arm_ergonomics_summary(experiments, methods, output_dir)

    print("所有分析已完成！")


if __name__ == "__main__":
    main()