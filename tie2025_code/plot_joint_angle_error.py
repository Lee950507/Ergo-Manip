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


def analyze_joint_angles_and_ergonomics(experiment_num, method_num, output_dir=None):
    """
    分析关节角误差和人体工程学分数

    参数：
    experiment_num: 实验编号
    method_num: 方法编号 (1: CSEF, 2: Point-to-Point)
    output_dir: 输出目录
    """
    # 设置数据路径
    base_path = f"/home/ubuntu/Ergo-Manip/data/drilling/0921/wuxi/3/{method_num}.{experiment_num}"

    # 加载数据
    human_positions = np.load(os.path.join(base_path, "recorded_human_position.npy"), allow_pickle=True).item()
    optimized_joint_angles = np.load(os.path.join(base_path, "optimized_joint_angles.npy"), allow_pickle=True)

    # 如果存在，加载优化的人体工程学分数
    try:
        optimized_ergonomics_scores = np.load(os.path.join(base_path, "ergonomics_scores.npy"), allow_pickle=True)
        has_optimized_scores = True
    except:
        has_optimized_scores = False

    # 获取人体关节位置
    shoulder_positions = np.array(human_positions['shoulder_positions'])
    elbow_positions = np.array(human_positions['elbow_positions'])
    wrist_positions = np.array(human_positions['wrist_positions'])

    # 设置坐标变换 (如果需要)
    try:
        sub_robot = np.array([-0.2195, 1.11462, 0, 0, 0, 0, 1])
        T_optitrack2robotbase = np.linalg.inv(
            tsf.transform_optitrack_origin_to_optitrack_robot(
                sub_robot) @ tsf.transform_optitrack_robot_to_robot_base())
    except:
        T_optitrack2robotbase = np.eye(4)  # 如果无法计算变换，使用单位矩阵

    # 计算实际关节角
    actual_joint_angles = []
    human_wrist_positions = []  # 肩部坐标系中的手腕位置

    for i in range(len(shoulder_positions)):
        shoulder_pos = shoulder_positions[i]
        elbow_pos = elbow_positions[i]
        wrist_pos = wrist_positions[i]

        # 转换到肩部坐标系
        p_elbow, p_wrist = trans_global2shoulder(shoulder_pos, elbow_pos, wrist_pos, arm='right')
        human_wrist_positions.append(p_wrist)

        # 计算手臂尺寸
        d_ual, d_uar, d_lal, d_lar = mos.calculate_arm_dimensions(
            shoulder_pos, elbow_pos, wrist_pos,
            shoulder_pos, elbow_pos, wrist_pos  # 假设左右臂对称
        )

        # 使用逆运动学计算关节角度
        try:
            q = mos.inverse_kinematics(p_elbow, p_wrist, d_uar, d_lar)
            actual_joint_angles.append(q)
        except Exception as e:
            print(f"帧 {i} 的逆运动学求解失败: {e}")
            # 如果失败，使用前一帧的角度或默认值
            if len(actual_joint_angles) > 0:
                actual_joint_angles.append(actual_joint_angles[-1])
            else:
                actual_joint_angles.append(np.zeros(4))

    actual_joint_angles = np.array(actual_joint_angles)

    # 计算优化关节角对应的手腕位置 (用于验证)
    optimized_wrist_positions = []
    for i in range(len(optimized_joint_angles)):
        if i < len(shoulder_positions):
            shoulder_pos = shoulder_positions[i]
            elbow_pos = elbow_positions[i]
            wrist_pos = wrist_positions[i]

            d_ual, d_uar, d_lal, d_lar = mos.calculate_arm_dimensions(
                shoulder_pos, elbow_pos, wrist_pos,
                shoulder_pos, elbow_pos, wrist_pos
            )

            # 使用前向运动学计算优化的手腕位置
            _, opt_wrist_pos = mos.forward_kinematics(optimized_joint_angles[i], d_uar, d_lar)
            optimized_wrist_positions.append(opt_wrist_pos)

    optimized_wrist_positions = np.array(optimized_wrist_positions)

    # 计算实际关节角的人体工程学分数
    actual_ergonomics_scores = []
    for q in actual_joint_angles:
        score = calculate_upper_limb_score_with_joint_angles(q)
        actual_ergonomics_scores.append(score)

    actual_ergonomics_scores = np.array(actual_ergonomics_scores)

    # 计算优化关节角的人体工程学分数
    optimized_ergo_calculated = []
    for q in optimized_joint_angles:
        score = calculate_upper_limb_score_with_joint_angles(q)
        optimized_ergo_calculated.append(score)

    optimized_ergo_calculated = np.array(optimized_ergo_calculated)

    # 对齐数据长度（通过时间归一化）
    # 创建时间轴（归一化到[0,1]）
    t_actual = np.linspace(0, 1, len(actual_joint_angles))
    t_optimized = np.linspace(0, 1, len(optimized_joint_angles))

    # 创建插值函数
    interpolated_optimized_angles = []
    for j in range(4):  # 对每个关节角度
        interp_func = interp1d(t_optimized, optimized_joint_angles[:, j], kind='linear')
        interpolated_angle = interp_func(t_actual)
        interpolated_optimized_angles.append(interpolated_angle)

    interpolated_optimized_angles = np.array(interpolated_optimized_angles).T

    # 插值优化的人体工程学分数
    t_opt_ergo = np.linspace(0, 1, len(optimized_ergo_calculated))
    interp_func = interp1d(t_opt_ergo, optimized_ergo_calculated, kind='linear')
    interpolated_optimized_ergo = interp_func(t_actual)

    # 插值原始优化的人体工程学分数（如果有）
    if has_optimized_scores:
        t_opt_scores = np.linspace(0, 1, len(optimized_ergonomics_scores))
        interp_func = interp1d(t_opt_scores, optimized_ergonomics_scores, kind='linear')
        interpolated_original_opt_scores = interp_func(t_actual)

    # 计算关节角误差（RMSE）
    angle_errors = np.sqrt(np.mean((actual_joint_angles - interpolated_optimized_angles) ** 2, axis=0))

    # 创建图形
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 1])

    # 设置标题
    method_name = "CSEF-based Motion Planning" if method_num == 1 else "Point-to-Point Motion Planning"
    fig.suptitle(f"Experiment {experiment_num}: {method_name}\nJoint Angle Analysis and Ergonomics Scores", fontsize=16)

    # 绘制关节角度对比
    joint_names = [
        "Shoulder Abduction",
        "Shoulder Flexion",
        "Shoulder Rotation",
        "Elbow Flexion"
    ]

    # 创建关节角度子图
    axs_angles = []
    for i in range(4):
        ax = plt.subplot(gs[i // 2, i % 2])
        axs_angles.append(ax)

        # 绘制实际和优化的关节角度
        ax.plot(t_actual, actual_joint_angles[:, i], 'b-', label='Actual', linewidth=2)
        ax.plot(t_actual, interpolated_optimized_angles[:, i], 'r-', label='Optimized', linewidth=2)

        # 添加标题和标签
        ax.set_title(f"{joint_names[i]} (RMSE: {angle_errors[i]:.3f} rad)")
        ax.set_xlabel('Normalized Time')
        ax.set_ylabel('Angle (rad)')
        ax.grid(False)
        ax.legend()

    # 绘制人体工程学分数
    ax_ergo = plt.subplot(gs[2, :])
    ax_ergo.plot(t_actual, actual_ergonomics_scores, 'b-', label='Actual Ergonomics Score', linewidth=2)
    ax_ergo.plot(t_actual, interpolated_optimized_ergo, 'r-', label='Optimized Ergonomics Score', linewidth=2)

    if has_optimized_scores:
        ax_ergo.plot(t_actual, interpolated_original_opt_scores, 'g--',
                     label='Original Optimized Score', linewidth=2)

    ax_ergo.set_title('Ergonomics Scores Comparison')
    ax_ergo.set_xlabel('Normalized Time')
    ax_ergo.set_ylabel('Ergonomics Score')
    ax_ergo.grid(False)
    ax_ergo.legend()

    # 计算平均人体工程学分数
    avg_actual_score = np.mean(actual_ergonomics_scores)
    avg_optimized_score = np.mean(interpolated_optimized_ergo)
    score_improvement = ((avg_optimized_score - avg_actual_score) / avg_actual_score) * 100

    score_text = f"Average Scores - Actual: {avg_actual_score:.2f}, Optimized: {avg_optimized_score:.2f}\n"
    if score_improvement < 0:
        score_text += f"Improvement: {abs(score_improvement):.2f}% (lower score is better)"
    else:
        score_text += f"Degradation: {score_improvement:.2f}% (lower score is better)"

    ax_ergo.text(0.02, 0.02, score_text, transform=ax_ergo.transAxes, fontsize=10,
                 bbox=dict(facecolor='white', alpha=0.7))

    # 设置透明背景
    fig.patch.set_alpha(0.0)
    for ax in axs_angles:
        ax.patch.set_alpha(0.0)
        ax.set_facecolor('none')
    ax_ergo.patch.set_alpha(0.0)
    ax_ergo.set_facecolor('none')

    # 保存图像
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if output_dir:
        output_path = os.path.join(output_dir, f"joint_analysis_exp{experiment_num}_method{method_num}.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight', transparent=True)
        print(f"关节分析图像已保存至: {output_path}")

    return fig, angle_errors, actual_ergonomics_scores


def compare_ergonomics_methods(experiment_num, output_dir=None):
    """
    比较两种方法的人体工程学分数

    参数：
    experiment_num: 实验编号
    output_dir: 输出目录
    """
    # 获取两种方法的人体工程学分数
    _, _, actual_scores_csef = analyze_joint_angles_and_ergonomics(experiment_num, 1)
    _, _, actual_scores_ptp = analyze_joint_angles_and_ergonomics(experiment_num, 2)

    # 创建归一化时间轴
    t_csef = np.linspace(0, 1, len(actual_scores_csef))
    t_ptp = np.linspace(0, 1, len(actual_scores_ptp))

    # 插值到相同长度进行比较
    t_common = np.linspace(0, 1, 100)  # 统一的时间轴

    interp_csef = interp1d(t_csef, actual_scores_csef, kind='linear')
    interp_ptp = interp1d(t_ptp, actual_scores_ptp, kind='linear')

    scores_csef_interp = interp_csef(t_common)
    scores_ptp_interp = interp_ptp(t_common)

    # 创建图形
    fig, ax = plt.subplots(figsize=(12, 8))

    # 绘制人体工程学分数
    ax.plot(t_common, scores_csef_interp, 'b-', label='CSEF-based Method', linewidth=2)
    ax.plot(t_common, scores_ptp_interp, 'r-', label='Point-to-Point Method', linewidth=2)

    # 计算平均分数和改进百分比
    avg_csef = np.mean(scores_csef_interp)
    avg_ptp = np.mean(scores_ptp_interp)
    improvement = ((avg_ptp - avg_csef) / avg_ptp) * 100  # 注意：较低的分数更好

    # 添加标题和标签
    ax.set_title(f'Experiment {experiment_num}: Ergonomics Scores Comparison Between Methods', fontsize=16)
    ax.set_xlabel('Normalized Time', fontsize=14)
    ax.set_ylabel('Ergonomics Score', fontsize=14)
    ax.grid(False)
    ax.legend(fontsize=12)

    # 添加平均分数和改进信息
    score_text = f"Average Scores - CSEF: {avg_csef:.2f}, Point-to-Point: {avg_ptp:.2f}\n"
    if improvement > 0:
        score_text += f"CSEF is {abs(improvement):.2f}% better than Point-to-Point (lower score is better)"
    else:
        score_text += f"CSEF is {abs(improvement):.2f}% worse than Point-to-Point (lower score is better)"

    ax.text(0.02, 0.02, score_text, transform=ax.transAxes, fontsize=12,
            bbox=dict(facecolor='white', alpha=0.7))

    # 设置透明背景
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    ax.set_facecolor('none')

    # 保存图像
    if output_dir:
        output_path = os.path.join(output_dir, f"ergonomics_comparison_exp{experiment_num}.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight', transparent=True)
        print(f"人体工程学对比图像已保存至: {output_path}")

    return fig


def plot_ergonomics_summary(experiments, methods, output_dir=None):
    """
    绘制不同方法的人体工程学评分时间序列对比图，包括平均值和变化范围

    参数：
    experiments: 实验编号列表
    methods: 方法编号列表
    output_dir: 输出目录
    """
    # 存储每种方法的所有分数
    all_scores = {method: [] for method in methods}
    method_names = {1: "CSEF-based Method", 2: "Point-to-Point Method"}
    method_colors = {1: "blue", 2: "red"}

    # 收集所有实验的分数
    for exp in experiments:
        for method in methods:
            _, _, scores = analyze_joint_angles_and_ergonomics(exp, method)

            # 归一化到0-1时间范围
            t_norm = np.linspace(0, 1, 100)
            t_orig = np.linspace(0, 1, len(scores))
            interp_func = interp1d(t_orig, scores, kind='linear')
            norm_scores = interp_func(t_norm)

            all_scores[method].append(norm_scores)

    # 转换为numpy数组以便计算
    for method in methods:
        all_scores[method] = np.array(all_scores[method])

    # 创建图形
    fig, ax = plt.subplots(figsize=(12, 8))

    # 时间轴
    time_points = np.linspace(0, 1, 100)

    # 对每种方法绘制平均曲线和置信区间
    for method in methods:
        method_scores = all_scores[method]
        mean_curve = np.mean(method_scores, axis=0)  # 每个时间点的平均值
        std_curve = np.std(method_scores, axis=0)  # 每个时间点的标准差

        # 计算95%置信区间
        lower_bound = mean_curve - 1.96 * std_curve / np.sqrt(len(experiments))
        upper_bound = mean_curve + 1.96 * std_curve / np.sqrt(len(experiments))

        # 绘制平均曲线
        ax.plot(time_points, mean_curve, '-', color=method_colors[method],
                linewidth=2, label=f"{method_names[method]} (Mean)")

        # 绘制置信区间
        ax.fill_between(time_points, lower_bound, upper_bound,
                        color=method_colors[method], alpha=0.2,
                        label=f"{method_names[method]} (95% CI)")

    # 计算每种方法的总平均分数
    method_means = {}
    method_stds = {}
    for method in methods:
        method_scores = all_scores[method]
        method_means[method] = np.mean(method_scores)  # 所有时间点和实验的平均值
        method_stds[method] = np.std(np.mean(method_scores, axis=1))  # 实验间的标准差

    # 添加比较文本
    comparison_text = "Average Scores:\n"
    for method in methods:
        comparison_text += f"{method_names[method]}: {method_means[method]:.2f} ± {method_stds[method]:.2f}\n"

    if len(methods) > 1:
        improvement = ((method_means[2] - method_means[1]) / method_means[2]) * 100
        if improvement > 0:
            comparison_text += f"\nCSEF is {improvement:.2f}% better than Point-to-Point"
        else:
            comparison_text += f"\nCSEF is {abs(improvement):.2f}% worse than Point-to-Point"
        comparison_text += " (lower score is better)"

    ax.text(0.02, 0.02, comparison_text, transform=ax.transAxes, fontsize=12,
            bbox=dict(facecolor='white', alpha=0.7))

    # 设置轴标签和标题
    ax.set_xlabel('Normalized Time', fontsize=14)
    ax.set_ylabel('Ergonomics Score', fontsize=14)
    ax.set_title('Comparison of Ergonomics Scores Across Methods (with 95% CI)', fontsize=16)
    ax.grid(False)
    ax.legend(fontsize=12)

    # 设置透明背景
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    ax.set_facecolor('none')

    # 保存图像
    plt.tight_layout()
    if output_dir:
        output_path = os.path.join(output_dir, "ergonomics_summary.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight', transparent=True)
        print(f"人体工程学总结图像已保存至: {output_path}")

    return fig


def main():
    # 指定要处理的实验和方法
    experiments = [1, 2, 3]  # 实验编号
    methods = [1, 2]  # 1: CSEF, 2: Point-to-Point

    # 创建保存图像的目录
    output_dir = "/home/ubuntu/Ergo-Manip/figures/drilling/wuxi/3"
    os.makedirs(output_dir, exist_ok=True)

    # 分析每个实验和方法的关节角度和人体工程学分数
    for exp in experiments:
        for method in methods:
            _, angle_errors, _ = analyze_joint_angles_and_ergonomics(exp, method, output_dir)
            print(f"实验 {exp}, 方法 {method} 的关节角度RMSE: {angle_errors}")

        # 比较两种方法的人体工程学分数
        compare_ergonomics_methods(exp, output_dir)

    # 绘制所有实验的人体工程学分数总结对比图
    plot_ergonomics_summary(experiments, methods, output_dir)

    print("所有分析已完成！")


if __name__ == "__main__":
    main()