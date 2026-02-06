#!/usr/bin/env python3
"""
对比四种方法的实验数据：3D 手腕轨迹、关节角变化、ergonomics_scores。
数据目录：0204/chenzui 下的 1_2, 2_5, 3_3, 4_2。
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 脚本所在目录 = data/composite_field，项目根目录用于导入 cf_plan
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.abspath(os.path.join(_script_dir, '..', '..'))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
import motion_planning_composite_filed_moving_base_ros as cf_plan

_base_dir = os.path.join(_script_dir, '0204', 'chenzui')

# 3D 图中可视化的 target goal 与 reference trajectory（与实验规划一致）
task_goal_global = np.array([1.15, 0.2, 1.05])
reference_trajectory = cf_plan.generate_reference_trajectory(
    task_goal_global + np.array([0.0, 0.0, 0.3]), task_goal_global,
    num_points=100, trajectory_type='straight')
reference_trajectory = np.asarray(reference_trajectory)

# 四组数据文件夹及对应方法名（1=Straight, 2=TSEF, 3=HD-SDF, 4=CF）
FOLDER_METHODS = [
    ('1_2', 'Method 1 (Straight)'),
    ('2_5', 'Method 2 (TSEF)'),
    ('3_3', 'Method 3 (HD-SDF)'),
    ('4_2', 'Method 4 (CF)'),
]

COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
MARKERS = ['o', 's', '^', 'D']

# ---------------------------------------------------------------------------
# 数据段截取：对每种方法只使用指定范围内的数据，去掉首尾不需要的段落。
# 键 = 文件夹名（如 '1_2'），值 = (start, end)，未列出的文件夹使用全部数据。
#
# (start, end) 两种写法：
#   - 按索引（整数）：[start, end)，例如 (0, 300) 表示取第 0 到第 299 个点；
#   - 按时间（秒，浮点数）：取 timestamps 在 [start, end] 内的点，例如 (2.0, 25.5)。
# 任一为 None 表示不限制该端，例如 (10, None) 表示从第 10 个点取到末尾，(None, 100) 表示从开头取到第 100 个点。
#
# 示例（按需修改后运行）：
SEGMENTS = {
      '1_2': (10.0, 17.0),         # 方法1：取索引 0~299
      '2_5': (10.0, 17.0),     # 方法2：取 2s~30s 之间的数据
      '3_3': (10.0, 17.0),      # 方法3：从第 50 个点取到末尾
      '4_2': (10.0, 17.0),            # 方法4：不截取，用全部
}
# 若全部不截取，可设 SEGMENTS = None 或 {}。
# ---------------------------------------------------------------------------
# SEGMENTS = None  # 或 {} 或 如上示例的 dict


def _segment_indices(n, timestamps, start, end):
    """根据 start/end 计算切片索引。start/end 为 int -> 索引；float -> 按时间；None -> 不限制。"""
    if start is None and end is None:
        return np.s_[:]
    t = np.asarray(timestamps)
    # 两端均为整数或 None 时按索引切片
    use_index = (start is None or isinstance(start, (int, np.integer))) and (
        end is None or isinstance(end, (int, np.integer)))
    if use_index:
        i0 = max(0, int(start)) if start is not None else 0
        i1 = min(n, int(end)) if end is not None else n
        return np.s_[i0:i1]
    # 按时间（浮点数表示秒）
    mask = np.ones(n, dtype=bool)
    if start is not None:
        mask &= (t >= float(start))
    if end is not None:
        mask &= (t <= float(end))
    return mask


def apply_segment(data, start, end):
    """
    截取数据段。data 为 load_folder_data 返回的 dict。
    start, end: 见 SEGMENTS 说明（索引或时间，None 表示不限制）。
    返回新 dict，不修改原 data。
    """
    if data is None or (start is None and end is None):
        return data
    n = len(data['timestamps'])
    seg = _segment_indices(n, data['timestamps'], start, end)
    if isinstance(seg, np.ndarray):
        # boolean mask
        return {
            'wrist': data['wrist'][seg],
            'timestamps': data['timestamps'][seg],
            'joint_angles': data['joint_angles'][seg],
            'scores': data['scores'][seg],
        }
    return {
        'wrist': data['wrist'][seg],
        'timestamps': data['timestamps'][seg],
        'joint_angles': data['joint_angles'][seg],
        'scores': data['scores'][seg],
    }


def load_folder_data(folder_path):
    """加载单个文件夹的 recorded_human_position, joint_angles, ergonomics_scores。"""
    recorded_path = os.path.join(folder_path, 'recorded_human_position.npy')
    joint_path = os.path.join(folder_path, 'optimized_joint_angles.npy')
    scores_path = os.path.join(folder_path, 'ergonomics_scores.npy')
    if not os.path.isfile(recorded_path) or not os.path.isfile(joint_path) or not os.path.isfile(scores_path):
        return None
    recorded = np.load(recorded_path, allow_pickle=True).item()
    joint_angles = np.load(joint_path)
    scores = np.load(scores_path)
    # recorded: wrist_positions, timestamps (and shoulder_positions, elbow_positions)
    wrist = np.array(recorded['wrist_positions'])
    ts = np.array(recorded['timestamps'])
    n = min(len(ts), len(wrist), len(joint_angles), len(scores))
    return {
        'wrist': wrist[:n],
        'timestamps': ts[:n],
        'joint_angles': np.asarray(joint_angles)[:n],
        'scores': np.asarray(scores).ravel()[:n],
    }


def plot_3d_wrist_trajectories(data_list, labels, save_path=None, task_goal_global=None, reference_trajectory=None):
    """一张 3D 图：四条 human wrist 轨迹对比，可选绘制 target goal 与 reference trajectory。"""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    # Reference trajectory（先画，在底层）
    if reference_trajectory is not None and len(reference_trajectory) > 0:
        ref = np.asarray(reference_trajectory)
        ax.plot(ref[:, 0], ref[:, 1], ref[:, 2], color='gold', linewidth=2.5, linestyle='--',
                alpha=0.85, label='Reference trajectory', zorder=0)
    # Target goal
    if task_goal_global is not None:
        g = np.asarray(task_goal_global).ravel()[:3]
        ax.scatter(g[0], g[1], g[2], c='red', s=280, marker='*', edgecolors='darkred', linewidths=2,
                   label='Target goal', zorder=10)
    # 四种方法的手腕轨迹
    for i, (data, label) in enumerate(zip(data_list, labels)):
        if data is None:
            continue
        w = data['wrist']
        ax.plot(w[:, 0], w[:, 1], w[:, 2], color=COLORS[i], linewidth=4, label=label, alpha=0.9)
        ax.scatter(w[0, 0], w[0, 1], w[0, 2], color=COLORS[i], s=50, marker=MARKERS[i])
        ax.scatter(w[-1, 0], w[-1, 1], w[-1, 2], color=COLORS[i], s=80, marker='*', edgecolors='k', linewidths=0.5)
    ax.set_xlabel('X (m)', fontsize=16)
    ax.set_ylabel('Y (m)', fontsize=16)
    ax.set_zlabel('Z (m)', fontsize=16)
    # ax.set_title('3D Human Wrist Trajectories Comparison', fontsize=14, fontweight='bold')
    # ax.legend(loc='upper left', fontsize=12)
    ax.tick_params(axis='x', labelsize=18)
    ax.tick_params(axis='y', labelsize=18)
    ax.tick_params(axis='z', labelsize=18)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.show()


def plot_joint_angles_comparison(data_list, labels, save_path=None):
    """四组关节角对比：按关节分 4 个子图，每个子图内四条曲线。"""
    joint_names = ['Shoulder Flexion', 'Shoulder Abduction', 'Elbow Flexion', 'Forearm Rotation']
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    axes = axes.flatten()
    for j in range(4):
        ax = axes[j]
        for i, (data, label) in enumerate(zip(data_list, labels)):
            if data is None or data['joint_angles'].ndim < 2:
                continue
            q = data['joint_angles']
            if q.shape[1] <= j:
                continue
            t = data['timestamps']
            n = min(len(t), len(q))
            deg = np.rad2deg(q[:n, j])
            ax.plot(t[:n], deg, color=COLORS[i], linewidth=3, label=label, alpha=0.9)
        ax.set_ylabel('Angle (deg)', fontsize=12)
        ax.set_title(joint_names[j], fontsize=12, fontweight='bold')
        # ax.legend(loc='upper right', fontsize=11)
        ax.tick_params(axis='both', labelsize=11)
        ax.grid(False)
    axes[2].set_xlabel('Time (s)', fontsize=12)
    axes[3].set_xlabel('Time (s)', fontsize=12)
    fig.suptitle('Joint Angles Comparison (Four Methods)', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.show()


def plot_ergonomics_scores_comparison(data_list, labels, save_path=None):
    """一张图：四条 ergonomics_scores 随时间变化。"""
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (data, label) in enumerate(zip(data_list, labels)):
        if data is None:
            continue
        t = data['timestamps']
        s = data['scores']
        n = min(len(t), len(s))
        ax.plot(t[:n], s[:n], color=COLORS[i], linewidth=3.5, label=label, alpha=0.9)
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Ergonomic Score', fontsize=12)
    ax.set_title('Ergonomics Scores Comparison (Four Methods)', fontsize=14, fontweight='bold')
    # ax.legend(loc='upper right', fontsize=12)
    ax.tick_params(axis='both', labelsize=11)
    ax.grid(False)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.show()


def run_comparison(base_dir=None, save_figures=True, segments=None):
    """
    base_dir: 数据根目录（其下为 1_2, 2_5 等子文件夹）。
    save_figures: 是否保存对比图。
    segments: 数据段截取配置，None 表示使用本文件顶部的 SEGMENTS。
              dict: 键为文件夹名，值为 (start, end)，见 SEGMENTS 说明。
    """
    base_dir = base_dir or _base_dir
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"Base directory not found: {base_dir}")

    seg_cfg = segments if segments is not None else SEGMENTS
    if seg_cfg is None:
        seg_cfg = {}

    data_list = []
    labels = []
    for folder_name, method_label in FOLDER_METHODS:
        folder_path = os.path.join(base_dir, folder_name)
        data = load_folder_data(folder_path)
        if data is None:
            print(f"Warning: skip {folder_name} (missing or invalid data).")
            data_list.append(None)
        else:
            if folder_name in seg_cfg and seg_cfg[folder_name] is not None:
                start, end = seg_cfg[folder_name]
                data = apply_segment(data, start, end)
                print(f"Loaded {folder_name}: {len(data['timestamps'])} samples (segment applied).")
            else:
                print(f"Loaded {folder_name}: {len(data['timestamps'])} samples.")
            data_list.append(data)
        labels.append(method_label)

    out_dir = os.path.join(base_dir, 'comparison_figures')
    if save_figures:
        os.makedirs(out_dir, exist_ok=True)

    save = lambda name: os.path.join(out_dir, name) if save_figures else None

    plot_3d_wrist_trajectories(data_list, labels, save_path=save('comparison_3d_wrist_trajectories.png'),
                               task_goal_global=task_goal_global, reference_trajectory=reference_trajectory)
    plot_joint_angles_comparison(data_list, labels, save_path=save('comparison_joint_angles.png'))
    plot_ergonomics_scores_comparison(data_list, labels, save_path=save('comparison_ergonomics_scores.png'))

    if save_figures:
        print(f"Figures saved to {out_dir}")


if __name__ == '__main__':
    base = _base_dir
    if len(sys.argv) > 1:
        base = os.path.abspath(sys.argv[1])
    run_comparison(base_dir=base, save_figures=True)
