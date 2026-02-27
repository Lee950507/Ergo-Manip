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

_base_dir = os.path.join(_script_dir, '0205', 'chenzui')

# 3D 图中可视化的 target goal 与 reference trajectory（与实验规划一致）
# 与 run_CF_motion_planning_bi.py 一致：target goal 与 reference trajectory（box_carrying）
task_goal_global = np.array([1.48, 0.3, 0.95])
reference_trajectory = cf_plan.generate_reference_trajectory(
    task_goal_global + np.array([-0.03, -0.6, 0.35]), task_goal_global,
    num_points=200, trajectory_type='box_carrying', transition_radius_ratio=0.05)
reference_trajectory = np.asarray(reference_trajectory)

# 四组数据文件夹及对应方法名（1=Straight, 2=TSEF, 3=HD-SDF, 4=CF）
FOLDER_METHODS = [
    ('5', 'Method 1 (Straight)'),
    ('6_2', 'Method 2 (TSEF)'),
    ('7', 'Method 3 (HD-SDF)'),
    ('8', 'Method 4 (CF)'),
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
      # '1': (0.0, 8.3),         # 方法1：取索引 0~299
      # '2': (0.0, 10),     # 方法2：取 2s~30s 之间的数据
      # '3': (0.0, 9),      # 方法3：从第 50 个点取到末尾
      # '4': (0.0, 10),            # 方法4：不截取，用全部
      '5': (0.0, 9),         # 方法1：取索引 0~299
      '6_2': (0.0, 10.0),     # 方法2：取 2s~30s 之间的数据
      '7': (0.0, 9.0),      # 方法3：从第 50 个点取到末尾
      '8': (0.0, 10),
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
        out = {
            'wrist': data['wrist'][seg],
            'timestamps': data['timestamps'][seg],
            'joint_angles': data['joint_angles'][seg],
            'scores': data['scores'][seg],
        }
        if 'shoulder' in data:
            out['shoulder'] = data['shoulder'][seg]
        if 'robot_left' in data:
            out['robot_left'] = data['robot_left'][seg]
        if 'robot_right' in data:
            out['robot_right'] = data['robot_right'][seg]
        return out
    out = {
        'wrist': data['wrist'][seg],
        'timestamps': data['timestamps'][seg],
        'joint_angles': data['joint_angles'][seg],
        'scores': data['scores'][seg],
    }
    if 'shoulder' in data:
        out['shoulder'] = data['shoulder'][seg]
    if 'robot_left' in data:
        out['robot_left'] = data['robot_left'][seg]
    if 'robot_right' in data:
        out['robot_right'] = data['robot_right'][seg]
    return out


def load_folder_data(folder_path):
    """加载单个文件夹的 recorded_human_position, joint_angles, ergonomics_scores, 以及双臂末端轨迹（与 run_CF_motion_planning_bi 保存格式一致）。"""
    recorded_path = os.path.join(folder_path, 'recorded_human_position.npy')
    joint_path = os.path.join(folder_path, 'optimized_joint_angles.npy')
    scores_path = os.path.join(folder_path, 'ergonomics_scores.npy')
    if not os.path.isfile(recorded_path) or not os.path.isfile(joint_path) or not os.path.isfile(scores_path):
        return None
    recorded = np.load(recorded_path, allow_pickle=True).item()
    joint_angles = np.load(joint_path)
    scores = np.load(scores_path)
    wrist = np.array(recorded['wrist_positions'])
    ts = np.array(recorded['timestamps'])
    n = min(len(ts), len(wrist), len(joint_angles), len(scores))
    left_path = os.path.join(folder_path, 'optimized_robot_positions_left.npy')
    right_path = os.path.join(folder_path, 'optimized_robot_positions_right.npy')
    robot_left = robot_right = None
    if os.path.isfile(left_path) and os.path.isfile(right_path):
        robot_left = np.asarray(np.load(left_path))
        robot_right = np.asarray(np.load(right_path))
        if robot_left.ndim == 2 and robot_right.ndim == 2 and robot_left.shape[1] >= 3 and robot_right.shape[1] >= 3:
            n = min(n, len(robot_left), len(robot_right))
    out = {
        'wrist': wrist[:n],
        'timestamps': ts[:n],
        'joint_angles': np.asarray(joint_angles)[:n],
        'scores': np.asarray(scores).ravel()[:n],
    }
    if robot_left is not None and robot_right is not None and n > 0:
        out['robot_left'] = robot_left[:n, :3]
        out['robot_right'] = robot_right[:n, :3]
    return out


def load_emg_in_time_range(folder_path, t_start, t_end):
    """
    加载文件夹中的 muscle_activation_smooth 与 emg_time，只保留时间在 [t_start, t_end] 内的样本。
    返回 (N, 5) 的数组，5 为肌肉通道数；若文件缺失或区间内无数据则返回 None。
    """
    emg_time_path = os.path.join(folder_path, 'emg_time.npy')
    muscle_path = os.path.join(folder_path, 'muscle_activation_smooth.npy')
    if not os.path.isfile(emg_time_path) or not os.path.isfile(muscle_path):
        return None
    emg_time = np.load(emg_time_path, allow_pickle=True)
    muscle = np.load(muscle_path, allow_pickle=True)
    emg_time = np.asarray(emg_time).ravel()
    # muscle 可能为 list of arrays (每通道一列) 或 (T, 5) / (5, T)
    if isinstance(muscle, np.ndarray):
        if muscle.ndim == 1:
            return None
        if muscle.shape[0] == 5 and muscle.shape[1] != 5:
            muscle = muscle.T  # (5, T) -> (T, 5)
    else:
        muscle = np.column_stack(muscle) if len(muscle) > 0 else None
        if muscle is None or muscle.size == 0:
            return None
        if muscle.shape[1] != 5 and muscle.shape[0] == 5:
            muscle = muscle.T
    n_emg = min(len(emg_time), len(muscle))
    emg_time = emg_time[:n_emg]
    muscle = np.asarray(muscle)[:n_emg]
    if muscle.shape[1] > 5:
        muscle = muscle[:, :5]
    elif muscle.shape[1] < 5:
        return None
    mask = (emg_time >= float(t_start)) & (emg_time <= float(t_end))
    if not np.any(mask):
        return None
    return muscle[mask]


# 五路肌肉名称（与 EMG 通道对应，便于 txt 中阅读）
MUSCLE_NAMES = ['Muscle 1 (Biceps)', 'Muscle 2 (Triceps)', 'Muscle 3', 'Muscle 4', 'Muscle 5']


def _resample_trajectory_by_path_length(traj, num_points=100):
    """按路径长度将轨迹重采样为 num_points 个点。traj: (N, 3)。"""
    traj = np.asarray(traj)
    if len(traj) < 2 or num_points < 2:
        return traj
    s = np.zeros(len(traj))
    s[1:] = np.cumsum(np.linalg.norm(np.diff(traj, axis=0), axis=1))
    if s[-1] < 1e-12:
        return np.linspace(traj[0], traj[-1], num_points)
    s = s / s[-1]
    s_new = np.linspace(0, 1, num_points)
    return np.column_stack([
        np.interp(s_new, s, traj[:, 0]),
        np.interp(s_new, s, traj[:, 1]),
        np.interp(s_new, s, traj[:, 2]),
    ])


def _trajectory_reference_correlation(wrist_traj, ref_traj, n_sample=100):
    """
    生成轨迹与参考轨迹的相关系数（按路径长度对齐后，对 X/Y/Z 分别求 Pearson 相关，再取平均）。
    返回 (r_mean, r_x, r_y, r_z)，若不足两点则返回 (np.nan, np.nan, np.nan, np.nan)。
    """
    wrist = _resample_trajectory_by_path_length(wrist_traj, n_sample)
    ref = _resample_trajectory_by_path_length(ref_traj, n_sample)
    if len(wrist) < 2 or len(ref) < 2:
        return (np.nan, np.nan, np.nan, np.nan)
    r_x = np.corrcoef(wrist[:, 0], ref[:, 0])[0, 1] if np.std(wrist[:, 0]) > 1e-12 and np.std(ref[:, 0]) > 1e-12 else np.nan
    r_y = np.corrcoef(wrist[:, 1], ref[:, 1])[0, 1] if np.std(wrist[:, 1]) > 1e-12 and np.std(ref[:, 1]) > 1e-12 else np.nan
    r_z = np.corrcoef(wrist[:, 2], ref[:, 2])[0, 1] if np.std(wrist[:, 2]) > 1e-12 and np.std(ref[:, 2]) > 1e-12 else np.nan
    valid = [r for r in [r_x, r_y, r_z] if not np.isnan(r)]
    r_mean = np.mean(valid) if valid else np.nan
    return (r_mean, r_x, r_y, r_z)


def save_comparison_stats_txt(base_dir, data_list, labels, folder_names, seg_cfg, txt_path=None, reference_trajectory=None):
    """
    将四种方法的 Ergonomic scores（mean ± std）、五种肌肉的 Muscle activation smooth（mean ± std）、
    以及生成轨迹与参考轨迹的相关性，写入 txt。
    """
    if txt_path is None:
        txt_path = os.path.join(base_dir, 'comparison_figures', 'comparison_stats.txt')
    os.makedirs(os.path.dirname(txt_path), exist_ok=True)
    lines = []
    lines.append('=' * 60)
    lines.append('Comparison statistics: mean ± std')
    lines.append('=' * 60)

    # 1. Ergonomic scores（四种方法）
    lines.append('')
    lines.append('--- Ergonomic scores ---')
    for i, (data, label, folder_name) in enumerate(zip(data_list, labels, folder_names)):
        if data is None:
            lines.append(f'{label}: N/A (no data)')
            continue
        s = np.asarray(data['scores']).ravel()
        if len(s) == 0:
            lines.append(f'{label}: N/A')
            continue
        m, std = np.mean(s), np.std(s)
        lines.append(f'{label}: {m:.4f} ± {std:.4f}')
    lines.append('')

    # 2. Muscle activation (smooth)，在时间区间内对五路肌肉分别算 mean ± std
    lines.append('--- Muscle activation (smooth), within segment time range ---')
    for i, (data, label, folder_name) in enumerate(zip(data_list, labels, folder_names)):
        lines.append(f'\n{label}:')
        if data is None or len(data['timestamps']) == 0:
            lines.append('  (no motion data, skip EMG)')
            continue
        t_min = float(np.min(data['timestamps']))
        t_max = float(np.max(data['timestamps']))
        folder_path = os.path.join(base_dir, folder_name)
        emg_segment = load_emg_in_time_range(folder_path, t_min, t_max)
        if emg_segment is None:
            lines.append('  (no EMG data in time range)')
            continue
        for ch in range(min(5, emg_segment.shape[1])):
            mu = np.mean(emg_segment[:, ch])
            std = np.std(emg_segment[:, ch])
            name = MUSCLE_NAMES[ch] if ch < len(MUSCLE_NAMES) else f'Muscle {ch + 1}'
            lines.append(f'  {name}: {mu:.4f} ± {std:.4f}')
    lines.append('')

    # 3. 生成轨迹与参考轨迹的相关性（按路径长度对齐后 X/Y/Z 的 Pearson 相关）
    lines.append('--- Trajectory vs Reference correlation (Pearson, after path-length alignment) ---')
    ref_traj = np.asarray(reference_trajectory) if reference_trajectory is not None and len(reference_trajectory) > 0 else None
    for i, (data, label, folder_name) in enumerate(zip(data_list, labels, folder_names)):
        if data is None or len(data.get('wrist', [])) < 2:
            lines.append(f'{label}: N/A (insufficient trajectory)')
            continue
        if ref_traj is None or len(ref_traj) < 2:
            lines.append(f'{label}: N/A (no reference trajectory)')
            continue
        r_mean, r_x, r_y, r_z = _trajectory_reference_correlation(data['wrist'], ref_traj, n_sample=100)
        if np.isnan(r_mean):
            lines.append(f'{label}: N/A')
        else:
            lines.append(f'{label}: mean r = {r_mean:.4f}  (r_x = {r_x:.4f}, r_y = {r_y:.4f}, r_z = {r_z:.4f})')
    lines.append('')
    lines.append('=' * 60)

    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"Stats saved to {txt_path}")


def plot_3d_wrist_trajectories(data_list, labels, save_path=None, task_goal_global=None, reference_trajectory=None):
    """一张 3D 图：四种方法的人体 wrist 轨迹对比，可选绘制 target goal 与 reference trajectory。"""
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
    # 四种方法：wrist 轨迹
    for i, (data, label) in enumerate(zip(data_list, labels)):
        if data is None:
            continue
        w = data['wrist']
        ax.plot(w[:, 0], w[:, 1], w[:, 2], color=COLORS[i], linewidth=4, label=label, alpha=0.9)
        ax.scatter(w[0, 0], w[0, 1], w[0, 2], color=COLORS[i], s=50, marker=MARKERS[i])
        ax.scatter(w[-1, 0], w[-1, 1], w[-1, 2], color=COLORS[i], s=80, marker='*', edgecolors='k', linewidths=0.5)
    ax.set_xlabel('X (m)', fontsize=14)
    ax.set_ylabel('Y (m)', fontsize=14)
    ax.set_zlabel('Z (m)', fontsize=14)
    # ax.legend(loc='upper left', fontsize=9)
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.tick_params(axis='z', labelsize=14)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=400, bbox_inches='tight')
    plt.show()


def plot_joint_angles_comparison(data_list, labels, save_path=None):
    """四组关节角对比：按关节分 4 个子图，每个子图内四条曲线。"""
    joint_names = ['Shoulder Abduction/Adduction', 'Shoulder Flexion/Extension', 'Shoulder Internal/External Rotation', 'Elbow Flexion/Extension']
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
            t = np.asarray(data['timestamps'])
            n = min(len(t), len(q))
            t, q = t[:n], q[:n]
            t_min, t_max = t.min(), t.max()
            t_norm = (t - t_min) / (t_max - t_min) if (t_max - t_min) > 1e-9 else np.linspace(0, 1, n)
            rad = q[:, j]
            ax.plot(t_norm, rad, color=COLORS[i], linewidth=3, label=label, alpha=0.9)
        ax.set_ylabel('Angle (rad)', fontsize=14)
        ax.set_title(joint_names[j], fontsize=14, fontweight='bold')
        # ax.legend(loc='upper right', fontsize=11)
        ax.tick_params(axis='both', labelsize=14)
        ax.grid(False)
    axes[2].set_xlabel('Normalized time', fontsize=14)
    axes[3].set_xlabel('Normalized time', fontsize=14)
    fig.suptitle('Joint Angles Comparison (Four Methods)', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=400, bbox_inches='tight')
    plt.show()


def plot_ergonomics_scores_comparison(data_list, labels, save_path=None):
    """一张图：四条 ergonomics_scores 随时间变化（时间轴统一为 0-1）。"""
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (data, label) in enumerate(zip(data_list, labels)):
        if data is None:
            continue
        t = np.asarray(data['timestamps'])
        s = np.asarray(data['scores'])
        n = min(len(t), len(s))
        t, s = t[:n], s[:n]
        t_min, t_max = t.min(), t.max()
        t_norm = (t - t_min) / (t_max - t_min) if (t_max - t_min) > 1e-9 else np.linspace(0, 1, n)
        ax.plot(t_norm, s, color=COLORS[i], linewidth=3.5, label=label, alpha=0.9)
    ax.set_xlabel('Normalized time', fontsize=14)
    ax.set_ylabel('Ergonomic Score', fontsize=14)
    ax.set_title('Ergonomics Scores Comparison (Four Methods)', fontsize=14, fontweight='bold')
    # ax.legend(loc='upper right', fontsize=12)
    ax.tick_params(axis='both', labelsize=14)
    ax.grid(False)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=400, bbox_inches='tight')
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

    out_dir = os.path.join(base_dir, 'comparison_figures_box2')
    if save_figures:
        os.makedirs(out_dir, exist_ok=True)

    save = lambda name: os.path.join(out_dir, name) if save_figures else None

    plot_3d_wrist_trajectories(data_list, labels, save_path=save('comparison_3d_wrist_trajectories.png'),
                               task_goal_global=task_goal_global, reference_trajectory=reference_trajectory)
    plot_joint_angles_comparison(data_list, labels, save_path=save('comparison_joint_angles.png'))
    plot_ergonomics_scores_comparison(data_list, labels, save_path=save('comparison_ergonomics_scores.png'))

    # 保存 ergonomic scores、muscle activation (smooth)、轨迹-参考相关性 到 txt
    folder_names = [f[0] for f in FOLDER_METHODS]
    save_comparison_stats_txt(base_dir, data_list, labels, folder_names, seg_cfg,
                              txt_path=save('comparison_stats.txt'), reference_trajectory=reference_trajectory)

    if save_figures:
        print(f"Figures saved to {out_dir}")


if __name__ == '__main__':
    base = _base_dir
    if len(sys.argv) > 1:
        base = os.path.abspath(sys.argv[1])
    run_comparison(base_dir=base, save_figures=True)
