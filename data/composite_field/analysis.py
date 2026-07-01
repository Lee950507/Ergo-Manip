#!/usr/bin/env python3
"""
Visualize composite field experiment results from composite_field/0204/test/1.
Data format: recorded_human_position.npy, optimized_robot_positions.npy,
             optimized_joint_angles.npy, ergonomics_scores.npy
"""
import os
import math
import sys
import numpy as np
import matplotlib.pyplot as plt

# Project root for importing iros2025_code and motion_planning modules
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
from scipy.spatial import KDTree
from iros2025_code import main_opt_static as mos
from IJSR.motion_planning_composite_filed_moving_base_ros import (
    trans_shoulder2global,
    generate_reference_trajectory,
    compute_ergonomic_vector_task_space,
)

# Joint limits (same as run_CF_motion_planning)
JOINT_ANGLE_BOUNDS = [
    (-math.pi / 18, 17 * math.pi / 18),
    (-math.pi / 18, 17 * math.pi / 18),
    (-math.pi / 3, math.pi / 2),
    (-math.pi / 2, math.pi / 3),
]


def load_data(folder):
    """Load .npy files from folder. folder is path to e.g. composite_field/0204/test/1."""
    recorded = np.load(os.path.join(folder, 'recorded_human_position.npy'), allow_pickle=True).item()
    robot_pos = np.load(os.path.join(folder, 'optimized_robot_positions.npy'))
    joint_angles = np.load(os.path.join(folder, 'optimized_joint_angles.npy'))
    scores = np.load(os.path.join(folder, 'ergonomics_scores.npy'))
    return recorded, robot_pos, joint_angles, scores


def load_planned_motion_directions(folder):
    """Load locally recorded planned_motion_directions.npy if present. Returns (N, 3) or None."""
    path = os.path.join(folder, 'planned_motion_directions.npy')
    if not os.path.isfile(path):
        return None
    return np.load(path)


def get_task_goal_global(recorded):
    """Target goal: initial_position + [0.1, 0.0, -0.3] (same as run_CF_motion_planning)."""
    wr = np.array(recorded['wrist_positions'])
    if len(wr) == 0:
        return None
    initial_position = np.array(wr[0])
    task_goal_global = np.array([1.4, 0.3, 1.1])
    return task_goal_global


def get_optimal_position_global(recorded):
    """Ergonomic optimal wrist position (same as run_CF): FK of optimal_q in global frame."""
    sh = recorded['shoulder_positions']
    el = recorded['elbow_positions']
    wr = recorded['wrist_positions']
    if not (sh and el and wr):
        return None
    sh0 = np.array(sh[0])
    el0 = np.array(el[0])
    wr0 = np.array(wr[0])
    d_ual, d_uar, d_lal, d_lar = mos.calculate_arm_dimensions(sh0, el0, wr0, sh0, el0, wr0)
    optimal_q = np.array([0, 0, 0, -math.pi / 4])
    _, hand_shoulder = mos.forward_kinematics(optimal_q, d_uar, d_lar)
    optimal_global = trans_shoulder2global(hand_shoulder.copy(), sh0, arm='right')
    return optimal_global


def compute_motion_directions_per_step(recorded, joint_angles, task_goal_global,
                                        robot_pos=None, planning_method=3, step_interval=1,
                                        planned_recorded=None):
    """
    Retrospectively compute planned motion direction (goal/ref/ergo and combined) at each step
    and compare with actual displacement direction. actual_direction from optimized_robot_positions.
    step_interval: compute only every N steps (1 = every step).
    planning_method: 0=Straight, 1=TSEF, 2=HD-SDF, 3=CF (same as run_CF for ref and weights).
    planned_recorded: optional (N, 3) array from planned_motion_directions.npy for comparison.
    """
    sh = np.array(recorded['shoulder_positions'])
    wr = np.array(recorded['wrist_positions'])
    n = len(wr)
    if n < 2 or len(joint_angles) < n:
        return None
    robot_pos = np.array(robot_pos) if robot_pos is not None and len(robot_pos) >= n else None

    step_interval = max(1, int(step_interval))
    indices = list(range(0, n - 1, step_interval))
    if not indices:
        return None

    sh0, el0, wr0 = sh[0], np.array(recorded['elbow_positions'][0]), wr[0]
    d_ual, d_uar, d_lal, d_lar = mos.calculate_arm_dimensions(sh0, el0, wr0, sh0, el0, wr0)

    if planning_method in (2, 3):
        ref_traj = generate_reference_trajectory(
            task_goal_global + np.array([0.0, -0.6, 0.35]), task_goal_global, num_points=200, trajectory_type='box_carrying', transition_radius_ratio=0.15)
        ref_kdtree = KDTree(ref_traj)
    else:
        ref_traj = None
        ref_kdtree = None

    if planning_method == 0:
        w_goal, w_ref, w_ergo = 1.0, 0.0, 0.0
    elif planning_method == 1:
        w_goal, w_ref, w_ergo = 0.55, 0.0, 0.45
    elif planning_method == 2:
        w_goal, w_ref, w_ergo = 0.45, 0.55, 0.0
    else:
        w_goal, w_ref, w_ergo = 0.45, 0.25, 0.30

    results = []
    for i in indices:
        endpoint = np.array(wr[i])
        shoulder = np.array(sh[i])
        current_q = np.array(joint_angles[i])
        goal_dist = np.linalg.norm(task_goal_global - endpoint)

        goal_vec = task_goal_global - endpoint
        if np.linalg.norm(goal_vec) > 1e-8:
            goal_vec = goal_vec / np.linalg.norm(goal_vec)
        else:
            goal_vec = np.zeros(3)

        ref_vec = np.zeros(3)
        if ref_kdtree is not None and w_ref > 0:
            _, idx = ref_kdtree.query(endpoint, k=1)
            idx = np.atleast_1d(idx)[0]
            closest = ref_traj[idx]
            ref_vec = closest - endpoint
            if np.linalg.norm(ref_vec) > 1e-8:
                ref_vec = ref_vec / np.linalg.norm(ref_vec)

        ergo_vec = np.zeros(3)
        if w_ergo > 0:
            ergo_vec = compute_ergonomic_vector_task_space(
                endpoint, shoulder, current_q, d_uar, d_lar, JOINT_ANGLE_BOUNDS,
                neighbor_radius=0.02, n_samples=27)

        combined = w_goal * goal_vec + w_ref * ref_vec + w_ergo * ergo_vec
        combined_norm = np.linalg.norm(combined)
        if combined_norm > 1e-8:
            combined = combined / combined_norm
        else:
            combined = goal_vec.copy()

        # Actual direction: from optimized robot positions (robot waypoints) at sampled steps
        if robot_pos is not None and i + 1 < len(robot_pos):
            actual_disp = robot_pos[i + 1] - robot_pos[i]
            actual_norm = np.linalg.norm(actual_disp)
            if actual_norm > 1e-8:
                actual_direction = actual_disp / actual_norm
            else:
                actual_direction = np.zeros(3)
        else:
            actual_direction = np.zeros(3)
            actual_norm = 0.0

        dot_planned_goal = np.dot(combined, goal_vec)
        dot_actual_goal = np.dot(actual_direction, goal_vec) if actual_norm > 1e-8 else 0.0
        dot_planned_actual = np.dot(combined, actual_direction) if actual_norm > 1e-8 else 0.0

        # Optional: compare with locally recorded planned direction (planned_motion_directions.npy)
        recorded_planned = None
        dot_recorded_goal = None
        dot_recorded_actual = None
        if planned_recorded is not None and i < len(planned_recorded):
            rec = np.asarray(planned_recorded[i], dtype=float)
            rn = np.linalg.norm(rec)
            if rn > 1e-8:
                recorded_planned = rec / rn
                dot_recorded_goal = np.dot(recorded_planned, goal_vec)
                dot_recorded_actual = np.dot(recorded_planned, actual_direction) if actual_norm > 1e-8 else 0.0

        entry = {
            'step': i,
            'goal_dist': goal_dist,
            'goal_vec': goal_vec,
            'ref_vec': ref_vec,
            'ergo_vec': ergo_vec,
            'combined': combined,
            'actual_direction': actual_direction,
            'actual_origin': robot_pos[i] if robot_pos is not None and i < len(robot_pos) else wr[i],
            'dot_planned_goal': dot_planned_goal,
            'dot_actual_goal': dot_actual_goal,
            'dot_planned_actual': dot_planned_actual,
        }
        if recorded_planned is not None:
            entry['recorded_planned'] = recorded_planned
            entry['dot_recorded_goal'] = dot_recorded_goal
            entry['dot_recorded_actual'] = dot_recorded_actual
        results.append(entry)
    return results


def plot_motion_direction_diagnostics(recorded, joint_angles, task_goal_global,
                                      robot_pos=None, planning_method=3, step_interval=1,
                                      planned_recorded=None, save_path=None):
    """Plot diagnostics of planned vs actual motion direction (sampled every step_interval steps).
    actual = from optimized_robot_positions. planned_recorded = from planned_motion_directions.npy."""
    directions = compute_motion_directions_per_step(
        recorded, joint_angles, task_goal_global, robot_pos=robot_pos,
        planning_method=planning_method, step_interval=step_interval,
        planned_recorded=planned_recorded)
    if not directions:
        print("Cannot compute motion directions (need at least 2 samples).")
        return

    steps = [d['step'] for d in directions]
    dot_planned_goal = np.array([d['dot_planned_goal'] for d in directions])
    dot_actual_goal = np.array([d['dot_actual_goal'] for d in directions])
    dot_planned_actual = np.array([d['dot_planned_actual'] for d in directions])
    has_recorded = any('dot_recorded_goal' in d for d in directions)
    if has_recorded:
        dot_recorded_goal = np.array([d.get('dot_recorded_goal', np.nan) for d in directions])
        dot_recorded_actual = np.array([d.get('dot_recorded_actual', np.nan) for d in directions])

    wr = np.array(recorded['wrist_positions'])
    n = len(wr)
    steps_all = np.arange(n)
    goal_dist_all = np.array([np.linalg.norm(task_goal_global - np.array(wr[i])) for i in range(n)]) * 1000  # mm

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    ax = axes[0]
    ax.plot(steps, dot_planned_goal, 'b.-', label='Recomputed planned · goal', linewidth=1.5, markersize=6)
    ax.plot(steps, dot_actual_goal, 'r.-', label='Actual dir · goal', linewidth=1.5, alpha=0.8, markersize=6)
    if has_recorded:
        ax.plot(steps, dot_recorded_goal, 'c.-', label='Recorded planned · goal', linewidth=1.5, markersize=6)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=1, color='green', linestyle=':', alpha=0.5, label='Perfect toward goal')
    ax.set_ylabel('Alignment with goal')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_title('Direction vs goal: >0 toward goal, <0 away, =1 aligned')
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(steps, dot_planned_actual, 'g.-', label='Recomputed planned · actual', linewidth=1.5, markersize=6)
    if has_recorded:
        ax.plot(steps, dot_recorded_actual, 'm.-', label='Recorded planned · actual', linewidth=1.5, markersize=6)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=1, color='green', linestyle=':', alpha=0.5)
    ax.set_ylabel('Planned · actual')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_title('Planned vs actual direction alignment')
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(steps_all, goal_dist_all, 'k-', label='Distance to goal (mm)', linewidth=1.5)
    ax.set_xlabel('Step')
    ax.set_ylabel('Goal distance (mm)')
    ax.set_title('Endpoint distance to goal (should decrease if moving toward goal)')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle('Motion direction diagnostics (planning_method={}, step_interval={})'.format(
        planning_method, step_interval), fontsize=12)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_3d_trajectories(recorded, robot_pos, task_goal_global=None, optimal_position_global=None,
                         motion_directions=None, arrow_scale=0.03, save_path=None):
    """3D: human shoulder/elbow/wrist, robot trajectory, target goal, ergonomic optimal,
    and motion direction arrows (actual + planned) at sampled steps if motion_directions provided."""
    sh = np.array(recorded['shoulder_positions'])
    el = np.array(recorded['elbow_positions'])
    wr = np.array(recorded['wrist_positions'])
    t = np.array(recorded['timestamps'])

    n = len(t)
    if n == 0:
        return

    if task_goal_global is None:
        task_goal_global = get_task_goal_global(recorded)
    if optimal_position_global is None:
        optimal_position_global = get_optimal_position_global(recorded)

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(wr[:, 0], wr[:, 1], wr[:, 2], 'b-', linewidth=2, label='Human wrist', alpha=0.8)
    ax.scatter(wr[0, 0], wr[0, 1], wr[0, 2], c='green', s=80, marker='o', label='Start')
    ax.scatter(wr[-1, 0], wr[-1, 1], wr[-1, 2], c='blue', s=80, marker='s', label='End (human)')

    if task_goal_global is not None:
        ax.scatter(task_goal_global[0], task_goal_global[1], task_goal_global[2],
                   c='magenta', s=200, marker='*', label='Target goal', edgecolors='darkviolet', linewidths=1.5)
    if optimal_position_global is not None:
        ax.scatter(optimal_position_global[0], optimal_position_global[1], optimal_position_global[2],
                   c='orange', s=150, marker='^', label='Ergonomic optimal', edgecolors='darkorange', linewidths=1.5)

    if len(robot_pos) > 0:
        ax.plot(robot_pos[:, 0], robot_pos[:, 1], robot_pos[:, 2], 'r-', linewidth=1.5,
               label='Optimized robot (waypoints)', alpha=0.8)
        ax.scatter(robot_pos[-1, 0], robot_pos[-1, 1], robot_pos[-1, 2], c='red', s=80, marker='^',
                  label='End (robot)')

    # Motion direction arrows at sampled steps (recomputed planned, recorded planned, actual)
    if motion_directions:
        planned_plotted = False
        recorded_plotted = False
        actual_plotted = False
        for d in motion_directions:
            i = d['step']
            if i >= len(wr):
                continue
            planned_origin = np.array(wr[i])
            combined = d['combined']
            if np.linalg.norm(combined) > 1e-8:
                u, v, w = arrow_scale * combined
                ax.quiver(planned_origin[0], planned_origin[1], planned_origin[2], u, v, w,
                          color='cyan', arrow_length_ratio=0.3, linewidth=1.5,
                          label='Recomputed planned' if not planned_plotted else None)
                planned_plotted = True
            # Recorded planned (from planned_motion_directions.npy)
            rec = d.get('recorded_planned')
            if rec is not None and np.linalg.norm(rec) > 1e-8:
                u, v, w = arrow_scale * rec
                ax.quiver(planned_origin[0], planned_origin[1], planned_origin[2], u, v, w,
                          color='orange', arrow_length_ratio=0.3, linewidth=1.5, alpha=0.9,
                          label='Recorded planned' if not recorded_plotted else None)
                recorded_plotted = True
            # Actual direction: at robot trajectory (optimized_robot_positions)
            actual_origin = d.get('actual_origin', planned_origin)
            actual = d['actual_direction']
            if np.linalg.norm(actual) > 1e-8:
                u, v, w = arrow_scale * actual
                ax.quiver(actual_origin[0], actual_origin[1], actual_origin[2], u, v, w,
                          color='red', arrow_length_ratio=0.3, linewidth=1.5, alpha=0.85,
                          label='Actual dir (robot)' if not actual_plotted else None)
                actual_plotted = True

    # Arm skeleton at last frame
    ax.plot([sh[-1, 0], el[-1, 0]], [sh[-1, 1], el[-1, 1]], [sh[-1, 2], el[-1, 2]], 'k-', linewidth=2)
    ax.plot([el[-1, 0], wr[-1, 0]], [el[-1, 1], wr[-1, 1]], [el[-1, 2], wr[-1, 2]], 'k-', linewidth=2)
    ax.scatter(sh[-1, 0], sh[-1, 1], sh[-1, 2], c='black', s=60, label='Shoulder')

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('3D Trajectories: Human vs Optimized Robot')
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_position_components(recorded, robot_pos, task_goal_global=None, optimal_position_global=None, save_path=None):
    """Time vs X, Y, Z for human wrist, robot waypoints, target goal, and ergonomic optimal."""
    wr = np.array(recorded['wrist_positions'])
    t = np.array(recorded['timestamps'])
    if len(t) == 0:
        return

    if task_goal_global is None:
        task_goal_global = get_task_goal_global(recorded)
    if optimal_position_global is None:
        optimal_position_global = get_optimal_position_global(recorded)

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(t, wr[:, 0], 'b-', label='Human wrist X')
    axes[1].plot(t, wr[:, 1], 'b-', label='Human wrist Y')
    axes[2].plot(t, wr[:, 2], 'b-', label='Human wrist Z')

    if task_goal_global is not None:
        axes[0].axhline(y=task_goal_global[0], color='magenta', linestyle='--', linewidth=1.5, alpha=0.8, label='Goal X')
        axes[1].axhline(y=task_goal_global[1], color='magenta', linestyle='--', linewidth=1.5, alpha=0.8, label='Goal Y')
        axes[2].axhline(y=task_goal_global[2], color='magenta', linestyle='--', linewidth=1.5, alpha=0.8, label='Goal Z')
    if optimal_position_global is not None:
        axes[0].axhline(y=optimal_position_global[0], color='orange', linestyle=':', linewidth=1.5, alpha=0.8, label='Optimal X')
        axes[1].axhline(y=optimal_position_global[1], color='orange', linestyle=':', linewidth=1.5, alpha=0.8, label='Optimal Y')
        axes[2].axhline(y=optimal_position_global[2], color='orange', linestyle=':', linewidth=1.5, alpha=0.8, label='Optimal Z')

    if len(robot_pos) > 0:
        n_robot = len(robot_pos)
        t_robot = np.linspace(t[0], t[-1], n_robot) if len(t) > 1 else t[0:1]
        if n_robot == 1:
            t_robot = t[0:1]
        axes[0].plot(t_robot, robot_pos[:, 0], 'r.-', markersize=4, label='Robot X')
        axes[1].plot(t_robot, robot_pos[:, 1], 'r.-', markersize=4, label='Robot Y')
        axes[2].plot(t_robot, robot_pos[:, 2], 'r.-', markersize=4, label='Robot Z')

    for ax, dim in zip(axes, ['X', 'Y', 'Z']):
        ax.set_ylabel(f'{dim} (m)')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel('Time (s)')
    fig.suptitle('Position Components vs Time')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_joint_angles(joint_angles, recorded_timestamps, save_path=None):
    """Joint angles (deg) vs time."""
    q = np.array(joint_angles)
    t = np.array(recorded_timestamps)
    if q.size == 0:
        return
    if q.ndim == 1:
        q = q.reshape(1, -1)
    n_steps = min(len(t), len(q))
    t = t[:n_steps]
    q = np.rad2deg(q[:n_steps])
    labels = ['Shoulder Flexion', 'Shoulder Abduction', 'Elbow Flexion', 'Forearm Rotation']
    fig, ax = plt.subplots(figsize=(10, 5))
    for i in range(q.shape[1]):
        ax.plot(t, q[:, i], label=labels[i] if i < len(labels) else f'Joint {i+1}')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Angle (deg)')
    ax.set_title('Joint Angles vs Time')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_ergonomics_scores(scores, recorded_timestamps, save_path=None):
    """Ergonomic score vs time."""
    s = np.array(scores).ravel()
    t = np.array(recorded_timestamps)
    n = min(len(t), len(s))
    t, s = t[:n], s[:n]
    if n == 0:
        return
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, s, 'b-', linewidth=1.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Ergonomic Score')
    ax.set_title('Ergonomic Score vs Time')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def run_analysis(folder, save_figures=True, planning_method=3, motion_direction_step_interval=None):
    folder = os.path.abspath(folder)
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Folder not found: {folder}")

    recorded, robot_pos, joint_angles, scores = load_data(folder)
    out_dir = os.path.join(folder, 'figures') if save_figures else None
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    print(f"Loaded: {len(recorded['timestamps'])} samples, "
          f"robot waypoints {len(robot_pos)}, joints {joint_angles.shape}, scores {scores.shape}")

    save = lambda name: os.path.join(out_dir, name) if out_dir else None
    task_goal_global = get_task_goal_global(recorded)
    optimal_position_global = get_optimal_position_global(recorded)
    if task_goal_global is not None:
        print(f"Target goal (initial + [0.1, 0, -0.3]): {task_goal_global}")
    if optimal_position_global is not None:
        print(f"Ergonomic optimal (optimal_q FK in global): {optimal_position_global}")

    step_interval = motion_direction_step_interval
    if step_interval is None:
        step_interval = max(1, len(recorded['timestamps']) // 50)
    planned_recorded = load_planned_motion_directions(folder)
    motion_directions = compute_motion_directions_per_step(
        recorded, joint_angles, task_goal_global, robot_pos=robot_pos,
        planning_method=planning_method, step_interval=step_interval,
        planned_recorded=planned_recorded)

    plot_3d_trajectories(recorded, robot_pos, task_goal_global=task_goal_global,
                         optimal_position_global=optimal_position_global,
                         motion_directions=motion_directions, arrow_scale=0.03,
                         save_path=save('trajectories_3d.png'))
    plot_position_components(recorded, robot_pos, task_goal_global=task_goal_global,
                             optimal_position_global=optimal_position_global, save_path=save('position_components.png'))
    plot_joint_angles(joint_angles, recorded['timestamps'], save_path=save('joint_angles.png'))
    plot_ergonomics_scores(scores, recorded['timestamps'], save_path=save('ergonomics_scores.png'))

    # Retrospective motion direction diagnostics (sampled steps; includes recorded planned if npy present)
    plot_motion_direction_diagnostics(
        recorded, joint_angles, task_goal_global, robot_pos=robot_pos,
        planning_method=planning_method,
        step_interval=step_interval,
        planned_recorded=planned_recorded,
        save_path=save('motion_direction_diagnostics.png'))


if __name__ == '__main__':
    import sys
    base = os.path.dirname(os.path.abspath(__file__))
    folder = os.path.join(base, '0316', 'chenzui', '3_mid')
    planning_method = 2
    step_interval = None  # default: auto (~50 points)
    if len(sys.argv) > 1:
        folder = sys.argv[1]
    if len(sys.argv) > 2:
        planning_method = int(sys.argv[2])  # 0=Straight, 1=TSEF, 2=HD-SDF, 3=CF
    if len(sys.argv) > 3:
        step_interval = int(sys.argv[3])  # motion direction sample every N steps
    run_analysis(folder, save_figures=True, planning_method=planning_method,
                 motion_direction_step_interval=step_interval)

# shoulder: [ 1.56601444 -0.40607426  1.14786445]
# elbow: [ 1.36429105 -0.31160378  1.18307883]
# wrist: [ 1.05226499 -0.29558505  1.0983042 ]

# shoulder: [ 1.50265166 -0.37376922  1.06467347]
# elbow: [ 1.27488495 -0.2695518   1.1101194 ]
# wrist: [ 1.00134111 -0.24058926  1.06534663]