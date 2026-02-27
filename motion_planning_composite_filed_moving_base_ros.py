#!/usr/bin/env python3
import numpy as np
import math
import matplotlib.pyplot as plt
import utils
import transformation as tsf
from iros2025_code import main_opt_static as mos

from scipy.optimize import minimize, differential_evolution
from scipy.interpolate import CubicSpline
from scipy.spatial import KDTree
from scipy.ndimage import gaussian_filter1d

import sys
import subprocess


def plot_right_upper_limb_skeleton(ax, global_positions, color='gray', linewidth=3):
    """
    Plot only the right upper limb skeleton (shoulder -> elbow -> wrist).
    global_positions[3]=shoulder, [4]=elbow, [5]=wrist.
    """
    shoulder = global_positions[3]
    elbow = global_positions[4]
    wrist = global_positions[5]
    ax.plot([shoulder[0], elbow[0]], [shoulder[1], elbow[1]], [shoulder[2], elbow[2]],
            color=color, linewidth=linewidth, alpha=0.9)
    ax.plot([elbow[0], wrist[0]], [elbow[1], wrist[1]], [elbow[2], wrist[2]],
            color=color, linewidth=linewidth, alpha=0.9)


# Get ROS workspace path
workspace_path = '/home/clover/catkin_ws'

last_relative_pose_wrists = None
last_object_pose = None
global ind
ind = 1


def launch_roslaunch():
    launch_file = "~/catkin_ws/src/curi_whole_body_interface/launch/python_curi_dual_arm_ic_qbhand.launch"
    command = f"roslaunch {launch_file}"
    return subprocess.Popen(command, shell=True)


def vrpn_launch_roslaunch():
    launch_file = "~/catkin_ws/src/vrpn_client_ros/launch/sample.launch"
    command = f"roslaunch {launch_file} server:=192.168.10.7"
    return subprocess.Popen(command, shell=True)


def signal_handler(sig, frame):
    print('Python shutdown signal received...')
    if 'roslaunch_process' in locals():
        print('Shutdown roslaunch process.')
        roslaunch_process.terminate()
        roslaunch_process.wait()
    print('Python shutdown.')
    sys.exit(0)


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
    if arm == 'left':
        joint_pos[[0, 1]] = -joint_pos[[1, 0]]
        joint_pos[1] = -joint_pos[1]
        joint_pos = joint_pos + shoulder_pos
    if arm == 'right':
        joint_pos[[0, 1]] = -joint_pos[[1, 0]]
        joint_pos = joint_pos + shoulder_pos
    return joint_pos


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


def smooth_trajectory(waypoints, smoothing_factor=0.8, iterations=3):
    """
    Smooth trajectory waypoints
    waypoints: original trajectory points [N, 3]
    smoothing_factor: smoothing factor (0-1), larger value means stronger smoothing
    iterations: number of smoothing iterations
    """
    if len(waypoints) <= 2:
        return waypoints

    smoothed = np.array(waypoints, copy=True)

    for _ in range(iterations):
        original_start = smoothed[0].copy()
        original_end = smoothed[-1].copy()

        for i in range(1, len(smoothed) - 1):
            smoothed[i] = smoothed[i] * (1 - smoothing_factor) + \
                          (smoothed[i - 1] + smoothed[i + 1]) * 0.5 * smoothing_factor

        smoothed[0] = original_start
        smoothed[-1] = original_end

    return smoothed


def smooth_trajectory_spline(waypoints, num_points=None):
    """
    Smooth trajectory by fitting CubicSpline per dimension and resampling.
    Produces C2-continuous trajectory. waypoints: [N, d] (joint d=4 or task d=3).
    """
    waypoints = np.asarray(waypoints)
    if len(waypoints) <= 2:
        return waypoints
    n, d = waypoints.shape
    if num_points is None:
        num_points = n
    t_orig = np.linspace(0, 1, n)
    t_new = np.linspace(0, 1, num_points)
    smoothed = np.zeros((num_points, d))
    for j in range(d):
        cs = CubicSpline(t_orig, waypoints[:, j])
        smoothed[:, j] = cs(t_new)
    return smoothed


def smooth_trajectory_gaussian(waypoints, sigma=1.0, axis=0):
    """Smooth trajectory with 1D Gaussian filter along axis (default: along time)."""
    waypoints = np.asarray(waypoints)
    if len(waypoints) <= 2 or sigma <= 0:
        return waypoints
    smoothed = np.zeros_like(waypoints)
    for j in range(waypoints.shape[1]):
        smoothed[:, j] = gaussian_filter1d(waypoints[:, j], sigma=sigma, mode='nearest')
    return smoothed


def generate_smooth_trajectory(waypoints, speed_limit, t_total, t_sample):
    """
    Generate smooth trajectory based on given waypoints (fixed version)
    waypoints: path points [N, 3]
    speed_limit: maximum velocity limit
    t_total: total time
    t_sample: sampling time interval
    """
    num_waypoints = len(waypoints)
    if num_waypoints < 2:
        raise ValueError("Need at least two waypoints to generate trajectory")

    # Path length estimation
    path_lengths = [0]
    total_length = 0

    for i in range(1, num_waypoints):
        segment_length = np.linalg.norm(waypoints[i] - waypoints[i - 1])
        total_length += segment_length
        path_lengths.append(total_length)

    # Time allocation based on path length
    t_waypoints = np.zeros(num_waypoints)
    for i in range(1, num_waypoints):
        if total_length > 0:
            t_waypoints[i] = t_total * path_lengths[i] / total_length
        else:
            t_waypoints[i] = t_total * i / (num_waypoints - 1)

    # Ensure time is strictly increasing (fix duplicate time issue)
    min_time_diff = 1e-6
    for i in range(1, len(t_waypoints)):
        if t_waypoints[i] <= t_waypoints[i - 1]:
            t_waypoints[i] = t_waypoints[i - 1] + min_time_diff

    print(f"Time waypoints: {t_waypoints}")

    # Use cubic spline interpolation
    splines = []
    for dim in range(waypoints.shape[1]):
        spline = CubicSpline(t_waypoints, waypoints[:, dim])
        splines.append(spline)

    # Generate trajectory sampling points
    t_samples = np.arange(0, t_total, t_sample)
    # Ensure last sampling point doesn't exceed t_total
    if t_samples[-1] > t_total:
        t_samples = t_samples[t_samples <= t_total]

    positions = np.zeros((len(t_samples), waypoints.shape[1]))
    velocities = np.zeros((len(t_samples), waypoints.shape[1]))
    accelerations = np.zeros((len(t_samples), waypoints.shape[1]))

    for dim, spline in enumerate(splines):
        positions[:, dim] = spline(t_samples)
        velocities[:, dim] = spline.derivative(1)(t_samples)
        accelerations[:, dim] = spline.derivative(2)(t_samples)

    # Velocity limiting
    speeds = np.linalg.norm(velocities, axis=1)
    max_speed = np.max(speeds) if len(speeds) > 0 else 0

    if max_speed > speed_limit and max_speed > 0:
        scale_factor = max_speed / speed_limit
        t_total_adjusted = t_total * scale_factor
        t_samples_adjusted = np.arange(0, t_total_adjusted, t_sample)

        positions = np.zeros((len(t_samples_adjusted), waypoints.shape[1]))
        velocities = np.zeros((len(t_samples_adjusted), waypoints.shape[1]))
        accelerations = np.zeros((len(t_samples_adjusted), waypoints.shape[1]))

        # Re-adjust time points
        t_waypoints_adjusted = t_waypoints * scale_factor

        for dim in range(waypoints.shape[1]):
            adjusted_spline = CubicSpline(t_waypoints_adjusted, waypoints[:, dim])
            positions[:, dim] = adjusted_spline(t_samples_adjusted)
            velocities[:, dim] = adjusted_spline.derivative(1)(t_samples_adjusted) / scale_factor
            accelerations[:, dim] = adjusted_spline.derivative(2)(t_samples_adjusted) / (scale_factor ** 2)

    return positions


def find_optimal_ik_solution(task_goal_global, shoulder, d_uar, d_lar, joint_angle_bounds, method='global'):
    """
    Find joint configuration that reaches task goal with minimal ergonomic score through optimization

    Parameters:
    - task_goal_global: task space goal point (global frame)
    - shoulder: shoulder position (global frame)
    - d_uar, d_lar: upper arm and lower arm length
    - joint_angle_bounds: joint angle limits [(lower1, upper1), ...]
    - method: 'global' (global search) or 'local' (local optimization)

    Returns:
    - target_q: optimal joint configuration
    - target_hand: corresponding wrist position
    - target_score: corresponding ergonomic score
    - position_error: position error
    """
    print("\n" + "=" * 70)
    print("Starting optimization to find optimal IK configuration...")
    print("=" * 70)

    # Convert target point to shoulder frame
    task_goal_relative = task_goal_global - shoulder
    task_goal_shoulder = np.array([
        -task_goal_relative[1],
        -task_goal_relative[0],
        task_goal_relative[2]
    ])

    print(f"Task goal (global): {task_goal_global}")
    print(f"Task goal (shoulder frame): {task_goal_shoulder}")

    # Define optimization objective function with tighter constraint
    def objective_function(q):
        """
        Combined objective function: minimize (ergonomic_score + position_error_penalty)
        """
        # 1. Calculate ergonomic score
        ergo_score = utils.calculate_upper_limb_score_with_joint_angles(q)

        # 2. Calculate position error
        _, hand_shoulder = mos.forward_kinematics(q, d_uar, d_lar)
        hand_global = trans_shoulder2global(hand_shoulder, shoulder, arm='right')
        position_error = np.linalg.norm(hand_global - task_goal_global)

        # 3. TIGHTER position error penalty (increased from 1000.0 to 10000.0)
        position_penalty_weight = 10000.0
        position_penalty = position_penalty_weight * (position_error ** 2)

        # 4. Combined objective
        total_cost = ergo_score + position_penalty

        return total_cost

    # Tighter constraint function
    def position_constraint(q):
        """Position constraint: wrist position should be very close to target"""
        _, hand_shoulder = mos.forward_kinematics(q, d_uar, d_lar)
        hand_global = trans_shoulder2global(hand_shoulder, shoulder, arm='right')
        error = np.linalg.norm(hand_global - task_goal_global)
        return 0.005 - error  # Constraint: error <= 5mm (tightened from 20mm)

    if method == 'global':
        print("Using global optimization algorithm (Differential Evolution)...")

        # Use differential evolution for global search
        result = differential_evolution(
            objective_function,
            bounds=joint_angle_bounds,
            strategy='best1bin',
            maxiter=2000,  # Increased from 1000
            popsize=20,  # Increased from 15
            tol=1e-9,  # Tighter tolerance (from 1e-7)
            mutation=(0.5, 1.0),
            recombination=0.7,
            seed=None,
            disp=True,
            polish=True,
            init='latinhypercube',
            atol=1e-6,  # Added absolute tolerance
            updating='immediate',
            workers=1
        )

        if result.success:
            target_q = result.x
            print("Global optimization succeeded")
        else:
            print("Global optimization did not fully converge, using current best solution")
            target_q = result.x

    elif method == 'local':
        print("Using local optimization algorithm (SLSQP)...")

        # Initial guess: use middle configuration
        q0 = np.array([
            (joint_angle_bounds[i][0] + joint_angle_bounds[i][1]) / 2
            for i in range(len(joint_angle_bounds))
        ])

        # Define constraints
        constraints = {
            'type': 'ineq',
            'fun': position_constraint
        }

        # Use SLSQP for constrained optimization
        result = minimize(
            objective_function,
            q0,
            method='SLSQP',
            bounds=joint_angle_bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'disp': True, 'ftol': 1e-12}  # Tighter tolerance
        )

        if result.success:
            target_q = result.x
            print("Local optimization succeeded")
        else:
            print("Local optimization did not converge")
            target_q = result.x

    else:
        raise ValueError(f"Unknown optimization method: {method}")

    # Verify result
    _, hand_shoulder = mos.forward_kinematics(target_q, d_uar, d_lar)
    target_hand = trans_shoulder2global(hand_shoulder, shoulder, arm='right')
    target_score = utils.calculate_upper_limb_score_with_joint_angles(target_q)
    position_error = np.linalg.norm(target_hand - task_goal_global)

    print(f"\nOptimization result:")
    print(f"  Joint angles (radians): {target_q}")
    print(f"  Joint angles (degrees): {np.rad2deg(target_q)}")
    print(f"  Ergonomic score: {target_score:.4f}")
    print(f"  Position error: {position_error * 1000:.2f} mm")
    print(f"  Wrist position (global): {target_hand}")
    print(f"  Target position (global): {task_goal_global}")

    # Check if position constraint is satisfied (tighter threshold)
    if position_error > 0.01:  # Changed from 0.05 to 0.01 (10mm)
        print(f"Warning: Position error is large ({position_error * 1000:.2f} mm)")
        print("  Possible reasons: target point outside workspace or joint limits too strict")
    else:
        print("Position constraint satisfied")

    print("=" * 70 + "\n")

    return target_q, target_hand, target_score, position_error


def find_optimal_ik_solution_hybrid(task_goal_global, shoulder, d_uar, d_lar, joint_angle_bounds):
    """
    Hybrid optimization strategy with tighter constraints: global search for rough solution first, then local refinement

    This is the recommended method, combining robustness of global search and precision of local optimization
    """
    print("\n" + "=" * 70)
    print("Using hybrid optimization strategy to find optimal IK configuration...")
    print("=" * 70)

    # Convert target point to shoulder frame
    task_goal_relative = task_goal_global - shoulder
    task_goal_shoulder = np.array([
        -task_goal_relative[1],
        -task_goal_relative[0],
        task_goal_relative[2]
    ])

    print(f"Task goal (global): {task_goal_global}")
    print(f"Task goal (shoulder frame): {task_goal_shoulder}")

    # Phase 1: Global search with very tight position constraint
    print("\nPhase 1: Global search for feasible solution...")

    def phase1_objective(q):
        """Phase 1: primarily optimize position error with heavy penalty"""
        _, hand_shoulder = mos.forward_kinematics(q, d_uar, d_lar)
        hand_global = trans_shoulder2global(hand_shoulder, shoulder, arm='right')
        position_error = np.linalg.norm(hand_global - task_goal_global)

        ergo_score = utils.calculate_upper_limb_score_with_joint_angles(q)

        # Phase 1 weights: MUCH stronger position error weight
        return 1000.0 * position_error + 0.01 * ergo_score  # Increased from 100.0 to 1000.0

    result1 = differential_evolution(
        phase1_objective,
        bounds=joint_angle_bounds,
        strategy='best1bin',
        maxiter=500,  # Increased from 300
        popsize=15,  # Increased from 10
        tol=1e-8,  # Tighter tolerance
        seed=None,
        disp=False,
        polish=True,  # Changed to True
        workers=1
    )

    q_phase1 = result1.x
    _, hand_phase1_shoulder = mos.forward_kinematics(q_phase1, d_uar, d_lar)
    hand_phase1 = trans_shoulder2global(hand_phase1_shoulder, shoulder, arm='right')
    error_phase1 = np.linalg.norm(hand_phase1 - task_goal_global)
    score_phase1 = utils.calculate_upper_limb_score_with_joint_angles(q_phase1)

    print(f"Phase 1 result: Position error = {error_phase1 * 1000:.2f} mm, Score = {score_phase1:.4f}")

    # Phase 2: Local optimization with very tight position penalty
    print("\nPhase 2: Local optimization for ergonomics...")

    def phase2_objective(q):
        """Phase 2: optimize ergonomics while maintaining very tight position constraint"""
        _, hand_shoulder = mos.forward_kinematics(q, d_uar, d_lar)
        hand_global = trans_shoulder2global(hand_shoulder, shoulder, arm='right')
        position_error = np.linalg.norm(hand_global - task_goal_global)

        ergo_score = utils.calculate_upper_limb_score_with_joint_angles(q)

        # Phase 2 weights: maintain strong position constraint
        if position_error < 0.005:  # If error < 5mm
            return ergo_score + 1000.0 * (position_error ** 2)  # Increased from 50.0
        else:
            return ergo_score + 10000.0 * (position_error ** 2)  # Increased from 500.0

    result2 = minimize(
        phase2_objective,
        q_phase1,
        method='SLSQP',
        bounds=joint_angle_bounds,
        options={'maxiter': 500, 'disp': False, 'ftol': 1e-12}  # Increased iterations and tighter tolerance
    )

    target_q = result2.x

    # Final verification
    _, hand_shoulder = mos.forward_kinematics(target_q, d_uar, d_lar)
    target_hand = trans_shoulder2global(hand_shoulder, shoulder, arm='right')
    target_score = utils.calculate_upper_limb_score_with_joint_angles(target_q)
    position_error = np.linalg.norm(target_hand - task_goal_global)

    print(f"\nFinal optimization result:")
    print(f"  Joint angles (radians): {target_q}")
    print(f"  Joint angles (degrees): {np.rad2deg(target_q)}")
    print(f"  Ergonomic score: {target_score:.4f}")
    print(f"  Position error: {position_error * 1000:.2f} mm")
    print(f"  Improvement: Score {score_phase1:.4f} -> {target_score:.4f}")

    if position_error > 0.01:  # Tightened from 0.05 to 0.01 (10mm)
        print(f"Warning: Position error {position_error * 1000:.2f} mm exceeds threshold")
    else:
        print("Position constraint satisfied")

    print("=" * 70 + "\n")

    return target_q, target_hand, target_score, position_error


# ==================== NEW SDF-BASED METHODS ====================

class TrajectorySDFField:
    """
    Signed Distance Field constructed from reference trajectory
    """

    def __init__(self, reference_trajectory, sigma=0.05):
        """
        Initialize SDF field

        Parameters:
        - reference_trajectory: reference trajectory points [N, 3]
        - sigma: field smoothness parameter
        """
        self.reference_trajectory = np.array(reference_trajectory)
        self.sigma = sigma
        self.kdtree = KDTree(self.reference_trajectory)

        print(f"\n{'=' * 60}")
        print("Initializing Trajectory SDF Field")
        print(f"Reference trajectory points: {len(self.reference_trajectory)}")
        print(f"Smoothness parameter (sigma): {self.sigma}")
        print(f"{'=' * 60}\n")

    def compute_sdf(self, point):
        """
        Compute signed distance from point to trajectory

        Parameters:
        - point: query point [3,]

        Returns:
        - distance: signed distance value
        - closest_point: closest point on trajectory
        - closest_idx: index of closest point
        """
        distance, closest_idx = self.kdtree.query(point)
        closest_point = self.reference_trajectory[closest_idx]

        return distance, closest_point, closest_idx

    def compute_gradient(self, point, delta=1e-4):
        """
        Compute SDF gradient at point using finite differences

        Parameters:
        - point: query point [3,]
        - delta: finite difference step

        Returns:
        - gradient: SDF gradient [3,]
        """
        gradient = np.zeros(3)
        sdf_center, _, _ = self.compute_sdf(point)

        for i in range(3):
            point_plus = point.copy()
            point_plus[i] += delta
            sdf_plus, _, _ = self.compute_sdf(point_plus)

            gradient[i] = (sdf_plus - sdf_center) / delta

        return gradient

    def compute_tangent_direction(self, point):
        """
        Compute tangent direction of trajectory at closest point

        Parameters:
        - point: query point [3,]

        Returns:
        - tangent: tangent direction (normalized) [3,]
        """
        _, _, closest_idx = self.compute_sdf(point)

        # Get tangent direction from trajectory
        if closest_idx == 0:
            # At start, use forward difference
            tangent = self.reference_trajectory[1] - self.reference_trajectory[0]
        elif closest_idx == len(self.reference_trajectory) - 1:
            # At end, use backward difference
            tangent = self.reference_trajectory[-1] - self.reference_trajectory[-2]
        else:
            # In middle, use central difference
            tangent = self.reference_trajectory[closest_idx + 1] - self.reference_trajectory[closest_idx - 1]

        # Normalize
        tangent_norm = np.linalg.norm(tangent)
        if tangent_norm > 1e-6:
            tangent = tangent / tangent_norm
        else:
            tangent = np.zeros(3)

        return tangent


def generate_reference_trajectory(start_pos, goal_pos, num_points=50, trajectory_type='straight',
                                  transition_radius_ratio=0.15):
    """
    Generate reference trajectory

    Parameters:
    - start_pos: start position [3,]
    - goal_pos: goal position [3,]
    - num_points: number of trajectory points
    - trajectory_type: 'straight', 'curved', or 'box_carrying'
    - transition_radius_ratio: for 'box_carrying', arc radius = ratio * horizontal_dist (default 0.15)

    Returns:
    - trajectory: reference trajectory [N, 3]
    """
    start_pos = np.asarray(start_pos).ravel()[:3]
    goal_pos = np.asarray(goal_pos).ravel()[:3]

    if trajectory_type == 'straight':
        # Linear interpolation
        trajectory = np.linspace(start_pos, goal_pos, num_points)

    elif trajectory_type == 'curved':
        # Parabolic curve (arc upward or downward)
        t = np.linspace(0, 1, num_points)

        # Compute middle control point (offset perpendicular to line)
        direction = goal_pos - start_pos
        midpoint = (start_pos + goal_pos) / 2

        # Add vertical offset
        offset = np.array([0, 0, np.linalg.norm(direction) * 0.2])
        control_point = midpoint + offset

        # Quadratic Bezier curve
        trajectory = np.outer((1 - t) ** 2, start_pos) + \
                     np.outer(2 * (1 - t) * t, control_point) + \
                     np.outer(t ** 2, goal_pos)

    elif trajectory_type == 'box_carrying':
        # Horizontal in XY to above goal, then smooth transition (quarter-circle arc, center inward) and descent to goal.
        above_goal = np.array([goal_pos[0], goal_pos[1], start_pos[2]])
        horiz_dist = np.linalg.norm(above_goal - start_pos)
        vert_dist = abs(goal_pos[2] - start_pos[2])
        r = transition_radius_ratio * horiz_dist
        r = np.clip(r, 0.01, min(horiz_dist * 0.5, max(vert_dist * 0.5, 0.02)))

        if horiz_dist < 1e-6:
            trajectory = np.linspace(start_pos, goal_pos, num_points)
        else:
            u_h = (above_goal - start_pos) / horiz_dist
            e_z = np.array([0.0, 0.0, 1.0])
            Q1 = above_goal - r * u_h
            Q2 = above_goal - r * e_z
            # 过渡圆弧圆心在轨迹内侧（角点内侧），使圆弧向 L 形内侧弯曲
            center_inner = above_goal - r * u_h - r * e_z
            # 圆弧：从 Q1 到 Q2，圆心在 center_inner，theta 从 pi/2 到 0
            n_h = max(2, int(num_points * 0.4))
            n_arc = max(3, int(num_points * 0.25))
            n_v = max(2, num_points - n_h - n_arc)

            seg1 = np.linspace(start_pos, Q1, n_h)
            theta = np.linspace(np.pi / 2, 0, n_arc)
            seg2 = center_inner + r * np.outer(np.cos(theta), u_h) + r * np.outer(np.sin(theta), e_z)
            seg3 = np.linspace(Q2, goal_pos, n_v)

            trajectory = np.vstack([seg1, seg2, seg3])
            # 用路径长度参数化 + 三次样条重采样，使整条轨迹为平滑曲线
            n_pts = len(trajectory)
            s = np.zeros(n_pts)
            s[1:] = np.cumsum(np.linalg.norm(np.diff(trajectory, axis=0), axis=1))
            if s[-1] < 1e-9:
                s = np.linspace(0, 1, n_pts)
            else:
                s = s / s[-1]
            # CubicSpline 要求 x 严格单调递增，去除重复或零长段导致的 s 相等
            s = np.maximum.accumulate(s) + np.linspace(0, 1e-10 * (n_pts - 1), n_pts)
            s = (s - s[0]) / (s[-1] - s[0] + 1e-12)
            s_new = np.linspace(0, 1, num_points)
            trajectory = np.column_stack([
                CubicSpline(s, trajectory[:, 0])(s_new),
                CubicSpline(s, trajectory[:, 1])(s_new),
                CubicSpline(s, trajectory[:, 2])(s_new)
            ])

    else:
        raise ValueError(f"Unknown trajectory type: {trajectory_type}. "
                        f"Use one of: straight, curved, box_carrying")

    return np.asarray(trajectory)


def generate_shoulder_reference_trajectory(shoulder_center, num_points, amplitude_x=0.02, amplitude_y=0.02,
                                          amplitude_z=0.01, trajectory_type='sinusoidal'):
    """
    Generate small-range reference trajectory for shoulder to simulate real body motion.

    Parameters:
    - shoulder_center: center position of shoulder [3,] (e.g. initial shoulder position)
    - num_points: number of trajectory points (typically num_iterations)
    - amplitude_x, amplitude_y, amplitude_z: half-range of motion in each axis (meters)
    - trajectory_type: 'sinusoidal' (smooth periodic), 'circular' (xy circle), 'ellipse', 'straight' (linear segment)

    Returns:
    - trajectory: shoulder positions [N, 3]
    """
    shoulder_center = np.asarray(shoulder_center).ravel()[:3]
    t = np.linspace(0, 1, num_points)

    if trajectory_type == 'straight':
        # Straight line: from (center - amplitude) to (center + amplitude) in 3D
        start_pos = shoulder_center
        end_pos = shoulder_center + np.array([amplitude_x, amplitude_y, amplitude_z])
        trajectory = np.outer(1 - t, start_pos) + np.outer(t, end_pos)

    elif trajectory_type == 'sinusoidal':
        # Smooth sinusoidal motion in each axis (different phases to avoid pure line)
        dx = amplitude_x * np.sin(2 * np.pi * t)
        dy = amplitude_y * np.sin(2 * np.pi * t + 0.7)
        dz = amplitude_z * np.sin(2 * np.pi * t + 1.3)
        trajectory = shoulder_center + np.column_stack([dx, dy, dz])

    elif trajectory_type == 'circular':
        # Circular motion in XY plane, slight Z variation
        dx = amplitude_x * np.cos(2 * np.pi * t)
        dy = amplitude_y * np.sin(2 * np.pi * t)
        dz = amplitude_z * np.sin(2 * np.pi * t * 0.5)
        trajectory = shoulder_center + np.column_stack([dx, dy, dz])

    elif trajectory_type == 'ellipse':
        # Elliptic motion
        dx = amplitude_x * np.cos(2 * np.pi * t)
        dy = amplitude_y * 0.6 * np.sin(2 * np.pi * t)
        dz = amplitude_z * np.sin(2 * np.pi * t)
        trajectory = shoulder_center + np.column_stack([dx, dy, dz])

    else:
        raise ValueError(f"Unknown shoulder trajectory type: {trajectory_type}. "
                        f"Use one of: sinusoidal, circular, ellipse, straight")

    return np.array(trajectory)


def ik_target_point(hand_target_global, shoulder, q_init, d_uar, d_lar, joint_angle_bounds,
                    maxiter=100, ftol=1e-8):
    """
    Simple IK: find q such that wrist (endpoint) reaches hand_target_global in task space.
    Returns (q, position_error).
    """
    hand_target_relative = hand_target_global - shoulder
    hand_target_shoulder = np.array([
        -hand_target_relative[1],
        -hand_target_relative[0],
        hand_target_relative[2]
    ])

    def objective(q):
        _, hand_shoulder = mos.forward_kinematics(q, d_uar, d_lar)
        return np.sum((hand_shoulder - hand_target_shoulder) ** 2)

    result = minimize(
        objective,
        q_init,
        method='SLSQP',
        bounds=joint_angle_bounds,
        options={'maxiter': maxiter, 'ftol': ftol}
    )
    q = result.x
    _, hand_shoulder = mos.forward_kinematics(q, d_uar, d_lar)
    hand_global = trans_shoulder2global(hand_shoulder, shoulder, arm='right')
    pos_error = np.linalg.norm(hand_global - hand_target_global)
    return q, pos_error


def compute_ergonomic_vector_task_space(endpoint_global, shoulder, q_current, d_uar, d_lar,
                                        joint_angle_bounds, neighbor_radius=0.02, n_samples=27,
                                        joint_neighbor_radius=0.06):
    """
    In joint space, sample neighbors of q_current; find the q with lowest ergo score (optimal_q),
    then map to task space to get the corresponding wrist position; return the direction from
    current endpoint to that position as the ergonomic vector.

    Returns:
    - ergo_direction: [3,] normalized vector pointing toward task-space position of optimal_q,
      or zero vector if no valid neighbor found.
    """
    q_current = np.asarray(q_current)
    ndof = len(q_current)

    # 1. Sample joint-space neighbors of q_current (random perturbations, then clip to bounds)
    neighbors_q = []
    for _ in range(n_samples):
        # Random direction in joint space, radius scaled by joint_neighbor_radius
        u = np.random.randn(ndof)
        u = u / (np.linalg.norm(u) + 1e-8)
        r = joint_neighbor_radius * (np.random.rand() ** (1 / ndof))
        q_neighbor = q_current + r * u
        # Clip to joint limits
        for i in range(ndof):
            low, high = joint_angle_bounds[i]
            q_neighbor[i] = np.clip(q_neighbor[i], low, high)
        neighbors_q.append(q_neighbor)

    # 2. Find optimal_q: neighbor with lowest ergonomic score
    best_score = np.inf
    optimal_q = None
    for q in neighbors_q:
        score = utils.calculate_upper_limb_score_with_joint_angles(q)
        if score < best_score:
            best_score = score
            optimal_q = q.copy()

    if optimal_q is None:
        return np.zeros(3)

    # 3. Map optimal_q to task space (wrist position in global frame)
    _, hand_shoulder = mos.forward_kinematics(optimal_q, d_uar, d_lar)
    pos_optimal = trans_shoulder2global(hand_shoulder, shoulder, arm='right')

    # 4. Ergonomic vector: from current endpoint toward optimal position
    vector_ergo = pos_optimal - endpoint_global
    norm = np.linalg.norm(vector_ergo)
    if norm < 1e-6:
        return np.zeros(3)
    return vector_ergo / norm


def run_iterations_task_space_direct(num_iterations, task_goal_global, reference_trajectory=None,
                                    shoulder_trajectory=None, w_goal=0.33, w_ref=0.33, w_ergo=0.34,
                                    method_name='Unified', goal_threshold=0.01, step_size=0.04,
                                    neighbor_radius=0.02, n_ergo_samples=27,
                                    use_moving_average=True, moving_avg_window=5):
    """
    Unified task-space direct motion planning. Plan trajectory by moving the skeleton endpoint
    (wrist) in task space. At each step:
    1. Goal vector = (task_goal - endpoint), normalized.
    2. Ref vector = (closest_point_on_reference_trajectory - endpoint), normalized (if reference given).
    3. Ergo vector = direction toward the point in a neighbor range that has lowest ergo score.
    4. Combined direction = w_goal*goal + w_ref*ref + w_ergo*ergo, normalized.
    5. Next waypoint = endpoint + step_size * combined_direction.
    6. IK(next_waypoint) -> q_next; update skeleton and trajectories.

    Returns:
    - trajectory: list of q
    - trajectory_hand: array [N, 3] wrist positions in global frame
    - score_history: list of ergo scores
    - joint_history: list of q
    - sdf_values: list of distances to reference trajectory (if reference given)
    - weights_history: list of dict with w_goal, w_ref, w_ergo (for logging)
    """
    global current_q, global_positions, shoulder, d_uar, d_lar, joint_angle_bounds

    ref_traj = np.array(reference_trajectory) if reference_trajectory is not None else None
    ref_kdtree = KDTree(ref_traj) if ref_traj is not None else None

    trajectory = []
    trajectory_hand = []
    trajectory_elbow = []
    score_history = []
    joint_history = []
    sdf_values = []
    weights_history = []

    if shoulder_trajectory is not None:
        shoulder = np.array(shoulder_trajectory[0]).copy()
        global_positions[3] = shoulder.copy()

    q_current = current_q.copy()
    trajectory.append(q_current.copy())

    _, hand_shoulder = mos.forward_kinematics(q_current, d_uar, d_lar)
    endpoint = trans_shoulder2global(hand_shoulder, shoulder, arm='right')
    trajectory_hand.append(endpoint.copy())

    initial_goal_dist = np.linalg.norm(task_goal_global - endpoint)
    print(f"[{method_name}] Task-space direct planning. Initial goal dist: {initial_goal_dist:.4f} m")
    print(f"  Weights: goal={w_goal:.2f}, ref={w_ref:.2f}, ergo={w_ergo:.2f}\n")

    for step in range(num_iterations):
        if shoulder_trajectory is not None:
            shoulder = np.array(shoulder_trajectory[min(step, len(shoulder_trajectory) - 1)]).copy()
            global_positions[3] = shoulder.copy()

        _, hand_current_shoulder = mos.forward_kinematics(q_current, d_uar, d_lar)
        endpoint = trans_shoulder2global(hand_current_shoulder, shoulder, arm='right')
        elbow_shoulder, _ = mos.forward_kinematics(q_current, d_uar, d_lar)
        elbow_global = trans_shoulder2global(elbow_shoulder, shoulder, arm='right')
        # Sync skeleton to current step's shoulder + q_current so ergo score is associated with correct pose
        global_positions[4] = elbow_global.copy()
        global_positions[5] = endpoint.copy()
        current_score = utils.calculate_upper_limb_score_with_joint_angles(q_current)

        score_history.append(current_score)
        joint_history.append(q_current.copy())

        goal_dist = np.linalg.norm(task_goal_global - endpoint)
        if goal_dist < goal_threshold:
            print(f"[{method_name}] Reached goal at iteration {step}")
            trajectory_hand.append(endpoint.copy())
            trajectory_elbow.append(elbow_global.copy())
            break

        # 1. Goal vector (task space)
        goal_vec = task_goal_global - endpoint
        if np.linalg.norm(goal_vec) > 1e-8:
            goal_vec = goal_vec / np.linalg.norm(goal_vec)
        else:
            goal_vec = np.zeros(3)

        # 2. Ref vector: closest point on reference trajectory (task space)
        if reference_trajectory is not None and w_ref > 0:
            _, closest_idx = ref_kdtree.query(endpoint, k=1)
            idx = np.atleast_1d(closest_idx)[0]
            closest_point = ref_traj[idx]
            ref_vec = closest_point - endpoint
            sdf_dist = np.linalg.norm(ref_vec)
            sdf_values.append(sdf_dist)
            if np.linalg.norm(ref_vec) > 1e-8:
                ref_vec = ref_vec / np.linalg.norm(ref_vec)
            else:
                ref_vec = np.zeros(3)
        else:
            ref_vec = np.zeros(3)
            sdf_values.append(0.0)

        # 3. Ergonomic vector: direction toward lowest-ergo point in neighborhood (task space)
        if w_ergo > 0:
            ergo_vec = compute_ergonomic_vector_task_space(
                endpoint, shoulder, q_current, d_uar, d_lar, joint_angle_bounds,
                neighbor_radius=neighbor_radius, n_samples=n_ergo_samples
            )
        else:
            ergo_vec = np.zeros(3)

        # 4. Combined direction (task space)
        combined = w_goal * goal_vec + w_ref * ref_vec + w_ergo * ergo_vec
        combined_norm = np.linalg.norm(combined)
        if combined_norm > 1e-8:
            combined = combined / combined_norm
        else:
            combined = goal_vec

        weights_history.append({'w_goal': w_goal, 'w_ref': w_ref, 'w_ergo': w_ergo})

        # 5. Next waypoint in task space
        adaptive_step = min(step_size, goal_dist * 0.4)
        next_waypoint = endpoint + adaptive_step * combined

        # 6. IK to get q_next
        q_next, ik_error = ik_target_point(next_waypoint, shoulder, q_current, d_uar, d_lar,
                                           joint_angle_bounds, maxiter=120, ftol=1e-8)

        if use_moving_average and len(trajectory) >= moving_avg_window:
            recent_qs = np.array(trajectory[-(moving_avg_window - 1):] + [q_next])
            q_smoothed = np.mean(recent_qs, axis=0)
            _, hand_smoothed = mos.forward_kinematics(q_smoothed, d_uar, d_lar)
            hand_smoothed_global = trans_shoulder2global(hand_smoothed, shoulder, arm='right')
            if np.linalg.norm(hand_smoothed_global - next_waypoint) < 0.03:
                q_next = q_smoothed

        trajectory.append(q_next.copy())
        _, new_hand_shoulder = mos.forward_kinematics(q_next, d_uar, d_lar)
        new_elbow_shoulder, _ = mos.forward_kinematics(q_next, d_uar, d_lar)
        new_hand_global = trans_shoulder2global(new_hand_shoulder, shoulder, arm='right')
        new_elbow_global = trans_shoulder2global(new_elbow_shoulder, shoulder, arm='right')
        trajectory_hand.append(new_hand_global.copy())
        trajectory_elbow.append(new_elbow_global.copy())

        global_positions[4] = new_elbow_global
        global_positions[5] = new_hand_global

        if step % 5 == 0 or step == num_iterations - 1:
            print(f"[{method_name}] Iter {step:3d}: Score={current_score:.4f}, "
                  f"Goal={goal_dist*1000:.1f}mm, IK_err={ik_error*1000:.2f}mm")

        q_current = q_next.copy()

    current_q = q_current.copy()
    traj_hand_arr = np.array(trajectory_hand)
    # Multi-stage smoothing for smoother trajectories
    if len(trajectory) > 5:
        traj_arr = np.array(trajectory)
        q_first_orig = traj_arr[0].copy()
        q_last_orig = traj_arr[-1].copy()
        # Stage 1: iterative averaging (preserves start/end)
        traj_arr = smooth_trajectory(traj_arr, smoothing_factor=0.52, iterations=4)
        # Stage 2: Gaussian filter along time (reduces jerk)
        traj_arr = smooth_trajectory_gaussian(traj_arr, sigma=1.2)
        # Stage 3: spline resampling for C2 continuity (optional, keeps same N)
        traj_arr = smooth_trajectory_spline(traj_arr, num_points=len(traj_arr))
        # Restore exact first/last joint so all three methods have identical start/end positions
        traj_arr[0] = q_first_orig
        traj_arr[-1] = q_last_orig
        trajectory = traj_arr.tolist()
        # Rebuild hand trajectory using per-step shoulder (fixes Y-direction start mismatch)
        n_pts = len(trajectory)
        if shoulder_trajectory is not None:
            sh_traj = np.array(shoulder_trajectory)
            n_sh = len(sh_traj)
            # Map trajectory index i to shoulder index (same relative progress along path)
            traj_hand_arr = np.array([
                trans_shoulder2global(mos.forward_kinematics(trajectory[i], d_uar, d_lar)[1],
                                      sh_traj[min(int(i * (n_sh - 1) / max(n_pts - 1, 1)), n_sh - 1)], arm='right')
                for i in range(n_pts)
            ])
        else:
            sh = shoulder.copy()
            traj_hand_arr = np.array([
                trans_shoulder2global(mos.forward_kinematics(q, d_uar, d_lar)[1], sh, arm='right')
                for q in trajectory
            ])
    q_final = np.array(trajectory[-1])
    final_score = utils.calculate_upper_limb_score_with_joint_angles(q_final)
    final_error = np.linalg.norm(traj_hand_arr[-1] - task_goal_global)
    print(f"[{method_name}] Done: Score={final_score:.4f}, Final error={final_error*1000:.2f} mm, Points={len(trajectory)}\n")
    return trajectory, traj_hand_arr, score_history, joint_history, sdf_values, weights_history


def run_iterations_with_sdf_guidance(num_iterations, task_goal_global, reference_trajectory,
                                     shoulder_trajectory=None):
    """
    Method 2: SDF-guided. Task-space direct planning with weights: goal + reference trajectory
    (no ergo vector). Endpoint moves toward goal and toward closest point on reference trajectory.
    """
    trajectory, trajectory_hand_sdf, score_history_sdf, joint_history_sdf, sdf_values, _ = \
        run_iterations_task_space_direct(
            num_iterations, task_goal_global,
            reference_trajectory=reference_trajectory,
            shoulder_trajectory=shoulder_trajectory,
            w_goal=0.3, w_ref=0.7, w_ergo=0.0,
            method_name='SDF',
            goal_threshold=0.01, step_size=0.04,
            use_moving_average=True, moving_avg_window=3
        )
    return trajectory, trajectory_hand_sdf, score_history_sdf, joint_history_sdf, sdf_values


def run_iterations_with_optimized_ik(num_iterations, task_goal_global, optimization_method='hybrid',
                                    shoulder_trajectory=None):
    """
    Method 1: Ergonomic field. Task-space direct planning with weights: goal + ergo vector
    (no reference trajectory). Endpoint moves toward goal and toward lowest-ergo point in neighborhood.
    """
    global current_q, global_positions, trajectory_hand, trajectory_elbow, score_history, joint_history, shoulder

    trajectory_result, traj_hand, score_history_out, joint_history_out, _, _ = \
        run_iterations_task_space_direct(
            num_iterations, task_goal_global,
            reference_trajectory=None,
            shoulder_trajectory=shoulder_trajectory,
            w_goal=0.55, w_ref=0.0, w_ergo=0.45,   # 提高 goal 引力，减小与 target 的位移差
            method_name='Ergo',
            goal_threshold=0.01, step_size=0.04,
            neighbor_radius=0.02, n_ergo_samples=27,
            use_moving_average=True, moving_avg_window=3
        )
    trajectory_hand = list(traj_hand)
    score_history[:] = score_history_out
    joint_history[:] = joint_history_out
    target_q = np.array(trajectory_result[-1])
    target_hand = np.array(traj_hand[-1])
    return trajectory_result, target_q, target_hand


def run_iterations_in_joint_space(num_iterations, target_q, task_goal_global=None, shoulder_trajectory=None):
    """
    CSEF-guided trajectory planning in joint space with tighter convergence criteria

    Parameters:
    - num_iterations: number of iterations
    - target_q: target joint configuration
    - task_goal_global: task space goal (for visualization only)
    - shoulder_trajectory: optional [N, 3] shoulder positions per step (moving base)
    """
    global current_q, global_positions, trajectory_hand, trajectory_elbow, score_history, joint_history, shoulder

    # CSEF field parameters
    q_opt = np.array(optimal_q)
    weights = np.array([1.0, 1.0, 1.0, 2.0])
    comfort_threshold = 0.1

    # Weight parameters - adjusted for tighter tracking
    alpha = 0.8  # Increased goal direction weight (from 0.7)
    beta = 0.2  # Decreased ergonomic gradient weight (from 0.3)
    step_size = 0.08

    # Helper functions to calculate SEF value and gradient
    def calculate_sef(q):
        ergo_score = utils.calculate_upper_limb_score_with_joint_angles(q)
        return ergo_score - comfort_threshold

    def calculate_sef_gradient(q, delta=1e-5):
        grad = np.zeros_like(q)
        sef_q = calculate_sef(q)

        for i in range(len(q)):
            q_plus = q.copy()
            q_plus[i] += delta
            sef_plus = calculate_sef(q_plus)
            grad[i] = (sef_plus - sef_q) / delta

        return grad

    def enforce_joint_limits(q, bounds):
        q_limited = np.copy(q)
        for i in range(len(q)):
            q_limited[i] = np.clip(q[i], bounds[i][0], bounds[i][1])
        return q_limited

    # If shoulder trajectory provided, use first point for initial state and sync skeleton
    if shoulder_trajectory is not None:
        shoulder = np.array(shoulder_trajectory[0]).copy()
        global_positions[3] = shoulder.copy()   # skeleton shoulder origin

    # Main loop
    trajectory = []
    q_current = current_q.copy()
    trajectory.append(q_current.copy())

    print(f"Joint space planning: from current_q to target_q")
    print(f"Initial joint distance: {np.linalg.norm(target_q - q_current):.4f} rad")
    print(f"Initial joint angles: {np.rad2deg(q_current)}")
    print(f"Target joint angles: {np.rad2deg(target_q)}\n")

    for step in range(num_iterations):
        # Update shoulder from reference trajectory (moving base) and sync skeleton model
        if shoulder_trajectory is not None:
            shoulder = np.array(shoulder_trajectory[min(step, len(shoulder_trajectory) - 1)]).copy()
            global_positions[3] = shoulder.copy()   # skeleton shoulder origin

        # 1. Calculate current state (using updated shoulder origin)
        elbow_current_shoulder, hand_current_shoulder = mos.forward_kinematics(q_current, d_uar, d_lar)
        hand_current_global = trans_shoulder2global(hand_current_shoulder, shoulder, arm='right')
        elbow_current_global = trans_shoulder2global(elbow_current_shoulder, shoulder, arm='right')
        # Sync skeleton to current step's shoulder + q_current so ergo score is associated with correct pose
        global_positions[4] = elbow_current_global.copy()
        global_positions[5] = hand_current_global.copy()
        current_score = utils.calculate_upper_limb_score_with_joint_angles(q_current)

        # Record history
        score_history.append(current_score)
        joint_history.append(q_current.copy())

        # 2. Calculate goal direction (joint space)
        goal_direction = target_q - q_current
        goal_distance = np.linalg.norm(goal_direction)

        # Tighter early termination check (from 0.01 to 0.005)
        if goal_distance < 0.005:  # About 0.29 degrees
            print(f"Reached target configuration at iteration {step}")
            # Add final position
            trajectory.append(target_q.copy())
            q_current = target_q.copy()

            # Update global state (skeleton: shoulder already set at start of step)
            final_elbow_shoulder, final_hand_shoulder = mos.forward_kinematics(q_current, d_uar, d_lar)
            final_hand_global = trans_shoulder2global(final_hand_shoulder, shoulder, arm='right')
            final_elbow_global = trans_shoulder2global(final_elbow_shoulder, shoulder, arm='right')

            global_positions[4] = final_elbow_global
            global_positions[5] = final_hand_global
            trajectory_hand.append(final_hand_global.copy())
            trajectory_elbow.append(final_elbow_global.copy())

            break

        # Normalize goal direction
        if goal_distance > 1e-6:
            goal_direction_normalized = goal_direction / goal_distance
        else:
            goal_direction_normalized = np.zeros(4)

        # 3. Get SEF gradient
        sef_gradient = calculate_sef_gradient(q_current)
        sef_gradient_norm = np.linalg.norm(sef_gradient)

        if sef_gradient_norm > 1e-6:
            sef_gradient_normalized = sef_gradient / sef_gradient_norm
        else:
            sef_gradient_normalized = np.zeros(4)

        # 4. Combine directions
        combined_direction = alpha * goal_direction_normalized - beta * sef_gradient_normalized
        combined_norm = np.linalg.norm(combined_direction)

        if combined_norm > 1e-6:
            combined_direction_normalized = combined_direction / combined_norm
        else:
            combined_direction_normalized = goal_direction_normalized

        # 5. Adaptive step size (reduce step size when close to goal)
        adaptive_step = min(step_size, goal_distance * 0.5)

        # 6. Update joint angles
        q_next = q_current + adaptive_step * combined_direction_normalized

        # 7. Ensure joint angles are within limits
        q_next = enforce_joint_limits(q_next, joint_angle_bounds)

        # 8. Update trajectory and global positions
        trajectory.append(q_next.copy())

        new_elbow_shoulder, new_hand_shoulder = mos.forward_kinematics(q_next, d_uar, d_lar)
        new_hand_global = trans_shoulder2global(new_hand_shoulder, shoulder, arm='right')
        new_elbow_global = trans_shoulder2global(new_elbow_shoulder, shoulder, arm='right')

        global_positions[4] = new_elbow_global
        global_positions[5] = new_hand_global

        trajectory_hand.append(new_hand_global.copy())
        trajectory_elbow.append(new_elbow_global.copy())

        # 9. Print progress
        if step % 5 == 0 or step == num_iterations - 1:
            task_distance = np.linalg.norm(
                hand_current_global - task_goal_global) if task_goal_global is not None else 0
            print(f"Iter {step:3d}: Score={current_score:.4f}, "
                  f"Joint dist={goal_distance:.4f} rad ({np.rad2deg(goal_distance):.2f} deg), "
                  f"Task dist={task_distance * 1000:.2f} mm, "
                  f"Step size={adaptive_step:.4f}")

        # 10. Update current state
        q_current = q_next.copy()

    # Update global state
    current_q = q_current.copy()

    # Final verification
    _, final_hand_shoulder = mos.forward_kinematics(current_q, d_uar, d_lar)
    final_hand_global = trans_shoulder2global(final_hand_shoulder, shoulder, arm='right')
    final_score = utils.calculate_upper_limb_score_with_joint_angles(current_q)

    print(f"\nFinal result:")
    print(f"  Final joint angles (degrees): {np.rad2deg(current_q)}")
    print(f"  Final Score: {final_score:.4f}")
    print(f"  Joint distance: {np.linalg.norm(current_q - target_q):.6f} rad")

    if task_goal_global is not None:
        final_task_error = np.linalg.norm(final_hand_global - task_goal_global)
        print(f"  Task space error: {final_task_error * 1000:.2f} mm")

    print(f"  Trajectory points: {len(trajectory)}")

    return trajectory


def run_iterations_with_hybrid_guidance(num_iterations, task_goal_global, reference_trajectory,
                                        shoulder_trajectory=None):
    """
    Method 3: Hybrid. Task-space direct planning with weights: goal + reference trajectory + ergo vector.
    Endpoint moves toward goal, toward closest point on reference trajectory, and toward lowest-ergo
    point in neighborhood.
    """
    (trajectory, trajectory_hand_hybrid, score_history_hybrid, joint_history_hybrid,
     sdf_values, guidance_weights_history) = run_iterations_task_space_direct(
        num_iterations, task_goal_global,
        reference_trajectory=reference_trajectory,
        shoulder_trajectory=shoulder_trajectory,
        w_goal=0.45, w_ref=0.25, w_ergo=0.3,
        method_name='Hybrid',
        goal_threshold=0.01, step_size=0.04,
        neighbor_radius=0.02, n_ergo_samples=27,
        use_moving_average=True, moving_avg_window=3
    )
    return (trajectory, trajectory_hand_hybrid, score_history_hybrid, joint_history_hybrid,
            sdf_values, guidance_weights_history)



def compare_three_methods(trajectory_hand_ergo, score_history_ergo, joint_history_ergo,
                          trajectory_hand_sdf, score_history_sdf, joint_history_sdf, sdf_values_sdf,
                          trajectory_hand_hybrid, score_history_hybrid, joint_history_hybrid, sdf_values_hybrid,
                          weights_history_hybrid,
                          reference_trajectory, task_goal_global, shoulder_trajectory=None):
    """
    Compare results of THREE planning methods with comprehensive visualization.
    shoulder_trajectory: optional [N, 3] shoulder positions (moving base); used for reference shoulder in plot.
    """
    shoulder_ref = np.array(shoulder_trajectory[0]).copy() if shoulder_trajectory is not None else np.array(shoulder).copy()

    print("\n" + "=" * 70)
    print("COMPARISON OF THREE PLANNING METHODS")
    print("=" * 70)

    # ============== Figure 1: 3D Trajectory Comparison (1x3 layout) ==============
    fig1 = plt.figure(figsize=(24, 8))

    # Define common view parameters
    xlim, ylim, zlim = (1.7, 2.2), (0.05, 0.55), (1.1, 1.6)
    elev, azim = 25, -35

    # Subplot 1: Ergonomic Field Method
    ax1 = fig1.add_subplot(131, projection='3d')
    ax1.set_xlim(xlim)
    ax1.set_ylim(ylim)
    ax1.set_zlim(zlim)
    ax1.view_init(elev=10, azim=80)

    plot_right_upper_limb_skeleton(ax1, global_positions, color='gray', linewidth=3)

    if shoulder_trajectory is not None:
        sh_traj = np.array(shoulder_trajectory)
        ax1.plot(sh_traj[:, 0], sh_traj[:, 1], sh_traj[:, 2],
                 c='gray', linewidth=1.5, alpha=0.6, linestyle=':', label='Shoulder path')
    ax1.scatter(shoulder_ref[0], shoulder_ref[1], shoulder_ref[2],
                c='black', s=120, label='Shoulder (ref)', marker='o', edgecolors='white', linewidth=1.5)
    ax1.scatter(optimal_position[0], optimal_position[1], optimal_position[2],
                c='magenta', s=120, label='Ergonomic Optimal', marker='^', edgecolors='white', linewidth=1.5)

    traj_ergo = np.array(trajectory_hand_ergo)
    num_points_ergo = len(traj_ergo)
    for i in range(num_points_ergo - 1):
        color_ratio = i / (num_points_ergo - 1)
        color = (0, color_ratio, 1 - color_ratio)
        ax1.plot(traj_ergo[i:i + 2, 0], traj_ergo[i:i + 2, 1], traj_ergo[i:i + 2, 2],
                 c=color, linewidth=3.5, alpha=0.9)

    ax1.scatter(traj_ergo[0, 0], traj_ergo[0, 1], traj_ergo[0, 2],
                c='cyan', s=150, marker='o', label='Start', edgecolors='black', linewidth=2)
    ax1.scatter(traj_ergo[-1, 0], traj_ergo[-1, 1], traj_ergo[-1, 2],
                c='green', s=150, marker='s', label='End', edgecolors='black', linewidth=2)
    ax1.scatter(task_goal_global[0], task_goal_global[1], task_goal_global[2],
                c='red', s=300, marker='*', label='Goal', edgecolors='darkred', linewidth=2.5)

    ax1.set_xlabel('X (m)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Y (m)', fontsize=12, fontweight='bold')
    ax1.set_zlabel('Z (m)', fontsize=12, fontweight='bold')
    ax1.set_title('Method 1: Ergonomic Field', fontsize=14, fontweight='bold', pad=20)
    ax1.legend(fontsize=9, loc='upper left', framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle='--')

    path_length_ergo = np.sum(np.linalg.norm(np.diff(traj_ergo, axis=0), axis=1))
    final_error_ergo = np.linalg.norm(traj_ergo[-1] - task_goal_global)
    text_ergo = f'Points: {len(traj_ergo)}\nLength: {path_length_ergo:.3f}m\nError: {final_error_ergo * 1000:.1f}mm\nScore: {score_history_ergo[-1]:.3f}'
    ax1.text2D(0.02, 0.98, text_ergo, transform=ax1.transAxes, fontsize=10,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.85))

    # Subplot 2: SDF Method
    ax2 = fig1.add_subplot(132, projection='3d')
    ax2.set_xlim(xlim)
    ax2.set_ylim(ylim)
    ax2.set_zlim(zlim)
    ax2.view_init(elev=10, azim=80)

    plot_right_upper_limb_skeleton(ax2, global_positions, color='gray', linewidth=3)

    if shoulder_trajectory is not None:
        sh_traj = np.array(shoulder_trajectory)
        ax2.plot(sh_traj[:, 0], sh_traj[:, 1], sh_traj[:, 2],
                 c='gray', linewidth=1.5, alpha=0.6, linestyle=':', label='Shoulder path')
    ax2.scatter(shoulder_ref[0], shoulder_ref[1], shoulder_ref[2],
                c='black', s=120, label='Shoulder (ref)', marker='o', edgecolors='white', linewidth=1.5)

    ref_traj = np.array(reference_trajectory)
    ax2.plot(ref_traj[:, 0], ref_traj[:, 1], ref_traj[:, 2],
             color='gold', linewidth=4, alpha=0.7, linestyle='--',
             label='Reference Trajectory', zorder=1)

    traj_sdf = np.array(trajectory_hand_sdf)
    num_points_sdf = len(traj_sdf)
    for i in range(num_points_sdf - 1):
        color_ratio = i / (num_points_sdf - 1)
        color = (1 - color_ratio, 0, color_ratio)
        ax2.plot(traj_sdf[i:i + 2, 0], traj_sdf[i:i + 2, 1], traj_sdf[i:i + 2, 2],
                 c=color, linewidth=3.5, alpha=0.9)

    ax2.scatter(traj_sdf[0, 0], traj_sdf[0, 1], traj_sdf[0, 2],
                c='cyan', s=150, marker='o', label='Start', edgecolors='black', linewidth=2)
    ax2.scatter(traj_sdf[-1, 0], traj_sdf[-1, 1], traj_sdf[-1, 2],
                c='blue', s=150, marker='s', label='End', edgecolors='black', linewidth=2)
    ax2.scatter(task_goal_global[0], task_goal_global[1], task_goal_global[2],
                c='red', s=300, marker='*', label='Goal', edgecolors='darkred', linewidth=2.5)

    ax2.set_xlabel('X (m)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Y (m)', fontsize=12, fontweight='bold')
    ax2.set_zlabel('Z (m)', fontsize=12, fontweight='bold')
    ax2.set_title('Method 2: SDF Guidance', fontsize=14, fontweight='bold', pad=20)
    ax2.legend(fontsize=9, loc='upper left', framealpha=0.9)
    ax2.grid(True, alpha=0.3, linestyle='--')

    path_length_sdf = np.sum(np.linalg.norm(np.diff(traj_sdf, axis=0), axis=1))
    final_error_sdf = np.linalg.norm(traj_sdf[-1] - task_goal_global)
    avg_sdf = np.mean(sdf_values_sdf)
    text_sdf = f'Points: {len(traj_sdf)}\nLength: {path_length_sdf:.3f}m\nError: {final_error_sdf * 1000:.1f}mm\nScore: {score_history_sdf[-1]:.3f}\nAvg SDF: {avg_sdf * 1000:.1f}mm'
    ax2.text2D(0.02, 0.98, text_sdf, transform=ax2.transAxes, fontsize=10,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.85))

    # Subplot 3: Hybrid Method
    ax3 = fig1.add_subplot(133, projection='3d')
    ax3.set_xlim(xlim)
    ax3.set_ylim(ylim)
    ax3.set_zlim(zlim)
    ax3.view_init(elev=10, azim=80)

    plot_right_upper_limb_skeleton(ax3, global_positions, color='gray', linewidth=3)

    if shoulder_trajectory is not None:
        sh_traj = np.array(shoulder_trajectory)
        ax3.plot(sh_traj[:, 0], sh_traj[:, 1], sh_traj[:, 2],
                 c='gray', linewidth=1.5, alpha=0.6, linestyle=':', label='Shoulder path')
    ax3.scatter(shoulder_ref[0], shoulder_ref[1], shoulder_ref[2],
                c='black', s=120, label='Shoulder (ref)', marker='o', edgecolors='white', linewidth=1.5)
    ax3.scatter(optimal_position[0], optimal_position[1], optimal_position[2],
                c='magenta', s=120, label='Ergonomic Optimal', marker='^', edgecolors='white', linewidth=1.5)

    ax3.plot(ref_traj[:, 0], ref_traj[:, 1], ref_traj[:, 2],
             color='gold', linewidth=3, alpha=0.6, linestyle='--',
             label='Reference Trajectory', zorder=1)

    traj_hybrid = np.array(trajectory_hand_hybrid)
    num_points_hybrid = len(traj_hybrid)
    for i in range(num_points_hybrid - 1):
        color_ratio = i / (num_points_hybrid - 1)
        color = (0.5 - 0.5 * color_ratio, 0.5 * color_ratio, 0.5 + 0.5 * color_ratio)  # Purple gradient
        ax3.plot(traj_hybrid[i:i + 2, 0], traj_hybrid[i:i + 2, 1], traj_hybrid[i:i + 2, 2],
                 c=color, linewidth=3.5, alpha=0.9)

    ax3.scatter(traj_hybrid[0, 0], traj_hybrid[0, 1], traj_hybrid[0, 2],
                c='cyan', s=150, marker='o', label='Start', edgecolors='black', linewidth=2)
    ax3.scatter(traj_hybrid[-1, 0], traj_hybrid[-1, 1], traj_hybrid[-1, 2],
                c='purple', s=150, marker='s', label='End', edgecolors='black', linewidth=2)
    ax3.scatter(task_goal_global[0], task_goal_global[1], task_goal_global[2],
                c='red', s=300, marker='*', label='Goal', edgecolors='darkred', linewidth=2.5)

    ax3.set_xlabel('X (m)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Y (m)', fontsize=12, fontweight='bold')
    ax3.set_zlabel('Z (m)', fontsize=12, fontweight='bold')
    ax3.set_title('Method 3: Hybrid SDF-SEF', fontsize=14, fontweight='bold', pad=20)
    ax3.legend(fontsize=9, loc='upper left', framealpha=0.9)
    ax3.grid(True, alpha=0.3, linestyle='--')

    path_length_hybrid = np.sum(np.linalg.norm(np.diff(traj_hybrid, axis=0), axis=1))
    final_error_hybrid = np.linalg.norm(traj_hybrid[-1] - task_goal_global)
    avg_sdf_hybrid = np.mean(sdf_values_hybrid)
    text_hybrid = f'Points: {len(traj_hybrid)}\nLength: {path_length_hybrid:.3f}m\nError: {final_error_hybrid * 1000:.1f}mm\nScore: {score_history_hybrid[-1]:.3f}\nAvg SDF: {avg_sdf_hybrid * 1000:.1f}mm'
    ax3.text2D(0.02, 0.98, text_hybrid, transform=ax3.transAxes, fontsize=10,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='plum', alpha=0.85))

    plt.tight_layout()
    plt.savefig('comparison_3methods_3d_trajectories.png', dpi=300, bbox_inches='tight')
    plt.show()

    # ============== Print Summary Table ==============
    print("\nDETAILED THREE-METHOD COMPARISON TABLE")
    print("=" * 90)
    print(f"{'Metric':<30} {'Method 1 (Ergo)':<20} {'Method 2 (SDF)':<20} {'Method 3 (Hybrid)':<20}")
    print("-" * 90)
    print(f"{'Path Length (m)':<30} {path_length_ergo:<20.4f} {path_length_sdf:<20.4f} {path_length_hybrid:<20.4f}")
    print(
        f"{'Final Error (mm)':<30} {final_error_ergo * 1000:<20.2f} {final_error_sdf * 1000:<20.2f} {final_error_hybrid * 1000:<20.2f}")
    print(
        f"{'Initial Score':<30} {score_history_ergo[0]:<20.4f} {score_history_sdf[0]:<20.4f} {score_history_hybrid[0]:<20.4f}")
    print(
        f"{'Final Score':<30} {score_history_ergo[-1]:<20.4f} {score_history_sdf[-1]:<20.4f} {score_history_hybrid[-1]:<20.4f}")
    print(
        f"{'Score Improvement':<30} {score_history_ergo[0] - score_history_ergo[-1]:<20.4f} {score_history_sdf[0] - score_history_sdf[-1]:<20.4f} {score_history_hybrid[0] - score_history_hybrid[-1]:<20.4f}")
    print(
        f"{'Iterations':<30} {len(score_history_ergo):<20} {len(score_history_sdf):<20} {len(score_history_hybrid):<20}")
    print(f"{'Avg SDF Distance (mm)':<30} {'-':<20} {avg_sdf * 1000:<20.2f} {avg_sdf_hybrid * 1000:<20.2f}")
    print("=" * 90)

    # ============== Figure 3: XYZ Direction Comparison ==============
    fig3 = plt.figure(figsize=(20, 6))

    # Extract trajectory coordinates
    traj_ergo_x = traj_ergo[:, 0]
    traj_ergo_y = traj_ergo[:, 1]
    traj_ergo_z = traj_ergo[:, 2]

    traj_sdf_x = traj_sdf[:, 0]
    traj_sdf_y = traj_sdf[:, 1]
    traj_sdf_z = traj_sdf[:, 2]

    traj_hybrid_x = traj_hybrid[:, 0]
    traj_hybrid_y = traj_hybrid[:, 1]
    traj_hybrid_z = traj_hybrid[:, 2]

    # Reference trajectory
    ref_x = ref_traj[:, 0]
    ref_y = ref_traj[:, 1]
    ref_z = ref_traj[:, 2]

    # Normalize trajectory indices for comparison
    indices_ergo = np.linspace(0, 100, len(traj_ergo))
    indices_sdf = np.linspace(0, 100, len(traj_sdf))
    indices_hybrid = np.linspace(0, 100, len(traj_hybrid))
    indices_ref = np.linspace(0, 100, len(ref_traj))

    # Subplot 1: X Direction
    ax1 = fig3.add_subplot(1, 3, 1)
    ax1.plot(indices_ergo, traj_ergo_x, linewidth=2.5, color='steelblue',
             label='Method 1: Ergonomic', alpha=0.8, marker='o', markersize=3, markevery=len(traj_ergo) // 10)
    ax1.plot(indices_sdf, traj_sdf_x, linewidth=2.5, color='coral',
             label='Method 2: SDF', alpha=0.8, marker='s', markersize=3, markevery=len(traj_sdf) // 10)
    ax1.plot(indices_hybrid, traj_hybrid_x, linewidth=2.5, color='purple',
             label='Method 3: Hybrid', alpha=0.8, marker='^', markersize=3, markevery=len(traj_hybrid) // 10)
    ax1.plot(indices_ref, ref_x, linewidth=2, color='gold',
             label='Reference', alpha=0.6, linestyle='--')

    # Mark start and goal
    ax1.axhline(y=task_goal_global[0], color='red', linestyle=':', linewidth=2, alpha=0.5, label='Goal X')
    ax1.scatter([0], [traj_ergo_x[0]], color='cyan', s=150, marker='o',
                edgecolors='black', linewidth=2, zorder=5, label='Start')

    ax1.set_xlabel('Trajectory Progress (%)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('X Position (m)', fontsize=12, fontweight='bold')
    ax1.set_title('X-Direction Trajectory Comparison', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=9, loc='best', framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle='--')

    # Calculate X direction statistics
    x_range_ergo = np.max(traj_ergo_x) - np.min(traj_ergo_x)
    x_range_sdf = np.max(traj_sdf_x) - np.min(traj_sdf_x)
    x_range_hybrid = np.max(traj_hybrid_x) - np.min(traj_hybrid_x)
    text_x = f'Range:\nM1: {x_range_ergo:.3f}m\nM2: {x_range_sdf:.3f}m\nM3: {x_range_hybrid:.3f}m'
    ax1.text(0.02, 0.98, text_x, transform=ax1.transAxes, fontsize=9,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Subplot 2: Y Direction
    ax2 = fig3.add_subplot(1, 3, 2)
    ax2.plot(indices_ergo, traj_ergo_y, linewidth=2.5, color='steelblue',
             label='Method 1: Ergonomic', alpha=0.8, marker='o', markersize=3, markevery=len(traj_ergo) // 10)
    ax2.plot(indices_sdf, traj_sdf_y, linewidth=2.5, color='coral',
             label='Method 2: SDF', alpha=0.8, marker='s', markersize=3, markevery=len(traj_sdf) // 10)
    ax2.plot(indices_hybrid, traj_hybrid_y, linewidth=2.5, color='purple',
             label='Method 3: Hybrid', alpha=0.8, marker='^', markersize=3, markevery=len(traj_hybrid) // 10)
    ax2.plot(indices_ref, ref_y, linewidth=2, color='gold',
             label='Reference', alpha=0.6, linestyle='--')

    # Mark start and goal
    ax2.axhline(y=task_goal_global[1], color='red', linestyle=':', linewidth=2, alpha=0.5, label='Goal Y')
    ax2.scatter([0], [traj_ergo_y[0]], color='cyan', s=150, marker='o',
                edgecolors='black', linewidth=2, zorder=5, label='Start')

    ax2.set_xlabel('Trajectory Progress (%)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Y Position (m)', fontsize=12, fontweight='bold')
    ax2.set_title('Y-Direction Trajectory Comparison', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=9, loc='best', framealpha=0.9)
    ax2.grid(True, alpha=0.3, linestyle='--')

    # Calculate Y direction statistics
    y_range_ergo = np.max(traj_ergo_y) - np.min(traj_ergo_y)
    y_range_sdf = np.max(traj_sdf_y) - np.min(traj_sdf_y)
    y_range_hybrid = np.max(traj_hybrid_y) - np.min(traj_hybrid_y)
    text_y = f'Range:\nM1: {y_range_ergo:.3f}m\nM2: {y_range_sdf:.3f}m\nM3: {y_range_hybrid:.3f}m'
    ax2.text(0.02, 0.98, text_y, transform=ax2.transAxes, fontsize=9,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Subplot 3: Z Direction
    ax3 = fig3.add_subplot(1, 3, 3)
    ax3.plot(indices_ergo, traj_ergo_z, linewidth=2.5, color='steelblue',
             label='Method 1: Ergonomic', alpha=0.8, marker='o', markersize=3, markevery=len(traj_ergo) // 10)
    ax3.plot(indices_sdf, traj_sdf_z, linewidth=2.5, color='coral',
             label='Method 2: SDF', alpha=0.8, marker='s', markersize=3, markevery=len(traj_sdf) // 10)
    ax3.plot(indices_hybrid, traj_hybrid_z, linewidth=2.5, color='purple',
             label='Method 3: Hybrid', alpha=0.8, marker='^', markersize=3, markevery=len(traj_hybrid) // 10)
    ax3.plot(indices_ref, ref_z, linewidth=2, color='gold',
             label='Reference', alpha=0.6, linestyle='--')

    # Mark start and goal
    ax3.axhline(y=task_goal_global[2], color='red', linestyle=':', linewidth=2, alpha=0.5, label='Goal Z')
    ax3.scatter([0], [traj_ergo_z[0]], color='cyan', s=150, marker='o',
                edgecolors='black', linewidth=2, zorder=5, label='Start')

    ax3.set_xlabel('Trajectory Progress (%)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Z Position (m)', fontsize=12, fontweight='bold')
    ax3.set_title('Z-Direction Trajectory Comparison', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=9, loc='best', framealpha=0.9)
    ax3.grid(True, alpha=0.3, linestyle='--')

    # Calculate Z direction statistics
    z_range_ergo = np.max(traj_ergo_z) - np.min(traj_ergo_z)
    z_range_sdf = np.max(traj_sdf_z) - np.min(traj_sdf_z)
    z_range_hybrid = np.max(traj_hybrid_z) - np.min(traj_hybrid_z)
    text_z = f'Range:\nM1: {z_range_ergo:.3f}m\nM2: {z_range_sdf:.3f}m\nM3: {z_range_hybrid:.3f}m'
    ax3.text(0.02, 0.98, text_z, transform=ax3.transAxes, fontsize=9,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.savefig('comparison_3methods_xyz_directions.png', dpi=300, bbox_inches='tight')
    plt.show()

    # ============== Print XYZ Direction Statistics ==============
    print("\nXYZ DIRECTION STATISTICS")
    print("=" * 90)
    print(f"{'Direction':<15} {'Metric':<20} {'Method 1':<20} {'Method 2':<20} {'Method 3':<20}")
    print("-" * 90)
    print(f"{'X-Direction':<15} {'Range (m)':<20} {x_range_ergo:<20.4f} {x_range_sdf:<20.4f} {x_range_hybrid:<20.4f}")
    print(
        f"{'X-Direction':<15} {'Start (m)':<20} {traj_ergo_x[0]:<20.4f} {traj_sdf_x[0]:<20.4f} {traj_hybrid_x[0]:<20.4f}")
    print(
        f"{'X-Direction':<15} {'End (m)':<20} {traj_ergo_x[-1]:<20.4f} {traj_sdf_x[-1]:<20.4f} {traj_hybrid_x[-1]:<20.4f}")
    print(
        f"{'X-Direction':<15} {'Goal (m)':<20} {task_goal_global[0]:<20.4f} {task_goal_global[0]:<20.4f} {task_goal_global[0]:<20.4f}")
    print("-" * 90)
    print(f"{'Y-Direction':<15} {'Range (m)':<20} {y_range_ergo:<20.4f} {y_range_sdf:<20.4f} {y_range_hybrid:<20.4f}")
    print(
        f"{'Y-Direction':<15} {'Start (m)':<20} {traj_ergo_y[0]:<20.4f} {traj_sdf_y[0]:<20.4f} {traj_hybrid_y[0]:<20.4f}")
    print(
        f"{'Y-Direction':<15} {'End (m)':<20} {traj_ergo_y[-1]:<20.4f} {traj_sdf_y[-1]:<20.4f} {traj_hybrid_y[-1]:<20.4f}")
    print(
        f"{'Y-Direction':<15} {'Goal (m)':<20} {task_goal_global[1]:<20.4f} {task_goal_global[1]:<20.4f} {task_goal_global[1]:<20.4f}")
    print("-" * 90)
    print(f"{'Z-Direction':<15} {'Range (m)':<20} {z_range_ergo:<20.4f} {z_range_sdf:<20.4f} {z_range_hybrid:<20.4f}")
    print(
        f"{'Z-Direction':<15} {'Start (m)':<20} {traj_ergo_z[0]:<20.4f} {traj_sdf_z[0]:<20.4f} {traj_hybrid_z[0]:<20.4f}")
    print(
        f"{'Z-Direction':<15} {'End (m)':<20} {traj_ergo_z[-1]:<20.4f} {traj_sdf_z[-1]:<20.4f} {traj_hybrid_z[-1]:<20.4f}")
    print(
        f"{'Z-Direction':<15} {'Goal (m)':<20} {task_goal_global[2]:<20.4f} {task_goal_global[2]:<20.4f} {task_goal_global[2]:<20.4f}")
    print("=" * 90)


def visualize_3d_trajectory(task_goal_global=None):
    """Visualize 3D trajectory in a separate larger figure with skeleton"""
    fig = plt.figure(figsize=(14, 11))
    ax = fig.add_subplot(111, projection='3d')

    ax.set_xlim((1.2, 2.2))
    ax.set_ylim((-0.2, 0.8))
    ax.view_init(elev=30, azim=-30)

    # Plot right upper limb skeleton only (shoulder -> elbow -> wrist)
    plot_right_upper_limb_skeleton(ax, global_positions, color='gray', linewidth=3)

    new_elbow = global_positions[4]
    new_hand = global_positions[5]

    # Plot key points
    ax.scatter(shoulder[0], shoulder[1], shoulder[2], c='black', s=120, label='Shoulder', marker='o',
               edgecolors='white', linewidth=1.5)
    ax.scatter(new_elbow[0], new_elbow[1], new_elbow[2], c='blue', s=120, label='Elbow', marker='o', edgecolors='white',
               linewidth=1.5)
    ax.scatter(new_hand[0], new_hand[1], new_hand[2], c='green', s=120, label='Wrist (Final)', marker='o',
               edgecolors='white', linewidth=1.5)

    if task_goal_global is not None:
        ax.scatter(task_goal_global[0], task_goal_global[1], task_goal_global[2],
                   c='red', s=250, marker='*', label='Task Goal', edgecolors='darkred', linewidth=2.5)

        # Add error distance line
        ax.plot([new_hand[0], task_goal_global[0]],
                [new_hand[1], task_goal_global[1]],
                [new_hand[2], task_goal_global[2]],
                'r--', linewidth=2.5, alpha=0.7,
                label=f'Error: {np.linalg.norm(new_hand - task_goal_global) * 1000:.2f} mm')

    # Plot wrist trajectory with gradient color
    traj = np.array(trajectory_hand)
    num_points = len(traj)

    # Create color gradient from blue to green
    for i in range(num_points - 1):
        color_ratio = i / (num_points - 1)
        color = (0, color_ratio, 1 - color_ratio)  # RGB gradient
        ax.plot(traj[i:i + 2, 0], traj[i:i + 2, 1], traj[i:i + 2, 2],
                c=color, linewidth=3, alpha=0.9)

    # Plot initial and final wrist positions
    ax.scatter(traj[0, 0], traj[0, 1], traj[0, 2],
               c='cyan', s=120, marker='s', label='Start Position', edgecolors='black', linewidth=1.5)

    # Plot optimal ergonomic position
    ax.scatter(optimal_position[0], optimal_position[1], optimal_position[2],
               c='magenta', s=140, label='Ergonomic Optimal', marker='^', edgecolors='white', linewidth=1.5)

    # Labels and styling
    ax.set_xlabel('X (m)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Y (m)', fontsize=14, fontweight='bold')
    ax.set_zlabel('Z (m)', fontsize=14, fontweight='bold')
    ax.set_title('3D Trajectory Visualization with Skeleton', fontsize=16, fontweight='bold', pad=20)

    # Legend
    ax.legend(fontsize=11, loc='upper left', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')

    # Add text annotation showing trajectory statistics
    traj_length = np.sum(np.linalg.norm(np.diff(traj, axis=0), axis=1))
    if task_goal_global is not None:
        final_error = np.linalg.norm(new_hand - task_goal_global)
        text_str = f'Trajectory Points: {len(traj)}\nPath Length: {traj_length:.3f} m\nFinal Error: {final_error * 1000:.2f} mm'
    else:
        text_str = f'Trajectory Points: {len(traj)}\nPath Length: {traj_length:.3f} m'

    ax.text2D(0.02, 0.98, text_str, transform=ax.transAxes, fontsize=11,
              verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))

    plt.tight_layout()
    plt.show()


def visualize_final_result(task_goal_global=None, target_q=None):
    """Visualize final result in multiple subplots"""

    # First show the 3D trajectory separately
    visualize_3d_trajectory(task_goal_global)

    # Then show the analysis plots
    fig = plt.figure(figsize=(16, 5))

    # Subplot 1: Ergonomic Score history
    ax1 = fig.add_subplot(121)
    iterations = range(1, len(score_history) + 1)
    ax1.plot(iterations, score_history, linewidth=2.5, color='steelblue')
    ax1.axhline(y=score_history[0], color='red', linestyle='--', alpha=0.5, label=f'Initial: {score_history[0]:.3f}')
    ax1.axhline(y=score_history[-1], color='green', linestyle='--', alpha=0.5, label=f'Final: {score_history[-1]:.3f}')
    ax1.set_xlabel('Iteration', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Ergonomic Score', fontsize=12, fontweight='bold')
    ax1.set_title('Ergonomic Score Evolution', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Subplot 2: Joint angle changes
    ax2 = fig.add_subplot(122)
    if len(joint_history) > 0:
        joint_history_array = np.array(joint_history)
        joint_labels = ['Shoulder Flexion', 'Shoulder Abduction', 'Elbow Flexion', 'Forearm Rotation']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

        for i in range(4):
            ax2.plot(range(1, len(joint_history) + 1),
                     np.rad2deg(joint_history_array[:, i]),
                     label=joint_labels[i], linewidth=2.5, color=colors[i])

        if target_q is not None:
            for i in range(4):
                ax2.axhline(y=np.rad2deg(target_q[i]), color=colors[i],
                            linestyle=':', alpha=0.5, linewidth=2)

    ax2.set_xlabel('Iteration', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Joint Angle (deg)', fontsize=12, fontweight='bold')
    ax2.set_title('Joint Angle Evolution', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    ## Initialization of robot end effector poses
    robot_left_position_init = np.array([0.85, 0.15, 1.3])
    robot_right_position_init = np.array([0.85, -0.25, 0.7])

    robot_left_rotation_matrix_init = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    robot_right_rotation_matrix_init = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])

    robot_left_pose_matrix_init = np.r_[
        np.c_[robot_left_rotation_matrix_init, robot_left_position_init.T], np.array([[0, 0, 0, 1]])]
    robot_right_pose_matrix_init = np.r_[
        np.c_[robot_right_rotation_matrix_init, robot_right_position_init.T], np.array([[0, 0, 0, 1]])]

    base2torso_matrix = np.array([[1, 0, 0, -0.29], [0, 1, 0, 0], [0, 0, 1, -0.985], [0, 0, 0, 1]])
    initial_robot_left_pose_matrix = base2torso_matrix @ robot_left_pose_matrix_init
    initial_robot_right_pose_matrix = base2torso_matrix @ robot_right_pose_matrix_init

    print("init_left_arm_pub", initial_robot_left_pose_matrix)
    print("init_right_arm_pub", initial_robot_right_pose_matrix)

    sub_robot = np.array([-0.2195, 1.11462, 0, 0, 0, 0, 1])
    sub_shouL = np.array([2, 1.5, 0.25, 0, 0, 0, 1])
    sub_shouR = np.array([2, 1.5, -0.25, 0, 0, 0, 1])
    sub_elbowL = np.array([1.9, 1.3, 0.3, 0, 0, 0, 1])
    sub_elbowR = np.array([1.9, 1.3, -0.3, 0, 0, 0, 1])
    sub_wristL = np.array([1.8, 1.2, 0.3, 0, 0, 0, 1])
    sub_wristR = np.array([1.8, 1.4, -0.3, 0, 0, 0, 1])

    T_optitrack2robotbase = np.linalg.inv(
        tsf.transform_optitrack_origin_to_optitrack_robot(
            sub_robot) @ tsf.transform_optitrack_robot_to_robot_base())
    shouL_position_init = T_optitrack2robotbase[:3, :3] @ sub_shouL[:3] + T_optitrack2robotbase[:3, 3]
    shouR_position_init = T_optitrack2robotbase[:3, :3] @ sub_shouR[:3] + T_optitrack2robotbase[:3, 3]
    elbowL_position_init = T_optitrack2robotbase[:3, :3] @ sub_elbowL[:3] + T_optitrack2robotbase[:3, 3]
    elbowR_position_init = T_optitrack2robotbase[:3, :3] @ sub_elbowR[:3] + T_optitrack2robotbase[:3, 3]
    wristL_position_init = T_optitrack2robotbase[:3, :3] @ sub_wristL[:3] + T_optitrack2robotbase[:3, 3]
    wristR_position_init = T_optitrack2robotbase[:3, :3] @ sub_wristR[:3] + T_optitrack2robotbase[:3, 3]

    joint_angle_bounds = [
        (-math.pi / 18, 17 * math.pi / 18),  # Joint 1
        (-math.pi / 18, 17 * math.pi / 18),  # Joint 2
        (-np.pi / 3, np.pi / 2),  # Joint 3
        (-np.pi / 2, np.pi / 3)  # Joint 4
    ]
    optimal_q = [0, 0, 0, -math.pi / 4]

    # skeleton_joint_name, skeleton_joints, skeleton_parent_indices, skeleton_joint_local_translation = \
    #     utils.read_skeleton_motion('/home/clover/Chenzui/Ergo-Manip/data/demo_2_test_chenzui_only_optitrack2hotu.npy')
    skeleton_joint_name, skeleton_joints, skeleton_parent_indices, skeleton_joint_local_translation = \
        utils.read_skeleton_motion('/home/clover/Chenzui/Ergo-Manip/data/demo_2_test_chenzui_only_optitrack2hotu.npy')
    skeleton_joint = skeleton_joints[500, :]
    global_positions, global_rotations = utils.forward_kinematics(skeleton_joint_local_translation,
                                                                  skeleton_joint, skeleton_parent_indices)
    global_positions[:, 2] = global_positions[:, 2] * 1.2

    global_positions[4] = global_positions[3] + (elbowR_position_init - shouR_position_init)
    global_positions[7] = global_positions[6] + (elbowL_position_init - shouL_position_init)
    global_positions[5] = global_positions[3] + (wristR_position_init - shouR_position_init)
    global_positions[8] = global_positions[6] + (wristL_position_init - shouL_position_init)

    shou_center = shouR_position_init
    global_positions = global_positions + np.array([shou_center[0], shou_center[1], 0])

    initial_position = global_positions[5]

    # Body dimensions
    d_ual, d_uar, d_lal, d_lar = mos.calculate_arm_dimensions(shouR_position_init, elbowR_position_init,
                                                              wristR_position_init, shouR_position_init,
                                                              elbowR_position_init, wristR_position_init)

    # Calculate initial "optimal" position
    _, optimal_position = mos.forward_kinematics(optimal_q, d_uar, d_lar)
    optimal_position = trans_shoulder2global(optimal_position, global_positions[3], arm='right')

    p_elbowR_init, p_wristR_init = trans_global2shoulder(global_positions[3], global_positions[4], global_positions[5],
                                                         arm='right')
    current_q = mos.inverse_kinematics(p_elbowR_init, p_wristR_init, d_uar, d_lar)
    _, p_hand = mos.forward_kinematics(current_q, d_uar, d_lar)
    print("p_wristR_init", p_wristR_init)
    print("current_q", current_q)
    print("p_hand", p_hand)
    current_score = utils.calculate_upper_limb_score_with_joint_angles(current_q)

    hand_current = global_positions[5]
    elbow_current = global_positions[4]
    shoulder = global_positions[3].copy()

    # Record trajectory history for animation
    trajectory_hand = [hand_current.copy()]
    trajectory_elbow = [elbow_current.copy()]

    score_history = []
    joint_history = []

    # Set number of iterations
    num_iterations = 100

    # Generate shoulder reference trajectory (small-range motion to simulate real body movement)
    # trajectory_type: 'sinusoidal', 'circular', 'ellipse', 'straight'
    shoulder_trajectory = generate_shoulder_reference_trajectory(
        shoulder,
        num_points=num_iterations,
        amplitude_x=0.00,   # ±2cm in X
        amplitude_y=0.10,   # ±2cm in Y
        amplitude_z=-0.00,   # ±1cm in Z
        trajectory_type='straight'   # 直线：从 (center-amp) 到 (center+amp) 线性插值
    )
    print(f"Shoulder moving-base: trajectory length={len(shoulder_trajectory)}, type=straight, "
          f"amplitude approx. ±(2,2,1)cm (x,y,z)")

    # Define task goal
    # 定义任务目标
    task_goal_global = hand_current + np.array([0.1, 0.0, -0.2])

    print(f"\n{'=' * 70}")
    print(f"Starting THREE-METHOD COMPARATIVE trajectory planning study (moving shoulder base)")
    print(f"Task goal (global): {task_goal_global}")
    print(f"{'=' * 70}\n")

    # ========== METHOD 1: Ergonomic Field Guidance ==========
    print("\n" + "=" * 70)
    print("RUNNING METHOD 1: ERGONOMIC FIELD GUIDANCE")
    print("=" * 70)

    # Save full initial state so Method 2 and 3 start from the same pose (fix Y-direction start mismatch)
    current_q_backup = current_q.copy()
    shoulder_backup = shoulder.copy()
    global_positions_backup = {3: global_positions[3].copy(), 4: global_positions[4].copy(), 5: global_positions[5].copy()}

    trajectory_hand = [hand_current.copy()]
    trajectory_elbow = [elbow_current.copy()]
    score_history = []
    joint_history = []

    trajectory_result_ergo, target_q, target_hand = run_iterations_with_optimized_ik(
        num_iterations=num_iterations,
        task_goal_global=task_goal_global,
        optimization_method='hybrid',
        shoulder_trajectory=shoulder_trajectory
    )

    trajectory_hand_ergo = np.array(trajectory_hand).copy()
    score_history_ergo = score_history.copy()
    joint_history_ergo = joint_history.copy()

    print(f"\nMethod 1 completed: Final Score={score_history_ergo[-1]:.4f}, Points={len(trajectory_hand_ergo)}")

    # ========== METHOD 2: SDF Trajectory Guidance ==========
    print("\n" + "=" * 70)
    print("RUNNING METHOD 2: SDF TRAJECTORY GUIDANCE")
    print("=" * 70)

    reference_trajectory = generate_reference_trajectory(
        task_goal_global + np.array([-0.0, 0.0, 0.3]),
        task_goal_global,
        num_points=50,
        trajectory_type='straight'
    )

    # Restore full initial state so Method 2 starts from same Y (and X,Z) as Method 1
    current_q = current_q_backup.copy()
    shoulder = shoulder_backup.copy()
    global_positions[3] = global_positions_backup[3].copy()
    global_positions[4] = global_positions_backup[4].copy()
    global_positions[5] = global_positions_backup[5].copy()

    trajectory_result_sdf, trajectory_hand_sdf, score_history_sdf, joint_history_sdf, sdf_values = \
        run_iterations_with_sdf_guidance(
            num_iterations=num_iterations,
            task_goal_global=task_goal_global,
            reference_trajectory=reference_trajectory,
            shoulder_trajectory=shoulder_trajectory
        )

    print(f"\nMethod 2 completed: Final Score={score_history_sdf[-1]:.4f}, Points={len(trajectory_hand_sdf)}")

    # ========== METHOD 3: Hybrid SDF-SEF Guidance ==========
    print("\n" + "=" * 70)
    print("RUNNING METHOD 3: HYBRID SDF-SEF GUIDANCE")
    print("=" * 70)

    # Restore full initial state so Method 3 starts from same Y (and X,Z) as Method 1
    current_q = current_q_backup.copy()
    shoulder = shoulder_backup.copy()
    global_positions[3] = global_positions_backup[3].copy()
    global_positions[4] = global_positions_backup[4].copy()
    global_positions[5] = global_positions_backup[5].copy()

    (trajectory_result_hybrid, trajectory_hand_hybrid, score_history_hybrid,
     joint_history_hybrid, sdf_values_hybrid, weights_history_hybrid) = \
        run_iterations_with_hybrid_guidance(
            num_iterations=num_iterations,
            task_goal_global=task_goal_global,
            reference_trajectory=reference_trajectory,
            shoulder_trajectory=shoulder_trajectory
        )

    print(f"\nMethod 3 completed: Final Score={score_history_hybrid[-1]:.4f}, Points={len(trajectory_hand_hybrid)}")

    # ========== COMPARISON AND VISUALIZATION ==========
    compare_three_methods(
        trajectory_hand_ergo, score_history_ergo, joint_history_ergo,
        trajectory_hand_sdf, score_history_sdf, joint_history_sdf, sdf_values,
        trajectory_hand_hybrid, score_history_hybrid, joint_history_hybrid, sdf_values_hybrid, weights_history_hybrid,
        reference_trajectory, task_goal_global,
        shoulder_trajectory=shoulder_trajectory
    )

    print("\n" + "=" * 70)
    print("THREE-METHOD COMPARATIVE STUDY COMPLETED!")
    print("=" * 70)