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

import sys
import subprocess

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


def generate_reference_trajectory(start_pos, goal_pos, num_points=50, trajectory_type='straight'):
    """
    Generate reference trajectory

    Parameters:
    - start_pos: start position [3,]
    - goal_pos: goal position [3,]
    - num_points: number of trajectory points
    - trajectory_type: 'straight' or 'curved'

    Returns:
    - trajectory: reference trajectory [N, 3]
    """
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

    else:
        raise ValueError(f"Unknown trajectory type: {trajectory_type}")

    return trajectory


def run_iterations_with_sdf_guidance(num_iterations, task_goal_global, reference_trajectory):
    """
    SDF-guided trajectory planning with Adaptive Weighting Strategy

    Key Innovation:
    - Large SDF distance → Follow gradient (pull toward trajectory)
    - Small SDF distance → Follow tangent (smooth flow along trajectory)
    """
    global current_q, global_positions, shoulder, d_uar, d_lar, joint_angle_bounds

    print("\n" + "=" * 70)
    print("Starting SDF-guided trajectory planning with Adaptive Weights")
    print("=" * 70)

    # Initialize SDF field
    sdf_field = TrajectorySDFField(reference_trajectory, sigma=0.05)

    # ========== ADAPTIVE WEIGHT PARAMETERS ==========
    # Define SDF distance thresholds for adaptive behavior
    sdf_near_threshold = 0.01  # 3cm - considered "near" the trajectory
    sdf_far_threshold = 0.03  # 10cm - considered "far" from trajectory

    # Weight ranges for adaptive interpolation
    # When FAR (large SDF): prioritize gradient to pull back to trajectory
    alpha_far = 0.85  # Strong gradient following
    beta_far = 0.1  # Weak tangent following
    gamma_far = 0.05  # Weak goal attraction

    # When NEAR (small SDF): prioritize tangent for smooth flow
    alpha_near = 0.1  # Weak gradient following
    beta_near = 0.8  # Strong tangent following
    gamma_near = 0.1  # Weak goal attraction

    # Step size parameters
    step_size_base = 0.05  # Base step size
    step_size_near = 0.04  # Smaller steps when near trajectory
    step_size_far = 0.06  # Larger steps when far from trajectory

    # Smoothness parameters
    joint_velocity_weight = 0.3
    joint_acceleration_weight = 0.2
    use_moving_average = True
    moving_avg_window = 3

    # Convergence threshold
    goal_threshold = 0.01  # 10mm

    # Storage
    trajectory = []
    trajectory_hand_sdf = []
    trajectory_elbow_sdf = []
    score_history_sdf = []
    joint_history_sdf = []
    sdf_values = []
    adaptive_weights_history = []  # Track weight evolution

    joint_velocities = []
    joint_accelerations = []

    q_current = current_q.copy()
    trajectory.append(q_current.copy())

    # Initial state
    _, hand_current_shoulder = mos.forward_kinematics(q_current, d_uar, d_lar)
    hand_current_global = trans_shoulder2global(hand_current_shoulder, shoulder, arm='right')
    trajectory_hand_sdf.append(hand_current_global.copy())

    print(f"Initial position: {hand_current_global}")
    print(f"Target position: {task_goal_global}")
    print(f"Initial distance: {np.linalg.norm(task_goal_global - hand_current_global):.4f} m")
    print(f"\nAdaptive Weight Strategy:")
    print(
        f"  Near threshold (<{sdf_near_threshold * 1000:.0f}mm): α={alpha_near:.1f}, β={beta_near:.1f}, γ={gamma_near:.1f}")
    print(
        f"  Far threshold (>{sdf_far_threshold * 1000:.0f}mm): α={alpha_far:.1f}, β={beta_far:.1f}, γ={gamma_far:.1f}")
    print(f"  Transition zone: smooth interpolation\n")

    for step in range(num_iterations):
        # 1. Compute current state
        elbow_current_shoulder, hand_current_shoulder = mos.forward_kinematics(q_current, d_uar, d_lar)
        hand_current_global = trans_shoulder2global(hand_current_shoulder, shoulder, arm='right')
        elbow_current_global = trans_shoulder2global(elbow_current_shoulder, shoulder, arm='right')
        current_score = utils.calculate_upper_limb_score_with_joint_angles(q_current)

        # Record history
        score_history_sdf.append(current_score)
        joint_history_sdf.append(q_current.copy())

        # 2. Compute SDF-related quantities
        sdf_distance, closest_point, _ = sdf_field.compute_sdf(hand_current_global)
        sdf_gradient = sdf_field.compute_gradient(hand_current_global, delta=1e-5)
        tangent_direction = sdf_field.compute_tangent_direction(hand_current_global)

        sdf_values.append(sdf_distance)

        # 3. Compute goal direction
        goal_direction = task_goal_global - hand_current_global
        goal_distance = np.linalg.norm(goal_direction)

        # Check if reached goal
        if goal_distance < goal_threshold:
            print(f"Reached target at iteration {step}")
            break

        # ========== ADAPTIVE WEIGHT COMPUTATION ==========
        # Compute interpolation factor based on SDF distance
        if sdf_distance <= sdf_near_threshold:
            # NEAR: Use near weights
            blend_factor = 0.0
        elif sdf_distance >= sdf_far_threshold:
            # FAR: Use far weights
            blend_factor = 1.0
        else:
            # TRANSITION: Smooth interpolation
            blend_factor = (sdf_distance - sdf_near_threshold) / (sdf_far_threshold - sdf_near_threshold)

        # Interpolate weights smoothly
        alpha_sdf = alpha_near + blend_factor * (alpha_far - alpha_near)
        beta_tangent = beta_near + blend_factor * (beta_far - beta_near)
        gamma_goal = gamma_near + blend_factor * (gamma_far - gamma_near)

        # Adaptive step size based on SDF distance
        step_size = step_size_near + blend_factor * (step_size_far - step_size_near)

        # Store weights for analysis
        adaptive_weights_history.append({
            'sdf_distance': sdf_distance,
            'alpha': alpha_sdf,
            'beta': beta_tangent,
            'gamma': gamma_goal,
            'blend_factor': blend_factor,
            'step_size': step_size
        })

        # Normalize directions
        if np.linalg.norm(sdf_gradient) > 1e-6:
            sdf_gradient_normalized = -sdf_gradient / np.linalg.norm(sdf_gradient)
        else:
            sdf_gradient_normalized = np.zeros(3)

        if np.linalg.norm(tangent_direction) > 1e-6:
            tangent_normalized = tangent_direction / np.linalg.norm(tangent_direction)
        else:
            tangent_normalized = np.zeros(3)

        if goal_distance > 1e-6:
            goal_direction_normalized = goal_direction / goal_distance
        else:
            goal_direction_normalized = np.zeros(3)

        # 4. Combine directions with adaptive weights
        combined_direction_task = (alpha_sdf * sdf_gradient_normalized +
                                   beta_tangent * tangent_normalized +
                                   gamma_goal * goal_direction_normalized)

        combined_norm = np.linalg.norm(combined_direction_task)
        if combined_norm > 1e-6:
            combined_direction_task_normalized = combined_direction_task / combined_norm
        else:
            combined_direction_task_normalized = goal_direction_normalized

        # 5. Adaptive step size with goal proximity consideration
        progress_ratio = 1.0 - (goal_distance / np.linalg.norm(task_goal_global - trajectory_hand_sdf[0]))
        adaptive_step = step_size * (1.0 - 0.4 * progress_ratio)
        adaptive_step = min(adaptive_step, goal_distance * 0.35)

        # 6. Task space displacement
        delta_hand_global = adaptive_step * combined_direction_task_normalized
        hand_target_global = hand_current_global + delta_hand_global

        # 7. Convert to shoulder frame for IK
        hand_target_relative = hand_target_global - shoulder
        hand_target_shoulder = np.array([
            -hand_target_relative[1],
            -hand_target_relative[0],
            hand_target_relative[2]
        ])

        # 8. Solve IK with smoothness constraints
        def ik_objective(q):
            _, hand_shoulder = mos.forward_kinematics(q, d_uar, d_lar)
            position_error = np.linalg.norm(hand_shoulder - hand_target_shoulder)

            if len(trajectory) > 0:
                q_prev = trajectory[-1]
                joint_velocity = q - q_prev
                velocity_penalty = joint_velocity_weight * np.sum(joint_velocity ** 2)

                if len(trajectory) > 1:
                    q_prev_prev = trajectory[-2]
                    prev_velocity = q_prev - q_prev_prev
                    acceleration = joint_velocity - prev_velocity
                    acceleration_penalty = joint_acceleration_weight * np.sum(acceleration ** 2)
                else:
                    acceleration_penalty = 0.0

                return position_error + velocity_penalty + acceleration_penalty

            return position_error

        result = minimize(
            ik_objective,
            q_current,
            method='SLSQP',
            bounds=joint_angle_bounds,
            options={'maxiter': 150, 'ftol': 1e-9}
        )

        q_next = result.x

        # 9. Apply moving average filter
        if use_moving_average and len(trajectory) >= moving_avg_window:
            recent_qs = np.array(trajectory[-(moving_avg_window - 1):] + [q_next])
            q_smoothed = np.mean(recent_qs, axis=0)

            _, hand_check_shoulder = mos.forward_kinematics(q_smoothed, d_uar, d_lar)
            check_error = np.linalg.norm(hand_check_shoulder - hand_target_shoulder)

            if check_error < 0.02:
                q_next = q_smoothed

        # 10. Update trajectory
        trajectory.append(q_next.copy())

        new_elbow_shoulder, new_hand_shoulder = mos.forward_kinematics(q_next, d_uar, d_lar)
        new_hand_global = trans_shoulder2global(new_hand_shoulder, shoulder, arm='right')
        new_elbow_global = trans_shoulder2global(new_elbow_shoulder, shoulder, arm='right')

        trajectory_hand_sdf.append(new_hand_global.copy())
        trajectory_elbow_sdf.append(new_elbow_global.copy())

        # Track velocities and accelerations
        if len(trajectory) > 1:
            velocity = q_next - trajectory[-2]
            joint_velocities.append(np.linalg.norm(velocity))

            if len(trajectory) > 2:
                prev_velocity = trajectory[-2] - trajectory[-3]
                acceleration = velocity - prev_velocity
                joint_accelerations.append(np.linalg.norm(acceleration))

        # 11. Print progress with adaptive weights
        if step % 5 == 0 or step == num_iterations - 1:
            avg_velocity = np.mean(joint_velocities[-5:]) if joint_velocities else 0
            print(f"Iter {step:3d}: Score={current_score:.4f}, "
                  f"SDF={sdf_distance * 1000:.1f}mm, "
                  f"Goal={goal_distance * 1000:.1f}mm, "
                  f"α={alpha_sdf:.2f}, β={beta_tangent:.2f}, "
                  f"Step={adaptive_step:.3f}")

        # 12. Update current state
        q_current = q_next.copy()

    # ========== POST-PROCESSING: MULTI-STAGE SMOOTHING ==========
    print("\nApplying multi-stage trajectory smoothing...")

    # Stage 1: Joint space smoothing
    trajectory_array = np.array(trajectory)
    if len(trajectory_array) > 5:
        print("  Stage 1: Joint space smoothing...")
        trajectory_smoothed = smooth_trajectory(
            trajectory_array,
            smoothing_factor=0.4,
            iterations=3
        )
        trajectory = trajectory_smoothed.tolist()

        # Recompute task space from smoothed joint space
        trajectory_hand_sdf = []
        for q in trajectory_smoothed:
            _, hand_shoulder = mos.forward_kinematics(q, d_uar, d_lar)
            hand_global = trans_shoulder2global(hand_shoulder, shoulder, arm='right')
            trajectory_hand_sdf.append(hand_global)

    # Stage 2: Light task space smoothing
    trajectory_hand_sdf_array = np.array(trajectory_hand_sdf)
    if len(trajectory_hand_sdf_array) > 5:
        print("  Stage 2: Task space smoothing...")
        trajectory_hand_sdf_smoothed = smooth_trajectory(
            trajectory_hand_sdf_array,
            smoothing_factor=0.3,
            iterations=2
        )
        trajectory_hand_sdf = trajectory_hand_sdf_smoothed

    # Final verification
    q_final = trajectory[-1] if isinstance(trajectory[-1], np.ndarray) else np.array(trajectory[-1])
    _, final_hand_shoulder = mos.forward_kinematics(q_final, d_uar, d_lar)
    final_hand_global = trans_shoulder2global(final_hand_shoulder, shoulder, arm='right')
    final_score = utils.calculate_upper_limb_score_with_joint_angles(q_final)
    final_error = np.linalg.norm(final_hand_global - task_goal_global)

    # Compute smoothness metrics
    avg_velocity = np.mean(joint_velocities) if joint_velocities else 0
    avg_acceleration = np.mean(joint_accelerations) if joint_accelerations else 0

    # Analyze adaptive weight behavior
    sdf_distances = [w['sdf_distance'] for w in adaptive_weights_history]
    alphas = [w['alpha'] for w in adaptive_weights_history]
    betas = [w['beta'] for w in adaptive_weights_history]

    print(f"\nSDF-guided planning result (Adaptive Weights):")
    print(f"  Final joint angles (degrees): {np.rad2deg(q_final)}")
    print(f"  Final Score: {final_score:.4f}")
    print(f"  Final task error: {final_error * 1000:.2f} mm")
    print(f"  Trajectory points: {len(trajectory)}")
    print(f"  Average SDF distance: {np.mean(sdf_values) * 1000:.2f} mm")
    print(f"  Min/Max SDF distance: {np.min(sdf_values) * 1000:.2f} / {np.max(sdf_values) * 1000:.2f} mm")
    print(f"  Average joint velocity: {avg_velocity:.4f} rad/iter")
    print(f"  Average joint acceleration: {avg_acceleration:.4f} rad/iter²")
    print(f"\nAdaptive Weight Statistics:")
    print(f"  α (gradient) range: [{np.min(alphas):.2f}, {np.max(alphas):.2f}]")
    print(f"  β (tangent) range: [{np.min(betas):.2f}, {np.max(betas):.2f}]")
    print(f"  Average α: {np.mean(alphas):.2f}, Average β: {np.mean(betas):.2f}")
    print("=" * 70 + "\n")

    return trajectory, np.array(trajectory_hand_sdf), score_history_sdf, joint_history_sdf, sdf_values


def run_iterations_with_optimized_ik(num_iterations, task_goal_global, optimization_method='hybrid'):
    """
    Use optimization algorithm to solve IK, then plan trajectory in joint space

    Parameters:
    - num_iterations: number of iterations
    - task_goal_global: task space goal point
    - optimization_method: 'global', 'local', or 'hybrid' (recommended)
    """
    global current_q, global_positions, trajectory_hand, trajectory_elbow, score_history, joint_history

    print("\n" + "=" * 70)
    print("Starting trajectory planning based on optimized IK")
    print("=" * 70)

    # Step 1: Optimize to solve target joint configuration
    if optimization_method == 'hybrid':
        target_q, target_hand_global, target_score, position_error = \
            find_optimal_ik_solution_hybrid(task_goal_global, shoulder, d_uar, d_lar, joint_angle_bounds)
    else:
        target_q, target_hand_global, target_score, position_error = \
            find_optimal_ik_solution(task_goal_global, shoulder, d_uar, d_lar, joint_angle_bounds,
                                     method=optimization_method)

    # Verify current state
    _, hand_current_shoulder = mos.forward_kinematics(current_q, d_uar, d_lar)
    hand_current_global = trans_shoulder2global(hand_current_shoulder, shoulder, arm='right')
    current_score = utils.calculate_upper_limb_score_with_joint_angles(current_q)

    print(f"Current state:")
    print(f"  Joint angles (degrees): {np.rad2deg(current_q)}")
    print(f"  Ergonomic score: {current_score:.4f}")
    print(f"  Wrist position: {hand_current_global}")

    initial_distance = np.linalg.norm(target_hand_global - hand_current_global)
    print(f"\nPlanning parameters:")
    print(f"  Initial distance: {initial_distance:.4f} m")
    print(f"  Score improvement: {current_score:.4f} -> {target_score:.4f}")
    print(f"  Joint space distance: {np.linalg.norm(target_q - current_q):.4f} rad")

    # Step 2: CSEF-guided trajectory planning in joint space
    print("\nStarting joint space trajectory planning...")
    trajectory_result = run_iterations_in_joint_space(num_iterations, target_q, task_goal_global)

    return trajectory_result, target_q, target_hand_global


def run_iterations_in_joint_space(num_iterations, target_q, task_goal_global=None):
    """
    CSEF-guided trajectory planning in joint space with tighter convergence criteria

    Parameters:
    - num_iterations: number of iterations
    - target_q: target joint configuration
    - task_goal_global: task space goal (for visualization only)
    """
    global current_q, global_positions, trajectory_hand, trajectory_elbow, score_history, joint_history

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

    # Main loop
    trajectory = []
    q_current = current_q.copy()
    trajectory.append(q_current.copy())

    print(f"Joint space planning: from current_q to target_q")
    print(f"Initial joint distance: {np.linalg.norm(target_q - q_current):.4f} rad")
    print(f"Initial joint angles: {np.rad2deg(q_current)}")
    print(f"Target joint angles: {np.rad2deg(target_q)}\n")

    for step in range(num_iterations):
        # 1. Calculate current state
        elbow_current_shoulder, hand_current_shoulder = mos.forward_kinematics(q_current, d_uar, d_lar)
        hand_current_global = trans_shoulder2global(hand_current_shoulder, shoulder, arm='right')
        elbow_current_global = trans_shoulder2global(elbow_current_shoulder, shoulder, arm='right')
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

            # Update global state
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


def run_iterations_with_hybrid_guidance(num_iterations, task_goal_global, reference_trajectory):
    """
    Hybrid SDF-SEF trajectory planning method (Method 3)

    Key Innovation:
    - Find nearest point on reference trajectory in task space
    - Map it to joint space as intermediate target
    - Combine three guidances in joint space:
        1. SEF gradient (ergonomic optimization)
        2. SDF guidance (trajectory tracking)
        3. Goal attraction (task completion)
    """
    global current_q, global_positions, shoulder, d_uar, d_lar, joint_angle_bounds

    print("\n" + "=" * 70)
    print("Starting HYBRID SDF-SEF trajectory planning (Method 3)")
    print("=" * 70)

    # Initialize SDF field
    sdf_field = TrajectorySDFField(reference_trajectory, sigma=0.05)

    # ========== HYBRID METHOD PARAMETERS ==========
    # Three-way guidance weights
    alpha_sef = 0.3  # SEF (ergonomic) weight
    alpha_sdf = 0.4  # SDF (trajectory tracking) weight
    alpha_goal = 0.3  # Goal attraction weight

    # Adaptive weight parameters
    sdf_near_threshold = 0.01  # 1cm - near trajectory
    sdf_far_threshold = 0.05  # 5cm - far from trajectory

    # When near trajectory, reduce SDF weight, increase SEF weight
    weight_adjust_factor = 0.5  # How much to adjust weights

    # Step size parameters
    step_size_base = 0.05
    step_size_near = 0.04
    step_size_far = 0.06

    # Smoothness parameters
    joint_velocity_weight = 0.3
    joint_acceleration_weight = 0.2
    use_moving_average = True
    moving_avg_window = 3

    # Convergence threshold
    goal_threshold = 0.01  # 10mm

    # SEF parameters
    q_opt = np.array(optimal_q)
    comfort_threshold = 0.1

    # Storage
    trajectory = []
    trajectory_hand_hybrid = []
    trajectory_elbow_hybrid = []
    score_history_hybrid = []
    joint_history_hybrid = []
    sdf_values = []
    intermediate_targets = []  # Store intermediate joint targets
    guidance_weights_history = []  # Track weight evolution

    joint_velocities = []
    joint_accelerations = []

    q_current = current_q.copy()
    trajectory.append(q_current.copy())

    # Initial state
    _, hand_current_shoulder = mos.forward_kinematics(q_current, d_uar, d_lar)
    hand_current_global = trans_shoulder2global(hand_current_shoulder, shoulder, arm='right')
    trajectory_hand_hybrid.append(hand_current_global.copy())

    print(f"Initial position: {hand_current_global}")
    print(f"Target position: {task_goal_global}")
    print(f"Initial distance: {np.linalg.norm(task_goal_global - hand_current_global):.4f} m")
    print(f"\nHybrid Guidance Strategy:")
    print(f"  SEF weight: {alpha_sef:.2f} (ergonomic optimization)")
    print(f"  SDF weight: {alpha_sdf:.2f} (trajectory tracking)")
    print(f"  Goal weight: {alpha_goal:.2f} (task completion)")
    print(f"  Adaptive adjustment based on SDF distance\n")

    # Helper functions for SEF
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

    for step in range(num_iterations):
        # 1. Compute current state in task space
        elbow_current_shoulder, hand_current_shoulder = mos.forward_kinematics(q_current, d_uar, d_lar)
        hand_current_global = trans_shoulder2global(hand_current_shoulder, shoulder, arm='right')
        elbow_current_global = trans_shoulder2global(elbow_current_shoulder, shoulder, arm='right')
        current_score = utils.calculate_upper_limb_score_with_joint_angles(q_current)

        # Record history
        score_history_hybrid.append(current_score)
        joint_history_hybrid.append(q_current.copy())

        # 2. Find nearest point on reference trajectory in TASK SPACE
        sdf_distance, closest_point_task, closest_idx = sdf_field.compute_sdf(hand_current_global)
        sdf_values.append(sdf_distance)

        # 3. Map closest point to JOINT SPACE as intermediate target
        # Convert closest point to shoulder frame
        closest_point_relative = closest_point_task - shoulder
        closest_point_shoulder = np.array([
            -closest_point_relative[1],
            -closest_point_relative[0],
            closest_point_relative[2]
        ])

        # Solve IK for intermediate target (with looser tolerance)
        def ik_intermediate_objective(q):
            _, hand_shoulder = mos.forward_kinematics(q, d_uar, d_lar)
            position_error = np.linalg.norm(hand_shoulder - closest_point_shoulder)

            # Add small ergonomic penalty to prefer comfortable configurations
            ergo_penalty = 0.01 * utils.calculate_upper_limb_score_with_joint_angles(q)

            return position_error + ergo_penalty

        result_ik = minimize(
            ik_intermediate_objective,
            q_current,
            method='SLSQP',
            bounds=joint_angle_bounds,
            options={'maxiter': 100, 'ftol': 1e-6}
        )

        q_intermediate = result_ik.x
        intermediate_targets.append(q_intermediate.copy())

        # 4. Compute final goal in joint space (if not already computed)
        goal_direction_task = task_goal_global - hand_current_global
        goal_distance_task = np.linalg.norm(goal_direction_task)

        # Check if reached goal
        if goal_distance_task < goal_threshold:
            print(f"Reached target at iteration {step}")
            break

        # Map final goal to joint space (use cached result or compute)
        task_goal_relative = task_goal_global - shoulder
        task_goal_shoulder = np.array([
            -task_goal_relative[1],
            -task_goal_relative[0],
            task_goal_relative[2]
        ])

        def ik_goal_objective(q):
            _, hand_shoulder = mos.forward_kinematics(q, d_uar, d_lar)
            return np.linalg.norm(hand_shoulder - task_goal_shoulder)

        result_goal = minimize(
            ik_goal_objective,
            q_current,
            method='SLSQP',
            bounds=joint_angle_bounds,
            options={'maxiter': 100, 'ftol': 1e-6}
        )

        q_goal = result_goal.x

        # ========== COMPUTE THREE GUIDANCE DIRECTIONS IN JOINT SPACE ==========

        # Direction 1: SEF gradient (ergonomic optimization)
        sef_gradient = calculate_sef_gradient(q_current)
        sef_gradient_norm = np.linalg.norm(sef_gradient)

        if sef_gradient_norm > 1e-6:
            sef_direction = -sef_gradient / sef_gradient_norm  # Negative to minimize SEF
        else:
            sef_direction = np.zeros(4)

        # Direction 2: SDF guidance (toward intermediate target on trajectory)
        sdf_direction_joint = q_intermediate - q_current
        sdf_direction_joint_norm = np.linalg.norm(sdf_direction_joint)

        if sdf_direction_joint_norm > 1e-6:
            sdf_direction_joint = sdf_direction_joint / sdf_direction_joint_norm
        else:
            sdf_direction_joint = np.zeros(4)

        # Direction 3: Goal attraction (toward final goal)
        goal_direction_joint = q_goal - q_current
        goal_direction_joint_norm = np.linalg.norm(goal_direction_joint)

        if goal_direction_joint_norm > 1e-6:
            goal_direction_joint = goal_direction_joint / goal_direction_joint_norm
        else:
            goal_direction_joint = np.zeros(4)

        # ========== ADAPTIVE WEIGHT ADJUSTMENT ==========
        # When near trajectory: increase SEF weight, decrease SDF weight
        # When far from trajectory: increase SDF weight, decrease SEF weight

        if sdf_distance <= sdf_near_threshold:
            # NEAR: prioritize ergonomics
            blend_factor = 0.0
        elif sdf_distance >= sdf_far_threshold:
            # FAR: prioritize trajectory tracking
            blend_factor = 1.0
        else:
            # TRANSITION: smooth interpolation
            blend_factor = (sdf_distance - sdf_near_threshold) / (sdf_far_threshold - sdf_near_threshold)

        # Adjust weights adaptively
        alpha_sef_adaptive = alpha_sef * (1.0 + weight_adjust_factor * (1.0 - blend_factor))
        alpha_sdf_adaptive = alpha_sdf * (1.0 + weight_adjust_factor * blend_factor)
        alpha_goal_adaptive = alpha_goal  # Keep goal weight constant

        # Normalize weights
        total_weight = alpha_sef_adaptive + alpha_sdf_adaptive + alpha_goal_adaptive
        alpha_sef_adaptive /= total_weight
        alpha_sdf_adaptive /= total_weight
        alpha_goal_adaptive /= total_weight

        # Store weights for analysis
        guidance_weights_history.append({
            'sdf_distance': sdf_distance,
            'alpha_sef': alpha_sef_adaptive,
            'alpha_sdf': alpha_sdf_adaptive,
            'alpha_goal': alpha_goal_adaptive,
            'blend_factor': blend_factor
        })

        # ========== COMBINE THREE DIRECTIONS ==========
        combined_direction = (alpha_sef_adaptive * sef_direction +
                              alpha_sdf_adaptive * sdf_direction_joint +
                              alpha_goal_adaptive * goal_direction_joint)

        combined_norm = np.linalg.norm(combined_direction)
        if combined_norm > 1e-6:
            combined_direction_normalized = combined_direction / combined_norm
        else:
            combined_direction_normalized = goal_direction_joint

        # ========== ADAPTIVE STEP SIZE ==========
        # Base step size on SDF distance and goal proximity
        step_size = step_size_near + blend_factor * (step_size_far - step_size_near)

        # Reduce step size when approaching goal
        progress_ratio = 1.0 - (goal_distance_task / np.linalg.norm(task_goal_global - trajectory_hand_hybrid[0]))
        adaptive_step = step_size * (1.0 - 0.4 * progress_ratio)
        adaptive_step = min(adaptive_step, goal_direction_joint_norm * 0.5)

        # ========== UPDATE JOINT ANGLES ==========
        q_next = q_current + adaptive_step * combined_direction_normalized

        # Enforce joint limits
        q_next = enforce_joint_limits(q_next, joint_angle_bounds)

        # ========== APPLY MOVING AVERAGE FILTER ==========
        if use_moving_average and len(trajectory) >= moving_avg_window:
            recent_qs = np.array(trajectory[-(moving_avg_window - 1):] + [q_next])
            q_smoothed = np.mean(recent_qs, axis=0)

            # Verify smoothed configuration still reaches near target
            _, hand_check_shoulder = mos.forward_kinematics(q_smoothed, d_uar, d_lar)
            hand_check_global = trans_shoulder2global(hand_check_shoulder, shoulder, arm='right')
            check_error = np.linalg.norm(hand_check_global - (hand_current_global + adaptive_step * 0.5))

            if check_error < 0.03:  # 3cm tolerance
                q_next = q_smoothed

        # ========== UPDATE TRAJECTORY ==========
        trajectory.append(q_next.copy())

        new_elbow_shoulder, new_hand_shoulder = mos.forward_kinematics(q_next, d_uar, d_lar)
        new_hand_global = trans_shoulder2global(new_hand_shoulder, shoulder, arm='right')
        new_elbow_global = trans_shoulder2global(new_elbow_shoulder, shoulder, arm='right')

        trajectory_hand_hybrid.append(new_hand_global.copy())
        trajectory_elbow_hybrid.append(new_elbow_global.copy())

        # Track velocities and accelerations
        if len(trajectory) > 1:
            velocity = q_next - trajectory[-2]
            joint_velocities.append(np.linalg.norm(velocity))

            if len(trajectory) > 2:
                prev_velocity = trajectory[-2] - trajectory[-3]
                acceleration = velocity - prev_velocity
                joint_accelerations.append(np.linalg.norm(acceleration))

        # ========== PRINT PROGRESS ==========
        if step % 5 == 0 or step == num_iterations - 1:
            print(f"Iter {step:3d}: Score={current_score:.4f}, "
                  f"SDF={sdf_distance * 1000:.1f}mm, "
                  f"Goal={goal_distance_task * 1000:.1f}mm, "
                  f"αSEF={alpha_sef_adaptive:.2f}, "
                  f"αSDF={alpha_sdf_adaptive:.2f}, "
                  f"αGoal={alpha_goal_adaptive:.2f}")

        # Update current state
        q_current = q_next.copy()

    # ========== POST-PROCESSING: MULTI-STAGE SMOOTHING ==========
    print("\nApplying multi-stage trajectory smoothing...")

    # Stage 1: Joint space smoothing
    trajectory_array = np.array(trajectory)
    if len(trajectory_array) > 5:
        print("  Stage 1: Joint space smoothing...")
        trajectory_smoothed = smooth_trajectory(
            trajectory_array,
            smoothing_factor=0.4,
            iterations=3
        )
        trajectory = trajectory_smoothed.tolist()

        # Recompute task space from smoothed joint space
        trajectory_hand_hybrid = []
        for q in trajectory_smoothed:
            _, hand_shoulder = mos.forward_kinematics(q, d_uar, d_lar)
            hand_global = trans_shoulder2global(hand_shoulder, shoulder, arm='right')
            trajectory_hand_hybrid.append(hand_global)

    # Stage 2: Light task space smoothing
    trajectory_hand_hybrid_array = np.array(trajectory_hand_hybrid)
    if len(trajectory_hand_hybrid_array) > 5:
        print("  Stage 2: Task space smoothing...")
        trajectory_hand_hybrid_smoothed = smooth_trajectory(
            trajectory_hand_hybrid_array,
            smoothing_factor=0.3,
            iterations=2
        )
        trajectory_hand_hybrid = trajectory_hand_hybrid_smoothed

    # Final verification
    q_final = trajectory[-1] if isinstance(trajectory[-1], np.ndarray) else np.array(trajectory[-1])
    _, final_hand_shoulder = mos.forward_kinematics(q_final, d_uar, d_lar)
    final_hand_global = trans_shoulder2global(final_hand_shoulder, shoulder, arm='right')
    final_score = utils.calculate_upper_limb_score_with_joint_angles(q_final)
    final_error = np.linalg.norm(final_hand_global - task_goal_global)

    # Compute smoothness metrics
    avg_velocity = np.mean(joint_velocities) if joint_velocities else 0
    avg_acceleration = np.mean(joint_accelerations) if joint_accelerations else 0

    # Analyze guidance weight behavior
    alphas_sef = [w['alpha_sef'] for w in guidance_weights_history]
    alphas_sdf = [w['alpha_sdf'] for w in guidance_weights_history]
    alphas_goal = [w['alpha_goal'] for w in guidance_weights_history]

    print(f"\nHybrid SDF-SEF planning result:")
    print(f"  Final joint angles (degrees): {np.rad2deg(q_final)}")
    print(f"  Final Score: {final_score:.4f}")
    print(f"  Final task error: {final_error * 1000:.2f} mm")
    print(f"  Trajectory points: {len(trajectory)}")
    print(f"  Average SDF distance: {np.mean(sdf_values) * 1000:.2f} mm")
    print(f"  Average joint velocity: {avg_velocity:.4f} rad/iter")
    print(f"  Average joint acceleration: {avg_acceleration:.4f} rad/iter²")
    print(f"\nGuidance Weight Statistics:")
    print(f"  αSEF range: [{np.min(alphas_sef):.2f}, {np.max(alphas_sef):.2f}], avg: {np.mean(alphas_sef):.2f}")
    print(f"  αSDF range: [{np.min(alphas_sdf):.2f}, {np.max(alphas_sdf):.2f}], avg: {np.mean(alphas_sdf):.2f}")
    print(f"  αGoal range: [{np.min(alphas_goal):.2f}, {np.max(alphas_goal):.2f}], avg: {np.mean(alphas_goal):.2f}")
    print("=" * 70 + "\n")

    return (trajectory, np.array(trajectory_hand_hybrid), score_history_hybrid,
            joint_history_hybrid, sdf_values, guidance_weights_history)


def compare_three_methods(trajectory_hand_ergo, score_history_ergo, joint_history_ergo,
                          trajectory_hand_sdf, score_history_sdf, joint_history_sdf, sdf_values_sdf,
                          trajectory_hand_hybrid, score_history_hybrid, joint_history_hybrid, sdf_values_hybrid,
                          weights_history_hybrid,
                          reference_trajectory, task_goal_global):
    """
    Compare results of THREE planning methods with comprehensive visualization
    """

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

    utils.plot_skeleton(ax1, global_positions, skeleton_parent_indices, color='gray')

    ax1.scatter(shoulder[0], shoulder[1], shoulder[2],
                c='black', s=120, label='Shoulder', marker='o', edgecolors='white', linewidth=1.5)
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

    utils.plot_skeleton(ax2, global_positions, skeleton_parent_indices, color='gray')

    ax2.scatter(shoulder[0], shoulder[1], shoulder[2],
                c='black', s=120, label='Shoulder', marker='o', edgecolors='white', linewidth=1.5)

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

    utils.plot_skeleton(ax3, global_positions, skeleton_parent_indices, color='gray')

    ax3.scatter(shoulder[0], shoulder[1], shoulder[2],
                c='black', s=120, label='Shoulder', marker='o', edgecolors='white', linewidth=1.5)
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

    # ============== Figure 2: Score Evolution and Statistics ==============
    fig2 = plt.figure(figsize=(20, 12))

    # Subplot 1: Ergonomic Score Comparison
    ax1 = fig2.add_subplot(3, 4, 1)
    iterations_ergo = range(1, len(score_history_ergo) + 1)
    iterations_sdf = range(1, len(score_history_sdf) + 1)
    iterations_hybrid = range(1, len(score_history_hybrid) + 1)

    ax1.plot(iterations_ergo, score_history_ergo, linewidth=2.5, color='steelblue',
             label='Method 1: Ergonomic', alpha=0.8)
    ax1.plot(iterations_sdf, score_history_sdf, linewidth=2.5, color='coral',
             label='Method 2: SDF', alpha=0.8)
    ax1.plot(iterations_hybrid, score_history_hybrid, linewidth=2.5, color='purple',
             label='Method 3: Hybrid', alpha=0.8)
    ax1.axhline(y=score_history_ergo[0], color='gray', linestyle=':', alpha=0.5,
                label=f'Initial: {score_history_ergo[0]:.3f}')

    ax1.set_xlabel('Iteration', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Ergonomic Score', fontsize=11, fontweight='bold')
    ax1.set_title('Ergonomic Score Evolution', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Subplot 2: SDF Distance (Methods 2 & 3)
    ax2 = fig2.add_subplot(3, 4, 2)
    ax2.plot(iterations_sdf, np.array(sdf_values_sdf) * 1000, linewidth=2.5,
             color='coral', alpha=0.8, label='Method 2: SDF')
    ax2.plot(iterations_hybrid, np.array(sdf_values_hybrid) * 1000, linewidth=2.5,
             color='purple', alpha=0.8, label='Method 3: Hybrid')
    ax2.set_xlabel('Iteration', fontsize=11, fontweight='bold')
    ax2.set_ylabel('SDF Distance (mm)', fontsize=11, fontweight='bold')
    ax2.set_title('Distance to Reference Trajectory', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # Subplot 3: Goal Distance
    ax3 = fig2.add_subplot(3, 4, 3)
    goal_dist_ergo = [np.linalg.norm(p - task_goal_global) * 1000 for p in traj_ergo]
    goal_dist_sdf = [np.linalg.norm(p - task_goal_global) * 1000 for p in traj_sdf]
    goal_dist_hybrid = [np.linalg.norm(p - task_goal_global) * 1000 for p in traj_hybrid]

    ax3.plot(range(len(goal_dist_ergo)), goal_dist_ergo, linewidth=2.5,
             color='steelblue', label='Method 1', alpha=0.8)
    ax3.plot(range(len(goal_dist_sdf)), goal_dist_sdf, linewidth=2.5,
             color='coral', label='Method 2', alpha=0.8)
    ax3.plot(range(len(goal_dist_hybrid)), goal_dist_hybrid, linewidth=2.5,
             color='purple', label='Method 3', alpha=0.8)
    ax3.set_xlabel('Trajectory Point', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Distance to Goal (mm)', fontsize=11, fontweight='bold')
    ax3.set_title('Goal Convergence', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    # Subplot 4: Hybrid Method Weight Evolution
    ax4 = fig2.add_subplot(3, 4, 4)
    if weights_history_hybrid:
        iters = range(len(weights_history_hybrid))
        alphas_sef = [w['alpha_sef'] for w in weights_history_hybrid]
        alphas_sdf = [w['alpha_sdf'] for w in weights_history_hybrid]
        alphas_goal = [w['alpha_goal'] for w in weights_history_hybrid]

        ax4.plot(iters, alphas_sef, linewidth=2, color='green', label='αSEF (Ergonomic)', alpha=0.8)
        ax4.plot(iters, alphas_sdf, linewidth=2, color='blue', label='αSDF (Trajectory)', alpha=0.8)
        ax4.plot(iters, alphas_goal, linewidth=2, color='red', label='αGoal (Task)', alpha=0.8)
        ax4.set_xlabel('Iteration', fontsize=11, fontweight='bold')
        ax4.set_ylabel('Weight Value', fontsize=11, fontweight='bold')
        ax4.set_title('Method 3: Adaptive Weights', fontsize=12, fontweight='bold')
        ax4.legend(fontsize=9)
        ax4.grid(True, alpha=0.3)

    # Subplots 5-8: Joint Angle Evolution
    joint_labels = ['Shoulder Flexion', 'Shoulder Abduction', 'Elbow Flexion', 'Forearm Rotation']

    for i in range(4):
        ax = fig2.add_subplot(3, 4, 5 + i)

        joint_ergo = np.array([np.rad2deg(q[i]) for q in joint_history_ergo])
        joint_sdf = np.array([np.rad2deg(q[i]) for q in joint_history_sdf])
        joint_hybrid = np.array([np.rad2deg(q[i]) for q in joint_history_hybrid])

        ax.plot(iterations_ergo, joint_ergo, linewidth=2, color='steelblue',
                label='Method 1', alpha=0.8)
        ax.plot(iterations_sdf, joint_sdf, linewidth=2, color='coral',
                label='Method 2', alpha=0.8)
        ax.plot(iterations_hybrid, joint_hybrid, linewidth=2, color='purple',
                label='Method 3', alpha=0.8)

        ax.set_xlabel('Iteration', fontsize=10, fontweight='bold')
        ax.set_ylabel('Angle (deg)', fontsize=10, fontweight='bold')
        ax.set_title(joint_labels[i], fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Subplots 9-12: Statistical Comparisons
    methods = ['Method 1\nErgonomic', 'Method 2\nSDF', 'Method 3\nHybrid']
    colors = ['steelblue', 'coral', 'purple']

    # Subplot 9: Path Length
    ax9 = fig2.add_subplot(3, 4, 9)
    lengths = [path_length_ergo, path_length_sdf, path_length_hybrid]
    bars = ax9.bar(methods, lengths, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax9.set_ylabel('Path Length (m)', fontsize=10, fontweight='bold')
    ax9.set_title('Total Path Length', fontsize=11, fontweight='bold')
    ax9.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, lengths):
        height = bar.get_height()
        ax9.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9)

    # Subplot 10: Final Error
    ax10 = fig2.add_subplot(3, 4, 10)
    errors = [final_error_ergo * 1000, final_error_sdf * 1000, final_error_hybrid * 1000]
    bars = ax10.bar(methods, errors, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax10.set_ylabel('Error (mm)', fontsize=10, fontweight='bold')
    ax10.set_title('Final Position Error', fontsize=11, fontweight='bold')
    ax10.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, errors):
        height = bar.get_height()
        ax10.text(bar.get_x() + bar.get_width() / 2., height,
                  f'{val:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=9)

    # Subplot 11: Final Score
    ax11 = fig2.add_subplot(3, 4, 11)
    final_scores = [score_history_ergo[-1], score_history_sdf[-1], score_history_hybrid[-1]]
    bars = ax11.bar(methods, final_scores, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax11.set_ylabel('Score', fontsize=10, fontweight='bold')
    ax11.set_title('Final Ergonomic Score', fontsize=11, fontweight='bold')
    ax11.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, final_scores):
        height = bar.get_height()
        ax11.text(bar.get_x() + bar.get_width() / 2., height,
                  f'{val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9)

    # Subplot 12: Score Improvement
    ax12 = fig2.add_subplot(3, 4, 12)
    improvements = [score_history_ergo[0] - score_history_ergo[-1],
                    score_history_sdf[0] - score_history_sdf[-1],
                    score_history_hybrid[0] - score_history_hybrid[-1]]
    bars = ax12.bar(methods, improvements, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax12.set_ylabel('Improvement', fontsize=10, fontweight='bold')
    ax12.set_title('Score Improvement', fontsize=11, fontweight='bold')
    ax12.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, improvements):
        height = bar.get_height()
        ax12.text(bar.get_x() + bar.get_width() / 2., height,
                  f'{val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9)

    plt.tight_layout()
    plt.savefig('comparison_3methods_statistics.png', dpi=300, bbox_inches='tight')
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

    # Plot skeleton
    utils.plot_skeleton(ax, global_positions, skeleton_parent_indices, color='gray')

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

    # Plot arm skeleton
    ax.plot([shoulder[0], new_elbow[0], new_hand[0]],
            [shoulder[1], new_elbow[1], new_hand[1]],
            [shoulder[2], new_elbow[2], new_hand[2]],
            c='red', linewidth=4.5, label='Arm', alpha=0.9)

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
        utils.read_skeleton_motion('/home/lee/Ergo-Manip/data/demo_2_test_chenzui_only_optitrack2hotu.npy')
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

    # Define task goal
    # 定义任务目标
    task_goal_global = hand_current + np.array([0.1, 0.0, -0.2])

    print(f"\n{'=' * 70}")
    print(f"Starting THREE-METHOD COMPARATIVE trajectory planning study")
    print(f"Task goal (global): {task_goal_global}")
    print(f"{'=' * 70}\n")

    # ========== METHOD 1: Ergonomic Field Guidance ==========
    print("\n" + "=" * 70)
    print("RUNNING METHOD 1: ERGONOMIC FIELD GUIDANCE")
    print("=" * 70)

    current_q_backup = current_q.copy()
    trajectory_hand = [hand_current.copy()]
    trajectory_elbow = [elbow_current.copy()]
    score_history = []
    joint_history = []

    trajectory_result_ergo, target_q, target_hand = run_iterations_with_optimized_ik(
        num_iterations=num_iterations,
        task_goal_global=task_goal_global,
        optimization_method='hybrid'
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

    current_q = current_q_backup.copy()

    trajectory_result_sdf, trajectory_hand_sdf, score_history_sdf, joint_history_sdf, sdf_values = \
        run_iterations_with_sdf_guidance(
            num_iterations=num_iterations,
            task_goal_global=task_goal_global,
            reference_trajectory=reference_trajectory
        )

    print(f"\nMethod 2 completed: Final Score={score_history_sdf[-1]:.4f}, Points={len(trajectory_hand_sdf)}")

    # ========== METHOD 3: Hybrid SDF-SEF Guidance ==========
    print("\n" + "=" * 70)
    print("RUNNING METHOD 3: HYBRID SDF-SEF GUIDANCE")
    print("=" * 70)

    current_q = current_q_backup.copy()

    (trajectory_result_hybrid, trajectory_hand_hybrid, score_history_hybrid,
     joint_history_hybrid, sdf_values_hybrid, weights_history_hybrid) = \
        run_iterations_with_hybrid_guidance(
            num_iterations=num_iterations,
            task_goal_global=task_goal_global,
            reference_trajectory=reference_trajectory
        )

    print(f"\nMethod 3 completed: Final Score={score_history_hybrid[-1]:.4f}, Points={len(trajectory_hand_hybrid)}")

    # ========== COMPARISON AND VISUALIZATION ==========
    compare_three_methods(
        trajectory_hand_ergo, score_history_ergo, joint_history_ergo,
        trajectory_hand_sdf, score_history_sdf, joint_history_sdf, sdf_values,
        trajectory_hand_hybrid, score_history_hybrid, joint_history_hybrid, sdf_values_hybrid, weights_history_hybrid,
        reference_trajectory, task_goal_global
    )

    print("\n" + "=" * 70)
    print("THREE-METHOD COMPARATIVE STUDY COMPLETED!")
    print("=" * 70)