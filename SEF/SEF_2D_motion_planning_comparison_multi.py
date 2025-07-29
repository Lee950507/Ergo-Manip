import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import Circle
import matplotlib.gridspec as gridspec
from scipy.interpolate import griddata, splprep, splev
import time
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


def forward_kinematics(q1, q2, l1, l2):
    """Forward kinematics: Calculate end-effector position from joint angles"""
    x = l1 * np.cos(q1) + l2 * np.cos(q1 + q2)
    y = l1 * np.sin(q1) + l2 * np.sin(q1 + q2)
    return x, y


def inverse_kinematics(x, y, l1, l2):
    """
    Inverse kinematics: Calculate joint angles from end-effector position

    Returns:
    - ((q1_sol1, q2_sol1), (q1_sol2, q2_sol2)): Two possible solutions
    """
    r_squared = x ** 2 + y ** 2
    r = np.sqrt(r_squared)

    # Check if position is within workspace
    if r > l1 + l2 or r < abs(l1 - l2):
        return None

    # Calculate the second joint angle
    cos_q2 = (r_squared - l1 ** 2 - l2 ** 2) / (2 * l1 * l2)
    cos_q2 = np.clip(cos_q2, -1.0, 1.0)  # Handle numerical errors

    sin_q2_pos = np.sqrt(1 - cos_q2 ** 2)  # Elbow-up solution
    sin_q2_neg = -sin_q2_pos  # Elbow-down solution

    q2_sol1 = np.arctan2(sin_q2_pos, cos_q2)
    q2_sol2 = np.arctan2(sin_q2_neg, cos_q2)

    # Calculate the first joint angle
    k1 = l1 + l2 * cos_q2
    k2 = l2 * sin_q2_pos
    alpha = np.arctan2(y, x)
    gamma = np.arctan2(k2, k1)
    q1_sol1 = alpha - gamma

    k1 = l1 + l2 * cos_q2
    k2 = l2 * sin_q2_neg
    gamma = np.arctan2(k2, k1)
    q1_sol2 = alpha - gamma

    return ((q1_sol1, q2_sol1), (q1_sol2, q2_sol2))


def calculate_joint_sef(q1, q2, q1_opt, q2_opt, comfort_threshold, weights=None):
    """Calculate the SEF value in joint space"""
    if weights is None:
        weights = [1, 1]
    distance = np.sqrt(weights[0] * (q1 - q1_opt) ** 2 + weights[1] * (q2 - q2_opt) ** 2)
    return distance - comfort_threshold


def create_joint_sef_field(q1_min, q1_max, q2_min, q2_max, q1_opt, q2_opt,
                           comfort_threshold, weights, resolution=200):
    """Create the SEF field in joint space with higher resolution"""
    q1_vals = np.linspace(q1_min, q1_max, resolution)
    q2_vals = np.linspace(q2_min, q2_max, resolution)
    Q1, Q2 = np.meshgrid(q1_vals, q2_vals)

    # Calculate SEF value for each point (vectorized for efficiency)
    distance = np.sqrt(weights[0] * (Q1 - q1_opt) ** 2 + weights[1] * (Q2 - q2_opt) ** 2)
    SEF = distance - comfort_threshold

    return Q1, Q2, SEF


def create_cartesian_sef_field(l1, l2, q1_opt, q2_opt, comfort_threshold, weights=None, resolution=200):
    """
    Create the SEF field in Cartesian space with higher resolution
    """
    if weights is None:
        weights = [1, 1]

    # Workspace boundaries
    workspace_radius = l1 + l2
    x_min, x_max = -workspace_radius, workspace_radius
    y_min, y_max = -workspace_radius, workspace_radius

    # Create grid
    x_vals = np.linspace(x_min, x_max, resolution)
    y_vals = np.linspace(y_min, y_max, resolution)
    X, Y = np.meshgrid(x_vals, y_vals)

    # Initialize SEF values
    SEF = np.full_like(X, np.nan)

    # Calculate optimal configuration in Cartesian coordinates
    x_opt, y_opt = forward_kinematics(q1_opt, q2_opt, l1, l2)

    # Calculate SEF for each grid point
    for i in range(resolution):
        for j in range(resolution):
            x, y = X[i, j], Y[i, j]

            # Calculate inverse kinematics
            ik_solutions = inverse_kinematics(x, y, l1, l2)

            if ik_solutions is not None:
                # Calculate SEF values for both solutions
                sef_values = []
                for q1, q2 in ik_solutions:
                    # Calculate SEF in joint space
                    joint_sef = calculate_joint_sef(q1, q2, q1_opt, q2_opt, comfort_threshold, weights)
                    sef_values.append(joint_sef)

                # Choose the solution with minimum SEF value (most comfortable)
                SEF[i, j] = min(sef_values)

    return X, Y, SEF, (x_opt, y_opt)


def calculate_cartesian_gradient(x, y, X, Y, SEF):
    """
    Calculate the gradient of SEF in Cartesian space using central differencing
    for smoother gradients
    """
    # Use griddata for interpolation, handle NaN values
    valid_mask = ~np.isnan(SEF)
    if not np.any(valid_mask):
        return np.array([0, 0])  # If no valid data

    x_flat = X[valid_mask].flatten()
    y_flat = Y[valid_mask].flatten()
    sef_flat = SEF[valid_mask].flatten()

    # Use smaller delta for more accurate gradients
    delta = 0.02

    # Calculate SEF value at interpolation point using cubic interpolation for smoother results
    sef_center = griddata((x_flat, y_flat), sef_flat, (x, y), method='cubic', fill_value=np.nan)

    if np.isnan(sef_center):
        # If cubic interpolation fails, fall back to linear
        sef_center = griddata((x_flat, y_flat), sef_flat, (x, y), method='linear', fill_value=0)

    # Calculate x-direction gradient with central differencing
    sef_x_plus = griddata((x_flat, y_flat), sef_flat, (x + delta, y), method='cubic', fill_value=np.nan)
    sef_x_minus = griddata((x_flat, y_flat), sef_flat, (x - delta, y), method='cubic', fill_value=np.nan)

    # Handle NaN values in gradient calculation
    if np.isnan(sef_x_plus) or np.isnan(sef_x_minus):
        # Fall back to one-sided difference or zero gradient
        if not np.isnan(sef_x_plus):
            dsef_dx = (sef_x_plus - sef_center) / delta
        elif not np.isnan(sef_x_minus):
            dsef_dx = (sef_center - sef_x_minus) / delta
        else:
            dsef_dx = 0
    else:
        # Central difference for better accuracy
        dsef_dx = (sef_x_plus - sef_x_minus) / (2 * delta)

    # Calculate y-direction gradient with central differencing
    sef_y_plus = griddata((x_flat, y_flat), sef_flat, (x, y + delta), method='cubic', fill_value=np.nan)
    sef_y_minus = griddata((x_flat, y_flat), sef_flat, (x, y - delta), method='cubic', fill_value=np.nan)

    # Handle NaN values in gradient calculation
    if np.isnan(sef_y_plus) or np.isnan(sef_y_minus):
        # Fall back to one-sided difference or zero gradient
        if not np.isnan(sef_y_plus):
            dsef_dy = (sef_y_plus - sef_center) / delta
        elif not np.isnan(sef_y_minus):
            dsef_dy = (sef_center - sef_y_minus) / delta
        else:
            dsef_dy = 0
    else:
        # Central difference for better accuracy
        dsef_dy = (sef_y_plus - sef_y_minus) / (2 * delta)

    # Construct gradient vector
    gradient = np.array([dsef_dx, dsef_dy])

    # Normalize gradient
    gradient_norm = np.linalg.norm(gradient)
    if gradient_norm > 1e-10:
        gradient = gradient / gradient_norm

    return gradient


def smooth_path(path, smoothing_factor=0.3):
    """
    Smooth a path using spline interpolation to create a smoother trajectory
    """
    if len(path) < 4:
        return path  # Not enough points to smooth

    # Extract coordinates
    path_array = np.array(path)
    points = path_array.T

    # Create parameterization variable
    u = np.arange(len(path))

    # Create spline representation
    try:
        tck, u = splprep(points, u=u, s=smoothing_factor, k=3)

        # Generate more points for a smoother curve
        u_new = np.linspace(0, u[-1], len(path) * 2)
        smooth_points = np.array(splev(u_new, tck)).T

        # Ensure start and end points are preserved
        smooth_points[0] = path[0]
        smooth_points[-1] = path[-1]

        return smooth_points
    except:
        # If smoothing fails, return original path
        return path


def calculate_joint_space_path(start_q, end_q, q1_opt, q2_opt, comfort_threshold,
                               weights=None, step_size=0.03, max_steps=2000,
                               goal_weight=0.5, random_factor=0.02):
    """
    Calculate path in joint space based on SEF with improved termination and smoothing
    """
    if weights is None:
        weights = [1, 1]

    # Initialize path and current position
    current_q = np.array(start_q)
    path = [current_q.copy()]
    sef_values = [calculate_joint_sef(current_q[0], current_q[1], q1_opt, q2_opt, comfort_threshold, weights)]

    # Target precision - more precise termination condition
    target_precision = 0.01

    # Stagnation detection
    stagnation_count = 0
    stagnation_threshold = 50
    prev_distance = np.linalg.norm(np.array(end_q) - current_q)

    for step in range(max_steps):
        # Calculate vector to goal
        to_goal = np.array(end_q) - current_q
        distance_to_goal = np.linalg.norm(to_goal)

        # Check for stagnation
        if abs(distance_to_goal - prev_distance) < 0.001:
            stagnation_count += 1
        else:
            stagnation_count = 0
        prev_distance = distance_to_goal

        # If close enough to goal, terminate and add exact end point
        if distance_to_goal < target_precision:
            path.append(np.array(end_q))
            sef_values.append(calculate_joint_sef(end_q[0], end_q[1], q1_opt, q2_opt, comfort_threshold, weights))
            break

        # Adaptive parameters to improve smoothness and convergence
        adaptive_goal_weight = goal_weight
        current_sef = sef_values[-1]

        # Increase goal attraction when stuck
        if stagnation_count > stagnation_threshold:
            adaptive_goal_weight = min(0.9, goal_weight + 0.2)
            random_factor *= 1.5  # Increase randomness to escape local minima
            stagnation_count = 0  # Reset counter

        # Calculate goal attraction direction
        goal_direction = to_goal / (distance_to_goal + 1e-10)

        # Calculate SEF gradient
        dsef_dq1 = 2 * weights[0] * (current_q[0] - q1_opt)
        dsef_dq2 = 2 * weights[1] * (current_q[1] - q2_opt)
        sef_gradient = np.array([dsef_dq1, dsef_dq2])

        gradient_norm = np.linalg.norm(sef_gradient)
        if gradient_norm > 1e-10:
            sef_gradient /= gradient_norm

        # Mixed direction: goal attraction and negative SEF gradient
        mixed_direction = adaptive_goal_weight * goal_direction - (1 - adaptive_goal_weight) * sef_gradient

        # Add random noise to avoid local minima but keep it small for smoothness
        random_noise = random_factor * np.random.normal(0, 1, 2)
        mixed_direction += random_noise

        # Normalize direction
        direction_norm = np.linalg.norm(mixed_direction)
        if direction_norm > 0:
            mixed_direction = mixed_direction / direction_norm

        # Adaptive step size for smoother motion
        adaptive_step = step_size

        # Smaller steps when crossing comfort boundary for smoother transition
        next_q = current_q + adaptive_step * mixed_direction
        current_sef = calculate_joint_sef(current_q[0], current_q[1], q1_opt, q2_opt, comfort_threshold, weights)
        next_sef = calculate_joint_sef(next_q[0], next_q[1], q1_opt, q2_opt, comfort_threshold, weights)

        # If crossing comfort boundary, reduce step size
        if current_sef * next_sef < 0:
            adaptive_step *= 0.5

        # Update position
        current_q = current_q + adaptive_step * mixed_direction

        # Save path and SEF values
        path.append(current_q.copy())
        sef_values.append(calculate_joint_sef(current_q[0], current_q[1], q1_opt, q2_opt, comfort_threshold, weights))

    # Apply path smoothing for final trajectory
    smoothed_path = smooth_path(np.array(path), smoothing_factor=0.3)

    # Recalculate SEF values for smoothed path
    smoothed_sef_values = []
    for q in smoothed_path:
        smoothed_sef_values.append(calculate_joint_sef(q[0], q[1], q1_opt, q2_opt, comfort_threshold, weights))

    return np.array(smoothed_path), np.array(smoothed_sef_values)


def calculate_cartesian_space_path(start_pos, end_pos, start_q, end_q, X, Y, SEF, l1, l2, q1_opt, q2_opt,
                                   comfort_threshold, weights=None, step_size=0.03,
                                   max_steps=2000, goal_weight=0.5, random_factor=0.02):
    """
    Calculate path in Cartesian space based on SEF with improved termination and smoothing
    """
    if weights is None:
        weights = [1, 1]

    # Workspace boundary
    workspace_radius = l1 + l2

    # Initialize path and current position
    current_pos = np.array(start_pos)
    path = [current_pos.copy()]

    # Use the exact same joint configuration as in joint space planning
    current_joint = np.array(start_q)
    joint_configs = [current_joint.copy()]

    # Calculate initial SEF value
    sef_values = [calculate_joint_sef(current_joint[0], current_joint[1], q1_opt, q2_opt, comfort_threshold, weights)]

    # Target precision - more precise termination condition
    target_precision = 0.01

    # Stagnation detection
    stagnation_count = 0
    stagnation_threshold = 50
    prev_distance = np.linalg.norm(np.array(end_pos) - current_pos)

    for step in range(max_steps):
        # Calculate vector to goal
        to_goal = np.array(end_pos) - current_pos
        distance_to_goal = np.linalg.norm(to_goal)

        # Check for stagnation
        if abs(distance_to_goal - prev_distance) < 0.001:
            stagnation_count += 1
        else:
            stagnation_count = 0
        prev_distance = distance_to_goal

        # If close enough to goal, terminate and add exact end point
        if distance_to_goal < target_precision:
            path.append(np.array(end_pos))
            joint_configs.append(np.array(end_q))
            sef_values.append(calculate_joint_sef(end_q[0], end_q[1], q1_opt, q2_opt, comfort_threshold, weights))
            break

        # Adaptive parameters to improve smoothness and convergence
        adaptive_goal_weight = goal_weight
        current_sef = sef_values[-1]

        # Increase goal attraction when stuck or in high-sef regions
        if stagnation_count > stagnation_threshold or current_sef > 1.0:
            adaptive_goal_weight = min(0.9, goal_weight + 0.2)
            random_factor *= 1.5  # Increase randomness to escape local minima
            stagnation_count = 0  # Reset counter

        # Calculate goal attraction direction
        goal_direction = to_goal / (distance_to_goal + 1e-10)

        # Calculate SEF gradient using improved method
        sef_gradient = calculate_cartesian_gradient(current_pos[0], current_pos[1], X, Y, SEF)

        # Mixed direction: goal attraction and negative SEF gradient
        mixed_direction = adaptive_goal_weight * goal_direction - (1 - adaptive_goal_weight) * sef_gradient

        # Add random noise to avoid local minima but keep it small for smoothness
        random_noise = random_factor * np.random.normal(0, 1, 2)
        mixed_direction += random_noise

        # Normalize direction
        direction_norm = np.linalg.norm(mixed_direction)
        if direction_norm > 0:
            mixed_direction = mixed_direction / direction_norm

        # Adaptive step size for smoother motion
        adaptive_step = step_size

        # Calculate new position
        new_pos = current_pos + adaptive_step * mixed_direction

        # Ensure new position is within workspace
        new_pos_distance = np.linalg.norm(new_pos)
        if new_pos_distance > workspace_radius:
            # If outside range, project to workspace boundary
            new_pos = new_pos * (workspace_radius / new_pos_distance)

        if new_pos_distance < abs(l1 - l2):
            # If inside inner boundary, project to inner boundary
            new_pos = new_pos * (abs(l1 - l2) / new_pos_distance)

        # Check if new position has valid inverse kinematics
        ik_solutions = inverse_kinematics(new_pos[0], new_pos[1], l1, l2)

        if ik_solutions is not None:
            # Choose solution closest to current joint configuration
            best_index = 0
            min_dist = float('inf')

            for i, (q1, q2) in enumerate(ik_solutions):
                dist = np.linalg.norm(np.array([q1, q2]) - current_joint)
                if dist < min_dist:
                    min_dist = dist
                    best_index = i

            new_joint = np.array(ik_solutions[best_index])
            current_joint = new_joint
            current_pos = new_pos

            # Save path, joint configuration and SEF value
            path.append(current_pos.copy())
            joint_configs.append(current_joint.copy())

            new_sef = calculate_joint_sef(current_joint[0], current_joint[1], q1_opt, q2_opt, comfort_threshold,
                                          weights)
            sef_values.append(new_sef)
        else:
            # If no valid solution, try reducing step size
            for scale in [0.5, 0.25, 0.1]:
                test_pos = current_pos + scale * adaptive_step * mixed_direction
                if inverse_kinematics(test_pos[0], test_pos[1], l1, l2) is not None:
                    new_pos = test_pos
                    ik_solutions = inverse_kinematics(new_pos[0], new_pos[1], l1, l2)

                    # Choose solution closest to current joint configuration
                    best_index = 0
                    min_dist = float('inf')

                    for i, (q1, q2) in enumerate(ik_solutions):
                        dist = np.linalg.norm(np.array([q1, q2]) - current_joint)
                        if dist < min_dist:
                            min_dist = dist
                            best_index = i

                    new_joint = np.array(ik_solutions[best_index])
                    current_joint = new_joint
                    current_pos = new_pos

                    path.append(current_pos.copy())
                    joint_configs.append(current_joint.copy())

                    new_sef = calculate_joint_sef(current_joint[0], current_joint[1], q1_opt, q2_opt, comfort_threshold,
                                                  weights)
                    sef_values.append(new_sef)
                    break
            else:
                # Special case for low-to-high ergonomic transitions
                # If stuck in a comfortable region, take larger steps towards goal
                if current_sef < 0 and stagnation_count > 20:
                    # Try larger direct steps toward goal
                    for scale in [0.2, 0.3, 0.4]:
                        test_pos = current_pos + scale * goal_direction
                        ik_solutions = inverse_kinematics(test_pos[0], test_pos[1], l1, l2)

                        if ik_solutions is not None:
                            best_index = 0
                            min_dist = float('inf')

                            for i, (q1, q2) in enumerate(ik_solutions):
                                dist = np.linalg.norm(np.array([q1, q2]) - current_joint)
                                if dist < min_dist:
                                    min_dist = dist
                                    best_index = i

                            new_joint = np.array(ik_solutions[best_index])
                            current_joint = new_joint
                            current_pos = test_pos

                            path.append(current_pos.copy())
                            joint_configs.append(current_joint.copy())

                            new_sef = calculate_joint_sef(current_joint[0], current_joint[1], q1_opt, q2_opt,
                                                          comfort_threshold, weights)
                            sef_values.append(new_sef)
                            stagnation_count = 0
                            break

    # Apply path smoothing for final trajectory
    path_array = np.array(path)
    joint_array = np.array(joint_configs)

    # Smooth the Cartesian path
    if len(path_array) > 4:
        smoothed_path = smooth_path(path_array, smoothing_factor=0.3)

        # Recalculate joint configurations and SEF values for smoothed path
        joint_configs_smooth = []
        sef_values_smooth = []

        for pos in smoothed_path:
            ik_solutions = inverse_kinematics(pos[0], pos[1], l1, l2)
            if ik_solutions is None:
                continue

            if joint_configs_smooth:
                # Choose configuration closest to previous one
                prev_joint = joint_configs_smooth[-1]
                best_index = 0
                min_dist = float('inf')

                for i, (q1, q2) in enumerate(ik_solutions):
                    dist = np.linalg.norm(np.array([q1, q2]) - prev_joint)
                    if dist < min_dist:
                        min_dist = dist
                        best_index = i

                joint_config = np.array(ik_solutions[best_index])
            else:
                # For first point, use original start joint
                joint_config = joint_array[0]

            joint_configs_smooth.append(joint_config)
            sef_values_smooth.append(calculate_joint_sef(joint_config[0], joint_config[1],
                                                         q1_opt, q2_opt, comfort_threshold, weights))

        # Use smoothed path if valid
        if len(joint_configs_smooth) > 4:
            return np.array(smoothed_path), np.array(joint_configs_smooth), np.array(sef_values_smooth)

    return np.array(path), np.array(joint_configs), np.array(sef_values)


def compare_paths(joint_path, cart_path_cartesian, cart_path_joint, sef_values_joint, sef_values_cart, l1, l2):
    """
    Compare paths planned in joint space and Cartesian space

    Parameters:
    - joint_path: Joint space path from joint space planning
    - cart_path_cartesian: Cartesian space path from Cartesian space planning
    - cart_path_joint: Joint space path from Cartesian space planning
    - sef_values_joint: SEF values for joint space planning path
    - sef_values_cart: SEF values for Cartesian space planning path
    """
    # Calculate Cartesian trajectory for joint space path
    joint_cartesian_path = []
    for q in joint_path:
        x, y = forward_kinematics(q[0], q[1], l1, l2)
        joint_cartesian_path.append([x, y])
    joint_cartesian_path = np.array(joint_cartesian_path)

    # Calculate path length (Cartesian space)
    joint_length = 0
    for i in range(1, len(joint_cartesian_path)):
        joint_length += np.linalg.norm(joint_cartesian_path[i] - joint_cartesian_path[i - 1])

    cart_length = 0
    for i in range(1, len(cart_path_cartesian)):
        cart_length += np.linalg.norm(cart_path_cartesian[i] - cart_path_cartesian[i - 1])

    # Calculate path length (joint space)
    joint_angle_length = 0
    for i in range(1, len(joint_path)):
        joint_angle_length += np.linalg.norm(joint_path[i] - joint_path[i - 1])

    cart_angle_length = 0
    for i in range(1, len(cart_path_joint)):
        cart_angle_length += np.linalg.norm(cart_path_joint[i] - cart_path_joint[i - 1])

    # Calculate average SEF value
    joint_avg_sef = np.mean(sef_values_joint)
    cart_avg_sef = np.mean(sef_values_cart)

    # Calculate maximum SEF value
    joint_max_sef = np.max(sef_values_joint)
    cart_max_sef = np.max(sef_values_cart)

    # Calculate path smoothness (measured by direction changes in Cartesian space)
    joint_smoothness = 0
    for i in range(1, len(joint_cartesian_path) - 1):
        v1 = joint_cartesian_path[i] - joint_cartesian_path[i - 1]
        v2 = joint_cartesian_path[i + 1] - joint_cartesian_path[i]

        v1_norm = np.linalg.norm(v1)
        v2_norm = np.linalg.norm(v2)

        if v1_norm > 0 and v2_norm > 0:
            cos_angle = np.dot(v1, v2) / (v1_norm * v2_norm)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Handle numerical errors
            angle = np.arccos(cos_angle)
            joint_smoothness += angle

    cart_smoothness = 0
    for i in range(1, len(cart_path_cartesian) - 1):
        v1 = cart_path_cartesian[i] - cart_path_cartesian[i - 1]
        v2 = cart_path_cartesian[i + 1] - cart_path_cartesian[i]

        v1_norm = np.linalg.norm(v1)
        v2_norm = np.linalg.norm(v2)

        if v1_norm > 0 and v2_norm > 0:
            cos_angle = np.dot(v1, v2) / (v1_norm * v2_norm)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.arccos(cos_angle)
            cart_smoothness += angle

    # Prepare comparison results
    comparison = {
        "Cartesian Path Length": {"Joint Space Planning": joint_length, "Cartesian Space Planning": cart_length},
        "Joint Space Path Length": {"Joint Space Planning": joint_angle_length,
                                    "Cartesian Space Planning": cart_angle_length},
        "Average SEF Value": {"Joint Space Planning": joint_avg_sef, "Cartesian Space Planning": cart_avg_sef},
        "Maximum SEF Value": {"Joint Space Planning": joint_max_sef, "Cartesian Space Planning": cart_max_sef},
        "Path Smoothness (angle sum)": {"Joint Space Planning": joint_smoothness,
                                        "Cartesian Space Planning": cart_smoothness}
    }

    return comparison


def visualize_comparison(joint_path, cart_path_cartesian, cart_path_joint, sef_values_joint, sef_values_cart,
                         q1_opt, q2_opt, l1, l2, comfort_threshold, weights, q_ranges, comparison, case_name=""):
    """
    Visualize comparison between joint space and Cartesian space planning results
    """
    q1_min, q1_max, q2_min, q2_max = q_ranges

    # Create joint space and Cartesian space SEF fields
    Q1, Q2, joint_sef = create_joint_sef_field(q1_min, q1_max, q2_min, q2_max,
                                               q1_opt, q2_opt, comfort_threshold,
                                               weights, resolution=200)

    X, Y, cart_sef, opt_pos = create_cartesian_sef_field(l1, l2, q1_opt, q2_opt,
                                                         comfort_threshold, weights,
                                                         resolution=200)

    # Calculate Cartesian trajectory for joint space path
    joint_cartesian_path = []
    for q in joint_path:
        x, y = forward_kinematics(q[0], q[1], l1, l2)
        joint_cartesian_path.append([x, y])
    joint_cartesian_path = np.array(joint_cartesian_path)

    # Create figure
    fig = plt.figure(figsize=(20, 14))
    gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 0.3])

    # Add case name to title
    title_suffix = f" - {case_name}" if case_name else ""

    # Joint space subplot
    ax_joint = plt.subplot(gs[0, 0])

    # Visualize joint space SEF
    cmap = plt.cm.RdBu_r
    max_abs = max(abs(np.min(joint_sef)), abs(np.max(joint_sef)))
    norm = plt.Normalize(-max_abs, max_abs)

    contour = ax_joint.contourf(Q1, Q2, joint_sef, 20, cmap=cmap, norm=norm, alpha=0.8)
    zero_contour = ax_joint.contour(Q1, Q2, joint_sef, [0], colors='green', linewidths=2)
    ax_joint.clabel(zero_contour, inline=True, fontsize=8, fmt='%1.1f')

    # Plot joint space paths
    ax_joint.plot(joint_path[:, 0], joint_path[:, 1], 'r-', linewidth=2, label='Joint Space Planning')
    ax_joint.plot(cart_path_joint[:, 0], cart_path_joint[:, 1], 'b--', linewidth=2, label='Cartesian Space Planning')

    start_q_joint = joint_path[0]
    end_q_joint = joint_path[-1]
    ax_joint.plot(start_q_joint[0], start_q_joint[1], 'go', markersize=8)
    ax_joint.plot(end_q_joint[0], end_q_joint[1], 'mo', markersize=8)
    ax_joint.plot(q1_opt, q2_opt, 'ko', markersize=8, )

    ax_joint.set_xlabel('q1 (rad)')
    ax_joint.set_ylabel('q2 (rad)')
    ax_joint.set_title(f'Joint Space Path Comparison{title_suffix}')
    ax_joint.grid(True, linestyle='--', alpha=0.6)
    ax_joint.legend(loc='best')

    # Workspace subplot
    ax_cart = plt.subplot(gs[0, 1])

    # Define workspace boundaries
    workspace_radius = l1 + l2

    # Inner boundary (|l1-l2|)
    inner_radius = abs(l1 - l2)
    inner_circle = Circle((0, 0), inner_radius, fill=False, linestyle='--', color='gray')
    ax_cart.add_patch(inner_circle)

    # Outer boundary (l1+l2)
    outer_circle = Circle((0, 0), workspace_radius, fill=False, linestyle='--', color='black')
    ax_cart.add_patch(outer_circle)

    # Visualize Cartesian space SEF
    valid_mask = ~np.isnan(cart_sef)
    if np.any(valid_mask):
        max_abs = max(abs(np.nanmin(cart_sef)), abs(np.nanmax(cart_sef)))
        norm_cart = plt.Normalize(-max_abs, max_abs)

        contour_cart = ax_cart.contourf(X, Y, cart_sef, 20, cmap=cmap, norm=norm_cart, alpha=0.8)
        zero_contour_cart = ax_cart.contour(X, Y, cart_sef, [0], colors='green', linewidths=2)
        ax_cart.clabel(zero_contour_cart, inline=True, fontsize=8, fmt='%1.1f')

    # Plot workspace trajectories
    ax_cart.plot(joint_cartesian_path[:, 0], joint_cartesian_path[:, 1], 'r-', linewidth=2)
    ax_cart.plot(cart_path_cartesian[:, 0], cart_path_cartesian[:, 1], 'b--', linewidth=2)

    start_cart_joint = joint_cartesian_path[0]
    end_cart_joint = joint_cartesian_path[-1]

    ax_cart.plot(start_cart_joint[0], start_cart_joint[1], 'go', markersize=8)
    ax_cart.plot(end_cart_joint[0], end_cart_joint[1], 'mo', markersize=8)
    ax_cart.plot(opt_pos[0], opt_pos[1], 'ko', markersize=8)

    ax_cart.set_xlabel('X')
    ax_cart.set_ylabel('Y')
    ax_cart.set_title(f'Cartesian Space Trajectory Comparison{title_suffix}')
    ax_cart.set_xlim([-workspace_radius * 1.2, workspace_radius * 1.2])
    ax_cart.set_ylim([-workspace_radius * 1.2, workspace_radius * 1.2])
    ax_cart.set_aspect('equal')
    ax_cart.grid(True, linestyle='--', alpha=0.6)
    ax_cart.legend(loc='best')

    # SEF value curve subplot
    ax_sef = plt.subplot(gs[1, 0])

    # Calculate normalized path length
    joint_length = np.linspace(0, 1, len(sef_values_joint))
    cart_length = np.linspace(0, 1, len(sef_values_cart))

    ax_sef.plot(joint_length, sef_values_joint, 'r-', linewidth=2)
    ax_sef.plot(cart_length, sef_values_cart, 'b--', linewidth=2)
    ax_sef.axhline(y=0, color='g', linestyle='-', linewidth=1)

    ax_sef.set_xlabel('Normalized Path Length')
    ax_sef.set_ylabel('SEF Value')
    ax_sef.set_title('SEF Values Along Path')
    ax_sef.grid(True, linestyle='--', alpha=0.6)
    ax_sef.legend(loc='best')

    # Comparison result table
    ax_comparison = plt.subplot(gs[1, 1])
    ax_comparison.axis('tight')
    ax_comparison.axis('off')

    # Prepare table data
    table_data = []
    table_colors = []

    for metric, values in comparison.items():
        joint_value = values["Joint Space Planning"]
        cart_value = values["Cartesian Space Planning"]

        # Determine which is better (for SEF values, lower is better; for smoothness, lower is better)
        if metric.startswith("Average SEF") or metric.startswith("Maximum SEF") or metric.startswith("Path Smoothness"):
            better = "Joint Space Planning" if joint_value < cart_value else "Cartesian Space Planning"
        else:
            # For path lengths, we don't judge better/worse, just show info
            better = "N/A"

        row = [metric, f"{joint_value:.4f}", f"{cart_value:.4f}"]

        row_colors = ['w', 'w', 'w']
        if better == "Joint Space Planning":
            row_colors[1] = '#d5f5e3'  # Light green
        elif better == "Cartesian Space Planning":
            row_colors[2] = '#d5f5e3'  # Light green

        table_data.append(row)
        table_colors.append(row_colors)

    column_labels = ['Metric', 'Joint Space Planning', 'Cartesian Space Planning']
    table = ax_comparison.table(cellText=table_data, colLabels=column_labels,
                                loc='center', cellLoc='center',
                                cellColours=table_colors)

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)

    # Color bar
    cax = plt.subplot(gs[2, :])
    cbar = plt.colorbar(contour, cax=cax, orientation='horizontal')
    cbar.set_label('SEF Value (negative values indicate comfortable regions)')

    plt.tight_layout()
    return fig


def run_comparison(start_q, end_q, q1_opt, q2_opt, l1, l2, comfort_threshold, weights, q_ranges, case_name=""):
    """
    Run comparison for a specific configuration
    """
    # Calculate start and end positions in Cartesian space
    start_pos = forward_kinematics(start_q[0], start_q[1], l1, l2)
    end_pos = forward_kinematics(end_q[0], end_q[1], l1, l2)

    print(f"\nProcessing case: {case_name}")
    print(f"Start: {start_q}, End: {end_q}")

    # Calculate path in joint space
    start_time = time.time()
    joint_path, sef_values_joint = calculate_joint_space_path(
        start_q, end_q, q1_opt, q2_opt, comfort_threshold, weights,
        step_size=0.03, max_steps=2000, goal_weight=0.5, random_factor=0.02
    )
    joint_planning_time = time.time() - start_time

    # Create Cartesian space SEF field
    X, Y, cart_sef, opt_pos = create_cartesian_sef_field(
        l1, l2, q1_opt, q2_opt, comfort_threshold, weights, resolution=200
    )

    # Calculate path in Cartesian space
    start_time = time.time()
    cart_path_cartesian, cart_path_joint, sef_values_cart = calculate_cartesian_space_path(
        start_pos, end_pos, start_q, end_q, X, Y, cart_sef, l1, l2, q1_opt, q2_opt,
        comfort_threshold, weights, step_size=0.03, max_steps=2000,
        goal_weight=0.5, random_factor=0.02
    )
    cart_planning_time = time.time() - start_time

    # Compare planning methods
    comparison = compare_paths(
        joint_path, cart_path_cartesian, cart_path_joint,
        sef_values_joint, sef_values_cart, l1, l2
    )

    # Add planning time to comparison results
    comparison["Planning Time (seconds)"] = {"Joint Space Planning": joint_planning_time,
                                             "Cartesian Space Planning": cart_planning_time}

    # Visualize comparison
    fig = visualize_comparison(
        joint_path, cart_path_cartesian, cart_path_joint,
        sef_values_joint, sef_values_cart,
        q1_opt, q2_opt, l1, l2, comfort_threshold, weights, q_ranges, comparison,
        case_name=case_name
    )

    # Return comparison results and visualization
    return comparison, fig


def main():
    # Set random seed for reproducibility
    np.random.seed(42)

    # Define parameters
    l1, l2 = 1.0, 0.8
    q1_opt, q2_opt = np.pi / 4, -np.pi / 3
    comfort_threshold = 0.5
    weights = [1, 1]

    # Define joint ranges
    q1_min, q1_max = -np.pi, np.pi
    q2_min, q2_max = -np.pi, np.pi
    q_ranges = [q1_min, q1_max, q2_min, q2_max]

    # Define 10 different test configurations
    test_configs = [
        {
            "name": "Case 1: Opposite Corners",
            "start_q": [-0.8, 0.6],
            "end_q": [0.8, -0.8]
        },
        {
            "name": "Case 2: Near Optimal to Far",
            "start_q": [np.pi / 4 + 0.1, -np.pi / 3 - 0.1],  # Near optimal point
            "end_q": [-np.pi / 2, np.pi / 2]  # Far from optimal
        },
        {
            "name": "Case 3: Cross Comfort Boundary",
            "start_q": [np.pi / 4 + 0.3, -np.pi / 3 + 0.3],  # Just outside comfort region
            "end_q": [np.pi / 4 - 0.3, -np.pi / 3 - 0.3]  # Inside comfort region
        },
        {
            "name": "Case 4: Long Path Around Optimal",
            "start_q": [np.pi, 0],
            "end_q": [-np.pi, 0]
        },
        {
            "name": "Case 5: Edge to Center",
            "start_q": [0, -np.pi],  # Near fully stretched configuration
            "end_q": [np.pi / 2, np.pi / 2]  # Bent arm position
        },
        {
            "name": "Case 6: Near Singularity",
            "start_q": [0, 0.1],  # Near singularity
            "end_q": [0, -0.1]  # Near singularity
        },
        {
            "name": "Case 7: Elbow-up to Elbow-down",
            "start_q": [np.pi / 4, np.pi / 4],  # Elbow up
            "end_q": [np.pi / 4, -np.pi / 4]  # Elbow down
        },
        {
            "name": "Case 8: Low Ergo to High Ergo",
            "start_q": [np.pi / 4 - 0.3, -np.pi / 3 - 0.3],  # Inside comfort region (negative SEF)
            "end_q": [np.pi, 0]  # High SEF region
        },
        {
            "name": "Case 9: High Ergo to Low Ergo",
            "start_q": [np.pi, 0],  # High SEF region
            "end_q": [np.pi / 4 - 0.3, -np.pi / 3 - 0.3]  # Inside comfort region (negative SEF)
        },
        {
            "name": "Case 10: Diagonal Comfort Crossing",
            "start_q": [np.pi / 4 - 0.6, -np.pi / 3 + 0.6],  # Outside comfort region
            "end_q": [np.pi / 4 + 0.6, -np.pi / 3 - 0.6]  # Outside comfort region, crossing through comfort zone
        }
    ]

    # Create PDF to save all visualizations
    pdf = PdfPages("path_planning_comparison.pdf")

    # Create lists to store comparison metrics for all cases
    all_comparisons = []

    # Run comparisons for all configurations
    for config in test_configs:
        start_q = config["start_q"]
        end_q = config["end_q"]
        case_name = config["name"]

        comparison, fig = run_comparison(
            start_q, end_q, q1_opt, q2_opt, l1, l2,
            comfort_threshold, weights, q_ranges, case_name
        )

        # Save comparison results
        comparison["Case"] = case_name
        all_comparisons.append(comparison)

        # Save figure to PDF
        pdf.savefig(fig)
        plt.close(fig)

    # Create a summary table of all comparisons
    metrics = [
        "Average SEF Value",
        "Maximum SEF Value",
        "Path Smoothness (angle sum)",
        "Planning Time (seconds)"
    ]

    summary_data = []
    for comp in all_comparisons:
        case_row = [comp["Case"]]

        for metric in metrics:
            joint_value = comp[metric]["Joint Space Planning"]
            cart_value = comp[metric]["Cartesian Space Planning"]

            # For SEF value and smoothness, lower is better
            if metric in ["Average SEF Value", "Maximum SEF Value", "Path Smoothness (angle sum)"]:
                winner = "Joint" if joint_value < cart_value else "Cart"
            else:
                winner = "Joint" if joint_value < cart_value else "Cart"

            case_row.append(f"{joint_value:.4f}")
            case_row.append(f"{cart_value:.4f}")
            case_row.append(winner)

        summary_data.append(case_row)

    # Create column headers for the summary table
    columns = ["Case"]
    for metric in metrics:
        columns.extend([f"{metric} (Joint)", f"{metric} (Cart)", "Winner"])

    # Create DataFrame for summary data
    df_summary = pd.DataFrame(summary_data, columns=columns)

    # Create a summary table figure
    fig_summary = plt.figure(figsize=(20, 10))
    ax_summary = fig_summary.add_subplot(111)
    ax_summary.axis('off')

    # Plot the table
    table = ax_summary.table(
        cellText=df_summary.values,
        colLabels=df_summary.columns,
        loc='center',
        cellLoc='center'
    )

    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.5)

    # Add a title
    plt.suptitle("Summary Comparison of Joint Space vs. Cartesian Space Planning", fontsize=16)
    plt.tight_layout()

    # Save the summary table
    pdf.savefig(fig_summary)
    plt.close(fig_summary)

    # Close the PDF
    pdf.close()

    # Also calculate win counts
    joint_wins = 0
    cart_wins = 0
    ties = 0

    for comp in all_comparisons:
        joint_avg_sef = comp["Average SEF Value"]["Joint Space Planning"]
        cart_avg_sef = comp["Average SEF Value"]["Cartesian Space Planning"]

        joint_smoothness = comp["Path Smoothness (angle sum)"]["Joint Space Planning"]
        cart_smoothness = comp["Path Smoothness (angle sum)"]["Cartesian Space Planning"]

        # Score for this case based on average SEF and smoothness
        joint_points = 0
        cart_points = 0

        if joint_avg_sef < cart_avg_sef:
            joint_points += 1
        elif cart_avg_sef < joint_avg_sef:
            cart_points += 1

        if joint_smoothness < cart_smoothness:
            joint_points += 1
        elif cart_smoothness < joint_smoothness:
            cart_points += 1

        if joint_points > cart_points:
            joint_wins += 1
        elif cart_points > joint_points:
            cart_wins += 1
        else:
            ties += 1

    print("\nOverall Comparison:")
    print(f"Joint Space Planning Wins: {joint_wins}")
    print(f"Cartesian Space Planning Wins: {cart_wins}")
    print(f"Ties: {ties}")

    print("\nDetailed results saved to 'path_planning_comparison.pdf'")


if __name__ == "__main__":
    main()