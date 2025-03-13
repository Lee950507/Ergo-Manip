import numpy as np
import math
from scipy.optimize import minimize
import utils


def relative_pose_constraint(x):
    # Compute hand positions
    elbow_right, wrist_right = forward_kinematics(x[:4], d_uar, d_lar)
    elbow_left, wrist_left = forward_kinematics(x[4:], d_ual, d_lal)

    # Transformation
    global_positions_wrist_right = global_positions[3] + [-wrist_right[1],-wrist_right[0], wrist_right[2]]
    global_positions_wrist_left = global_positions[6] + [-wrist_left[1], wrist_left[0], wrist_left[2]]

    # Compute relative pose between hands
    updated_relative_hand = euclidean_distance(global_positions_wrist_right, global_positions_wrist_left)
    return updated_relative_hand - last_relative_displacement_wrists


def joint_range_constraint(x):
    """
    Ensure joint angles remain within realistic ranges.
    """
    # Check right shoulder and elbow
    if not (-math.pi / 18 <= x[0] <= 17 * math.pi / 18):
        return -1  # Violation
    if not (-math.pi / 3 <= x[1] <= 17 * math.pi / 18):
        return -1  # Violation
    if not (-math.pi / 3 <= x[2] <= math.pi / 2):
        return -1  # Violation
    if not (-math.pi / 2 <= x[3] <= math.pi / 3):
        return -1  # Violation

    # Check right shoulder and elbow
    if not (-math.pi / 18 <= x[4] <= 17 * math.pi / 18):
        return -1  # Violation
    if not (-math.pi / 3 <= x[5] <= 17 * math.pi / 18):
        return -1  # Violation
    if not (-math.pi / 3 <= x[6] <= math.pi / 2):
        return -1  # Violation
    if not (-math.pi / 2 <= x[7] <= math.pi / 3):
        return -1  # Violation

    return 0  # Satisfied


def manipulability_z_constraint(q, d_ua, d_la, arm='right'):
    # Compute the Jacobian matrix
    J = compute_jacobian(q, d_ua, d_la, arm)

    # Compute the force manipulability ellipsoid matrix
    JJ_T_inv = np.linalg.inv(J @ J.T)
    eigenvalues, eigenvectors = np.linalg.eigh(JJ_T_inv)

    z_axis = np.array([0, 0, 1])
    projections = [abs(np.dot(eigenvectors[:, i], z_axis)) for i in range(3)]
    z_index = np.argmax(projections)  # Index of the eigenvector closest to Z-axis
    length_z = np.sqrt(eigenvalues[z_index])  # Length along Z-axis

    # Return the constraint value (should be >= 0)
    return length_z


# Objective function: Minimize REBA score
def objective_function(x):
    """
    Compute the total REBA score for the right and left arms.
    :param x: Array of joint angles [right_shoulder_angles, right_elbow_angle, left_shoulder_angles, left_elbow_angle].
    :return: Total REBA score.
    """
    # Compute elbow and hand positions
    elbow_right, hand_right = forward_kinematics(x[:4], d_uar, d_lar)
    elbow_left, hand_left = forward_kinematics(x[4:], d_ual, d_lal)

    # Compute REBA scores (assume compute_reba_score() is defined elsewhere)
    overall_arm_score_left = utils.calculate_upper_limb_score_with_joint_angles(x[4:])

    overall_arm_score_right = utils.calculate_upper_limb_score_with_joint_angles(x[:4])

    print(overall_arm_score_right, overall_arm_score_left)

    reba_score = max(overall_arm_score_left, overall_arm_score_right)

    reba_threshold = 2  # Define the ergonomic threshold
    penalty_reba = 1000 * reba_score ** 2

    manip_right = manipulability_z_constraint(x[:4], d_uar, d_lar, arm='right')
    manip_left = manipulability_z_constraint(x[4:], d_ual, d_lal, arm='left')
    manip_threshold = 2.5

    if reba_score <= reba_threshold and manip_right >= manip_threshold and manip_left >= manip_threshold:
        return reba_score  # Allow optimization to stop early

    penalty_manip_l = 10 * 1 / math.log(manip_left)
    penalty_manip_r = 10 * 1 / math.log(manip_right)

    regularization_2 = 0.05 * np.sum(np.square(x - initial_guess))

    # Total cost: REBA score + penalty + regularization
    total_cost = penalty_reba + regularization_2 + penalty_manip_r + penalty_manip_l

    return total_cost


def optimized_joints(q):
    # Initial_guess, constraints and bounds
    initial_guess = q
    constraints = [
        {'type': 'eq', 'fun': relative_pose_constraint},  # Fixed relative pose
        {'type': 'eq', 'fun': joint_range_constraint}  # Joint range of motion
    ]
    bounds = [
        (-math.pi / 18, 17 * math.pi / 18),
        (-np.pi / 3, 17 * math.pi / 18),
        (-np.pi / 3, np.pi / 2),
        (-np.pi / 2, np.pi / 3),
        (-math.pi / 18, 17 * math.pi / 18),
        (-np.pi / 3, 17 * math.pi / 18),
        (-np.pi / 3, np.pi / 2),
        (-np.pi / 2, np.pi / 3)
    ]

    # Solve the optimization problem
    result = minimize(objective_function, initial_guess, method='trust-constr', constraints=constraints, bounds=bounds)

    # Output the optimized joint angles
    if result.success:
        optimized_angles = result.x
        final_reba_score = objective_function(optimized_angles)

        if final_reba_score <= 10:  # Check if the score is acceptable
            print("Optimization successful!")
            print("Optimized Joint Angles:", optimized_angles)
            print("Final REBA Score:", final_reba_score)
        else:
            print("Optimized Joint Angles:", optimized_angles)
            print(
                f"Optimization finished, but REBA score is above the acceptable threshold. Final REBA score: {final_reba_score}")
    else:
        print("Optimization failed:", result.message)

    return optimized_angles