import numpy as np
import utils
import matplotlib.pyplot as plt
import math

from scipy.optimize import minimize
from mpl_toolkits.mplot3d import Axes3D


def euclidean_distance(point1, point2):
    distance = 0
    for i in range(len(point1)):
        distance += math.pow(point2[i] - point1[i], 2)
    return math.sqrt(distance)


def compute_rotation_matrix(v_initial, v_target):
    """
    Compute the rotation matrix to align v_initial with v_target.
    :param v_initial: Initial vector (3D).
    :param v_target: Target vector (3D).
    :return: Rotation matrix (3x3).
    """
    v_initial = v_initial / np.linalg.norm(v_initial)  # Normalize initial vector
    v_target = v_target / np.linalg.norm(v_target)    # Normalize target vector
    cross_prod = np.cross(v_initial, v_target)        # Cross product
    dot_prod = np.dot(v_initial, v_target)           # Dot product
    identity = np.eye(3)

    # Skew-symmetric cross-product matrix
    skew_symmetric = np.array([
        [0, -cross_prod[2], cross_prod[1]],
        [cross_prod[2], 0, -cross_prod[0]],
        [-cross_prod[1], cross_prod[0], 0]
    ])

    # Compute the rotation matrix using Rodrigues' rotation formula
    rotation_matrix = identity + skew_symmetric + skew_symmetric @ skew_symmetric * ((1 - dot_prod) / (np.linalg.norm(cross_prod) ** 2 + 1e-9))
    return rotation_matrix


def transform_box(box_center, box_orientation, box_dims, rotation_matrix, position_offset):
    """
    Transform the box pose based on the given rotation matrix.
    :param box_center: The center of the box (3D).
    :param box_orientation: Initial orientation of the box (3x3 matrix).
    :param box_dims: Dimensions of the box [length, width, height].
    :param rotation_matrix: Rotation matrix to apply to the box.
    :return: Transformed corners of the box.
    """
    # Box dimensions
    length, width, height = box_dims

    # Define the 8 corners of the box relative to its center
    half_length, half_width, half_height = length / 2, width / 2, height / 2
    corners = np.array([
        [-half_length, -half_width, -half_height],
        [half_length, -half_width, -half_height],
        [-half_length, half_width, -half_height],
        [half_length, half_width, -half_height],
        [-half_length, -half_width, half_height],
        [half_length, -half_width, half_height],
        [-half_length, half_width, half_height],
        [half_length, half_width, half_height],
    ])

    # Apply the initial box orientation
    oriented_corners = (box_orientation @ corners.T).T

    # Apply the rotation matrix to the corners
    rotated_corners = (rotation_matrix @ oriented_corners.T).T
    updated_center = box_center + position_offset

    # Translate the corners to the new box center
    transformed_corners = rotated_corners + updated_center

    return transformed_corners


def update_box_and_hands(ax, left_hand, right_hand, init_left_hand, init_right_hand, box_center, box_orientation, box_dims, v_initial, color='b'):
    """
    Update the box pose and hands positions in the Cartesian space.
    :param ax: The Matplotlib 3D axis to plot on.
    :param left_hand: New position of the left hand (3D).
    :param right_hand: New position of the right hand (3D).
    :param box_center: Initial center of the box (3D).
    :param box_orientation: Initial orientation of the box (3x3 matrix).
    :param box_dims: Dimensions of the box [length, width, height].
    :param v_initial: Initial vector between the left and right hands.
    """
    # Compute the new vector between the hands
    v_target = np.array(right_hand) - np.array(left_hand)

    # Compute the rotation matrix to align the initial vector with the target vector
    rotation_matrix = compute_rotation_matrix(v_initial, v_target)

    # Compute the initial and new hand midpoints
    initial_midpoint = (np.array(init_left_hand) + np.array(init_right_hand)) / 2
    new_midpoint = (np.array(left_hand) + np.array(right_hand)) / 2

    # Compute the position offset for the box
    position_offset = new_midpoint - initial_midpoint

    # Transform the box pose
    transformed_corners = transform_box(box_center, box_orientation, box_dims, rotation_matrix, position_offset)

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 2)
    ax.set_zlim(0, 2)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Plot the transformed box
    utils.plot_box(ax, transformed_corners, color=color)
    utils.plot_skeleton(ax, global_positions, skeleton_parent_indices, color=color)


def forward_kinematics(q, d_ua, d_la):
    T1 = np.array([[math.cos(q[0]), 0, math.sin(q[0])], [0, 1, 0], [-math.sin(q[0]), 0, math.cos(q[0])]])
    T2 = np.array([[1, 0, 0], [0, math.cos(q[1]), -math.sin(q[1])], [0, math.sin(q[1]), math.cos(q[1])]])
    T3 = np.array([[math.cos(q[2]), -math.sin(q[2]), 0], [math.sin(q[2]), math.cos(q[2]), 0], [0, 0, 1]])
    T4 = np.array([[1, 0, 0], [0, math.cos(q[3]), -math.sin(q[3])], [0, math.sin(q[3]), math.cos(q[3])]])
    p_elbow = T1 @ T2 @ T3 @ d_ua
    p_hand = T1 @ T2 @ T3 @ (d_ua + T4 @ d_la)
    # print(p_elbow, p_hand)
    return p_elbow, p_hand


def inverse_kinematics(p_elbow_desired, p_hand_desired, d_ua, d_la, q_initial=None):
    """
    Optimize the joint angles to match the desired elbow and hand positions.
    :param p_elbow_desired: Desired elbow position (3D vector)
    :param p_hand_desired: Desired hand position (3D vector)
    :param d_ua: Upper arm vector (3D vector from shoulder to elbow)
    :param d_la: Lower arm vector (3D vector from elbow to hand)
    :param q_initial: Initial guess for the joint angles, optional
    :return: Optimized joint angles [q0, q1, q2, q3]
    """
    # Objective function: Minimize the position error for the elbow and hand
    def objective(q):
        p_elbow, p_hand = forward_kinematics(q, d_ua, d_la)
        error_elbow = np.linalg.norm(p_elbow_desired - p_elbow) ** 2
        error_hand = np.linalg.norm(p_hand_desired - p_hand) ** 2
        return error_elbow + error_hand  # Total error

    # Constraints on joint angles (e.g., joint limits)
    bounds = [
        (-np.pi, np.pi),  # q0 (shoulder rotation around Y-axis)
        (-np.pi / 2, np.pi / 2),  # q1 (shoulder rotation around X-axis)
        (-np.pi, np.pi),  # q2 (shoulder rotation around Z-axis)
        (-np.pi / 2, np.pi / 2),  # q3 (elbow rotation around X-axis)
    ]

    # Initial guess for joint angles
    if q_initial is None:
        q_initial = [0, 0, 0, 0]  # Default initial guess

    # Optimize the joint angles
    result = minimize(objective, q_initial, bounds=bounds, method='SLSQP')

    # Check if the optimization was successful
    if result.success:
        return result.x  # Optimized joint angles
    else:
        raise ValueError(f"Inverse kinematics optimization failed: {result.message}")


def compute_jacobian(q, d_ua, d_la, arm):
    """
    Compute the Jacobian matrix for the arm.
    :param q: Joint angles [q0, q1, q2, q3]
    :param d_ua: Upper arm vector (3D vector from shoulder to elbow)
    :param d_la: Lower arm vector (3D vector from elbow to hand)
    :return: Jacobian matrix (3x4)
    """
    q0, q1, q2, q3 = q
    l1 = np.linalg.norm(d_ua)  # Length of the upper arm
    l2 = np.linalg.norm(d_la)  # Length of the lower arm

    # Precompute trigonometric terms for efficiency
    c0, s0 = math.cos(q0), math.sin(q0)
    c1, s1 = math.cos(q1), math.sin(q1)
    c2, s2 = math.cos(q2), math.sin(q2)
    c3, s3 = math.cos(q3), math.sin(q3)

    # Position of the hand
    p_elbow, p_hand = forward_kinematics(q, d_ua, d_la)

    # Partial derivatives with respect to q0 (shoulder rotation about Y-axis)
    dp_dq0 = np.array([
        -l1 * c0 * c1 + l2 * (s0 * s2 * c3 + c0 * s1 * c2 * c3 + c0 * c1 * s3),
        0,
        l1 * s0 * c1 + l2 * (c0 * s2 * c3 - s0 * s1 * c2 * c3 - s0 * c1 * s3)
    ])

    # Partial derivatives with respect to q1 (shoulder rotation about X-axis)
    dp_dq1 = np.array([
        l1 * s0 * s1 + l2 * (s0 * c1 * c2 * c3 - s0 * s1 * s3),
        l1 * c1 - l2 * s1 * c2 * c3 - l2 * c1 * s3,
        l1 * c0 * s1 + l2 * (c0 * c1 * c2 * c3 - c0 * s1 * s3)
    ])

    # Partial derivatives with respect to q2 (shoulder rotation about Z-axis)
    dp_dq2 = np.array([
        -l2 * c0 * c2 * c3 - l2 * s0 * s1 * s2 * c3,
        -l2 * c1 * s2 * c3,
        l2 * s0 * c2 * c3 - l2 * c0 * s1 * s2 * c3
    ])

    # Partial derivatives with respect to q3 (elbow rotation about X-axis)
    dp_dq3 = np.array([
        l2 * c0 * s2 * s3 - l2 * s0 * s1 * c2 * s3 + l2 * s0 * c1 * c3,
        -l2 * c1 * c2 * s3 - l2 * s1 * c3,
        -l2 * s0 * s2 * s3 - l2 * c0 * s1 * c2 * s3 + l2 * c0 * c1 * c3
    ])

    # Assemble the Jacobian matrix
    J = np.column_stack((dp_dq0, dp_dq1, dp_dq2, dp_dq3))
    if arm == 'left':
        J[[0, 1], :] = J[[1, 0], :]
        J[0, :] = -J[0, :]
    if arm == 'right':
        J[[0, 1], :] = J[[1, 0], :]
        J[0, :] = -J[0, :]
        J[1, :] = -J[1, :]
    # print(J)
    return J


def compute_force_ellipsoid(q, d_ua, d_la, arm='left'):
    """
    Compute the force manipulability ellipsoid for the hand.
    :param q: Joint angles [q0, q1, q2, q3]
    :param d_ua: Upper arm vector (3D vector from shoulder to elbow)
    :param d_la: Lower arm vector (3D vector from elbow to hand)
    :return: Ellipsoid matrix (3x3), eigenvalues, and eigenvectors of the ellipsoid
    """
    # Compute the Jacobian matrix
    J = compute_jacobian(q, d_ua, d_la, arm)

    # Compute the ellipsoid matrix (inverse of J * J^T)
    JJ_T_inv = np.linalg.inv(J @ J.T)

    # Eigenvalues and eigenvectors of the ellipsoid matrix
    eigenvalues, eigenvectors = np.linalg.eigh(JJ_T_inv)

    return eigenvalues, eigenvectors


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


# Constraint: Fixed relative pose between hands
def relative_pose_constraint(x):
    """
    Ensure the relative pose between the two hands remains fixed.
    :param x: Array of joint angles [right_shoulder_angles, right_elbow_angle, left_shoulder_angles, left_elbow_angle].
    :return: Difference between the current and fixed relative pose of the hands.
    """
    # Compute hand positions
    elbow_right, hand_right = forward_kinematics(x[:4], d_uar, d_lar)
    elbow_left, hand_left = forward_kinematics(x[4:], d_ual, d_lal)

    # Transformation
    global_positions_hand_right = global_positions[3] + [-hand_right[1],-hand_right[0], hand_right[2]]
    global_positions_hand_left = global_positions[6] + [-hand_left[1], hand_left[0], hand_left[2]]

    # Compute relative pose between hands
    updated_relative_hand = euclidean_distance(global_positions_hand_right, global_positions_hand_left)
    # print("updated_hand_pos:", global_positions_hand_right, global_positions_hand_left)
    # print("updated_relative_pose_hand:", updated_relative_hand)
    return updated_relative_hand - relative_pose_hands


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

    # Transformation
    global_positions[4] = global_positions[3] + [-elbow_right[1], -elbow_right[0], elbow_right[2]]
    global_positions[7] = global_positions[6] + [-elbow_left[1], elbow_left[0], elbow_left[2]]
    global_positions[5] = global_positions[3] + [-hand_right[1], -hand_right[0], hand_right[2]]
    global_positions[8] = global_positions[6] + [-hand_left[1], hand_left[0], hand_left[2]]

    # Compute REBA scores (assume compute_reba_score() is defined elsewhere)
    overall_arm_score_left = utils.calculate_upper_limb_score_with_joint_angles(x[4:])

    overall_arm_score_right = utils.calculate_upper_limb_score_with_joint_angles(x[:4])

    print(overall_arm_score_right, overall_arm_score_left)

    reba_score = max(overall_arm_score_left, overall_arm_score_right)

    # rf.visualab.plot_skeleton(ax, global_positions, skeleton_parent_indices, color='b')
    # plt.show()

    reba_threshold = 2  # Define the ergonomic threshold
    penalty_reba = 1000 * reba_score ** 2
    # if reba_score > reba_threshold:
    #     penalty_reba = 1000 * (reba_score - reba_threshold) ** 2
    # else:
    #     penalty_reba = reba_score - reba_threshold

    manip_right = manipulability_z_constraint(x[:4], d_uar, d_lar, arm='right')
    manip_left = manipulability_z_constraint(x[4:], d_ual, d_lal, arm='left')
    manip_threshold = 2.5

    if reba_score <= reba_threshold and manip_right >= manip_threshold and manip_left >= manip_threshold:
        return reba_score  # Allow optimization to stop early

    penalty_manip_l = 10 * 1 / math.log(manip_left)
    penalty_manip_r = 10 * 1 / math.log(manip_right)

    # if manip_left < manip_threshold:
    #     penalty_manip_l = 100 * (manip_left - manip_threshold) ** 2
    # else:
    #     penalty_manip_l = - (manip_left - manip_threshold)
    # if manip_right < manip_threshold:
    #     penalty_manip_r = 100 * (manip_right - manip_threshold) ** 2
    # else:
    #     penalty_manip_r = - (manip_right - manip_threshold)

    # Regularization term to encourage joint angles close to neutral (0)
    # regularization = 0.1 * np.sum(np.square(x))
    regularization_2 = 0.05 * np.sum(np.square(x - initial_guess))

    # Total cost: REBA score + penalty + regularization
    total_cost = penalty_reba + regularization_2 + penalty_manip_r + penalty_manip_l

    return total_cost


if __name__ == '__main__':
    # Read and process data
    skeleton_joint_name, skeleton_joint, skeleton_parent_indices, skeleton_joint_local_translation = utils.read_skeleton_motion(
        '/home/ubuntu/Rofunc/examples/data/hotu2/demo_3_chenzui_only_optitrack2hotu.npy')
    skeleton_joint = skeleton_joint[450, :]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.clear()

    global_positions, global_rotations = utils.forward_kinematics(skeleton_joint_local_translation,
                                                                        skeleton_joint, skeleton_parent_indices)

    box_pos = np.array([-0.75, 0, 1.1]) # demo_3_chenzui
    # box_pos = np.array([-0.8, 0, 1.2]) # demo_2_chenzui
    box_ori = np.eye(3)
    box_dims = [1, 0.6, 0.5]


    # Define a structure to map joint types to their corresponding functions and indices
    joint_analysis = {
        'upper_arm': {
            'indices': [skeleton_joint_name.index('right_upper_arm'), skeleton_joint_name.index('left_upper_arm')],
            'degree_func': utils.UpperArmDegree(global_positions).upper_arm_degrees,
            'reba_func': utils.UAREBA,
            'score_func': lambda reba: reba.upper_arm_reba_score_continous(),
            'color_func': utils.ua_get_color
        },
        'lower_arm': {
            'indices': [skeleton_joint_name.index('right_lower_arm'), skeleton_joint_name.index('left_lower_arm')],
            'degree_func': utils.LADegrees(global_positions).lower_arm_degree,
            'reba_func': utils.LAREBA,
            'score_func': lambda reba: reba.lower_arm_score_continous(),
            'color_func': utils.la_get_color
        },
        'trunk': {
            'indices': [skeleton_joint_name.index('pelvis')],
            'degree_func': utils.TrunkDegree(global_positions, global_rotations).trunk_degrees,
            'reba_func': utils.TrunkREBA,
            'score_func': lambda reba: reba.trunk_reba_score(),
            'color_func': utils.trunk_get_color
        }
    }

    length_left_upper_arm = euclidean_distance(global_positions[7], global_positions[6])
    length_right_upper_arm = euclidean_distance(global_positions[4], global_positions[3])
    length_left_lower_arm = euclidean_distance(global_positions[8], global_positions[7])
    length_right_lower_arm = euclidean_distance(global_positions[5], global_positions[4])
    d_ual = np.array([0, 0, -length_left_upper_arm]).T
    d_uar = np.array([0, 0, -length_right_upper_arm]).T
    d_lal = np.array([0, length_left_lower_arm, 0]).T
    d_lar = np.array([0, length_right_lower_arm, 0]).T

    relative_pose_hands = euclidean_distance(global_positions[8], global_positions[5])
    print("relative_pose_hand:", relative_pose_hands)

    p_elbow_left_initial = global_positions[7] - global_positions[6]
    p_elbow_left_initial = np.array([p_elbow_left_initial[1], -p_elbow_left_initial[0], p_elbow_left_initial[2]])
    p_hand_left_initial = global_positions[8] - global_positions[6]
    p_hand_left_initial = np.array([p_hand_left_initial[1], -p_hand_left_initial[0], p_hand_left_initial[2]])

    p_elbow_right_initial = global_positions[4] - global_positions[3]
    p_elbow_right_initial = np.array([-p_elbow_right_initial[1], -p_elbow_right_initial[0], p_elbow_right_initial[2]])
    p_hand_right_initial = global_positions[5] - global_positions[3]
    p_hand_right_initial = np.array([-p_hand_right_initial[1], -p_hand_right_initial[0], p_hand_right_initial[2]])

    q_l = inverse_kinematics(p_elbow_left_initial, p_hand_left_initial, d_ual, d_lal)
    q_r = inverse_kinematics(p_elbow_right_initial, p_hand_right_initial, d_uar, d_lar)

    _, hand_right = forward_kinematics(q_r, d_uar, d_lar)
    _, hand_left = forward_kinematics(q_l, d_ual, d_lal)

    # Transformation
    init_right_hand = global_positions[3] + [-hand_right[1], -hand_right[0], hand_right[2]]
    init_left_hand = global_positions[6] + [-hand_left[1], hand_left[0], hand_left[2]]
    v_initial = init_right_hand - init_left_hand

    update_box_and_hands(ax, init_left_hand, init_right_hand, init_left_hand, init_right_hand, box_pos, box_ori,
                         box_dims, v_initial, color='b')

    eigenvalues_right, eigenvectors_right = compute_force_ellipsoid(q_r, d_uar, d_lar, arm='right')
    eigenvalues_left, eigenvectors_left = compute_force_ellipsoid(q_l, d_ual, d_lal, arm='left')
    utils.plot_ellipsoid(ax, eigenvalues_right, eigenvectors_right, global_positions[5], 'b')
    utils.plot_ellipsoid(ax, eigenvalues_left, eigenvectors_left, global_positions[8], 'r')

    # initial_guess = [0.33, np.pi / 6, 0, - np.pi / 6, np.pi / 2, np.pi / 6, 0, - np.pi / 6]
    # initial_guess = [0, 0, 0, - np.pi / 2, 0, 0, 0, - np.pi / 2]
    initial_guess = [q_r[0], q_r[1], q_r[2], q_r[3], q_l[0], q_l[1], q_l[2], q_l[3]]

    # Constraints and bounds
    constraints = [
        {'type': 'eq', 'fun': relative_pose_constraint},  # Fixed relative pose
        {'type': 'eq', 'fun': joint_range_constraint}  # Joint range of motion
        # {'type': 'ineq', 'fun': right_arm_manipulability_constraint},
        # {'type': 'ineq', 'fun': left_arm_manipulability_constraint}
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
            print("Final Length of Manipulability Ellipsoid Along Z-Axis (Right Arm):",
                  manipulability_z_constraint(result.x[:4], d_uar, d_lar, arm='right'))
            print("Final Length of Manipulability Ellipsoid Along Z-Axis (Left Arm):",
                  manipulability_z_constraint(result.x[4:], d_ual, d_lal, arm='left'))
            print("Final REBA Score:", final_reba_score)
        else:
            print("Optimized Joint Angles:", optimized_angles)
            print("Final Length of Manipulability Ellipsoid Along Z-Axis (Right Arm):",
                  manipulability_z_constraint(result.x[:4], d_uar, d_lar, arm='right'))
            print("Final Length of Manipulability Ellipsoid Along Z-Axis (Left Arm):",
                  manipulability_z_constraint(result.x[4:], d_ual, d_lal, arm='left'))
            print(
                f"Optimization finished, but REBA score is above the acceptable threshold. Final REBA score: {final_reba_score}")
    else:
        print("Optimization failed:", result.message)

    global_positions = global_positions + np.array([0.5, 1, 0])
    update_box_and_hands(ax, global_positions[8], global_positions[5],
                         init_left_hand, init_right_hand, box_pos, box_ori, box_dims, v_initial, color='r')

    eigenvalues_right, eigenvectors_right = compute_force_ellipsoid(optimized_angles[:4], d_uar, d_lar, arm='right')
    utils.plot_ellipsoid(ax, eigenvalues_right, eigenvectors_right, global_positions[5], 'b')
    eigenvalues_left, eigenvectors_left = compute_force_ellipsoid(optimized_angles[4:], d_ual, d_lal, arm='left')
    utils.plot_ellipsoid(ax, eigenvalues_left, eigenvectors_left, global_positions[8], 'r')

    plt.show()

