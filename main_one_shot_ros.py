# This is a Python script for bi-manual human-robot collaborative box carrying.
import numpy as np
import math
from scipy.optimize import minimize
import transformation as tsf
import utils
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

import rospy
import message_filters
from geometry_msgs.msg import PoseStamped


last_relative_pose_wrists = None
last_object_pose = None


def transform_to_pose(pose_stamped):
    return np.array([
        pose_stamped.pose.position.x,
        pose_stamped.pose.position.y,
        pose_stamped.pose.position.z,
        pose_stamped.pose.orientation.x,
        pose_stamped.pose.orientation.y,
        pose_stamped.pose.orientation.z,
        pose_stamped.pose.orientation.w
    ])


def transform_to_joint(joint_state):
    return np.array(joint_state.position[:3])


def convert_to_pose_stamped(pose, frame_id, stamp):
    pose_stamped = PoseStamped()
    pose_stamped.header.frame_id = frame_id
    pose_stamped.header.stamp = stamp
    pose_stamped.pose.position.x = pose[0]
    pose_stamped.pose.position.y = pose[1]
    pose_stamped.pose.position.z = pose[2]
    # q = tf.transformations.quaternion_from_euler(pose[3], pose[4], pose[5])
    # pose_stamped.pose.orientation = Quaternion(*q)
    pose_stamped.pose.orientation.x = pose[3]
    pose_stamped.pose.orientation.y = pose[4]
    pose_stamped.pose.orientation.z = pose[5]
    pose_stamped.pose.orientation.w = pose[6]
    return pose_stamped


def euclidean_distance(point1, point2):
    return np.linalg.norm(point2 - point1)


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


def calculate_arm_dimensions(shouL_pose, elbowL_pose, wristL_pose, shouR_pose, elbowR_pose, wristR_pose):
    """Calculate arm dimensions based on joint positions."""
    length_left_upper_arm = euclidean_distance(shouL_pose[:3], elbowL_pose[:3])
    length_right_upper_arm = euclidean_distance(shouR_pose[:3], elbowR_pose[:3])
    length_left_lower_arm = euclidean_distance(elbowL_pose[:3], wristL_pose[:3])
    length_right_lower_arm = euclidean_distance(elbowR_pose[:3], wristR_pose[:3])

    return (
        np.array([0, 0, -length_left_upper_arm]),
        np.array([0, 0, -length_right_upper_arm]),
        np.array([0, length_left_lower_arm, 0]),
        np.array([0, length_right_lower_arm, 0])
    )


def quaternion_to_transformation_matrix(quaternion, translation):
    # Extract quaternion components
    qx, qy, qz, qw = quaternion

    # Compute the rotation matrix from the quaternion
    R = np.array([
        [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)]
    ])

    # Construct the transformation matrix
    T = np.eye(4)  # Create a 4x4 identity matrix
    T[:3, :3] = R  # Set the rotation part
    T[:3, 3] = translation  # Set the translation part

    return T


def quaternion_to_rotation_matrix(quaternion):
    # Extract quaternion components
    qx, qy, qz, qw = quaternion

    # Compute the rotation matrix from the quaternion
    R = np.array([
        [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)]
    ])

    return R


def compute_updated_object_pose(initial_pose, initial_points, updated_points):
    # Extract rotation and translation from the initial pose
    rotation_initial = initial_pose[:3, :3]
    translation_initial = initial_pose[:3, 3]

    # Compute the vector for the initial and updated second points
    vector_initial = initial_points[1] - initial_points[0]
    vector_updated = updated_points[1] - updated_points[0]

    # Normalize the vectors
    vector_initial_normalized = vector_initial / np.linalg.norm(vector_initial)
    vector_updated_normalized = vector_updated / np.linalg.norm(vector_updated)

    # Calculate the rotation required to align the initial vector with the updated vector
    rotation_vector = np.cross(vector_initial_normalized, vector_updated_normalized)
    angle_of_rotation = np.arccos(np.clip(np.dot(vector_initial_normalized, vector_updated_normalized), -1.0, 1.0))

    # Create a rotation matrix using the rotation vector and angle
    if np.linalg.norm(rotation_vector) > 1e-6:  # Avoid division by zero
        rotation = R.from_rotvec(rotation_vector * angle_of_rotation)
    else:
        rotation = R.identity()

    # Update the rotation of the initial pose
    updated_rotation = rotation * R.from_matrix(rotation_initial)

    # Calculate the vectors from the initial pose to the initial points
    vector_initial_1 = translation_initial - initial_points[0]

    # Apply the rotation to the initial vectors
    rotated_vector_1 = rotation.apply(vector_initial_1)
    translation_updated = updated_points[0] + rotated_vector_1

    # Construct the updated pose
    updated_pose = np.eye(4)
    updated_pose[:3, :3] = updated_rotation.as_matrix()
    updated_pose[:3, 3] = translation_updated  # Set the updated translation

    return updated_pose, rotation


def update_additional_points(initial_pose, updated_pose, robot_point, rotation):
    initial_position = initial_pose[:3, 3]
    updated_position = updated_pose[:3, 3]
    vector_initial = robot_point - initial_position
    rotated_vector_1 = rotation.apply(vector_initial)
    updated_point = updated_position + rotated_vector_1
    return updated_point


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


def middle_position_constraint(x):
    # Compute hand positions
    elbow_right, wrist_right = forward_kinematics(x[:4], d_uar, d_lar)
    elbow_left, wrist_left = forward_kinematics(x[4:], d_ual, d_lal)

    # Transformation
    global_positions_wrist_right = global_positions[3] + [-wrist_right[1],-wrist_right[0], wrist_right[2]]
    global_positions_wrist_left = global_positions[6] + [-wrist_left[1], wrist_left[0], wrist_left[2]]

    # Compute relative pose between hands
    middle_wrist = (global_positions_wrist_right + global_positions_wrist_left) / 2
    middle_shoulder = (global_positions[3] + global_positions[6]) / 2
    return middle_wrist - middle_shoulder


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


# Objective function: Minimize REBA score and ensure manipulability
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

    # reba_score = overall_arm_score_left + overall_arm_score_right
    reba_score = max(overall_arm_score_left, overall_arm_score_right)

    reba_threshold = 1.7  # Define the ergonomic threshold
    penalty_reba = 1000 * reba_score ** 2

    manip_right = manipulability_z_constraint(x[:4], d_uar, d_lar, arm='right')
    manip_left = manipulability_z_constraint(x[4:], d_ual, d_lal, arm='left')
    manip_threshold = 4

    # if reba_score <= reba_threshold and manip_right >= manip_threshold and manip_left >= manip_threshold:
    #     print("Conditions met for early stopping.")
    #     return 0

    penalty_manip_l = 100 * (manip_left - manip_threshold) ** 2
    penalty_manip_r = 100 * (manip_right - manip_threshold) ** 2

    regularization = 10 * np.sum(np.square(x[:4] - x[4:]))

    # Total cost: REBA score + penalty + regularization
    total_cost = penalty_reba + regularization + penalty_manip_r + penalty_manip_l

    return total_cost


def optimized_joints():
    # Initial_guess, constraints and bounds
    constraints = [
        {'type': 'eq', 'fun': relative_pose_constraint},  # Fixed relative pose
        # {'type': 'eq', 'fun': middle_position_constraint},
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

        if final_reba_score <= 50:  # Check if the score is acceptable
            print("Optimization successful!")
            print("Optimized Joint Angles:", optimized_angles)
            print("Final REBA Score:", final_reba_score)
        else:
            print("Optimized Joint Angles:", optimized_angles)
            print(
                f"Optimization finished, but REBA score is above the acceptable threshold. Final REBA score: {final_reba_score}")
    else:
        optimized_angles = initial_guess
        print("Optimization failed:", result.message)

    return optimized_angles


def optimizing_joint_angles(shouL_position_init, shouR_position_init, elbowL_position_init, elbowR_position_init, wristL_position_init, wristR_position_init):
    global last_relative_displacement_wrists, last_object_pose

    # Relative displacement of contact points
    relative_displacement_wrists = euclidean_distance(wristL_position_init[:3], wristR_position_init[:3])
    print("relative_pose_hand:", relative_displacement_wrists)

    last_relative_displacement_wrists = relative_displacement_wrists
    joints_angles = optimized_joints()
    print("Optimized pose calculated:", joints_angles)

    return joints_angles


def draw_skeleton_and_robot(global_positions, skeleton_parent_indices, robot_left_pose, robot_right_pose, object_pose, q_r, q_l):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    utils.plot_skeleton(ax, global_positions, skeleton_parent_indices, color='blue')

    eigenvalues_right, eigenvectors_right = compute_force_ellipsoid(q_r, d_uar, d_lar, arm='right')
    utils.plot_ellipsoid(ax, eigenvalues_right, eigenvectors_right, global_positions[5], 'b')
    eigenvalues_left, eigenvectors_left = compute_force_ellipsoid(q_l, d_ual, d_lal, arm='left')
    utils.plot_ellipsoid(ax, eigenvalues_left, eigenvectors_left, global_positions[8], 'r')

    # Draw robot left end-effector pose
    ax.scatter(robot_left_pose[0], robot_left_pose[1], robot_left_pose[2], color='red', label='Cobot Left EE')
    left_rotation_matrix = quaternion_to_rotation_matrix(robot_left_pose[3:7])
    draw_orientation(ax, robot_left_pose[:3], left_rotation_matrix, scale=0.2)

    # Draw robot right end-effector pose
    ax.scatter(robot_right_pose[0], robot_right_pose[1], robot_right_pose[2], color='green', label='Cobot Right EE')
    right_rotation_matrix = quaternion_to_rotation_matrix(robot_right_pose[3:7])
    draw_orientation(ax, robot_right_pose[:3], right_rotation_matrix, scale=0.2)

    ax.scatter(object_pose[0, 3], object_pose[1, 3], object_pose[2, 3], color='blue', label='Object')
    draw_orientation(ax, object_pose[:3, 3], object_pose[:3, :3], scale=0.2)

    # Set labels and title
    ax.set_xticks([0.5, 1.0, 1.5, 2.0, 2.5])
    ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax.set_zticks([0, 0.5, 1.0, 1.5, 2])
    ax.set_xlabel('X', fontsize=16)
    ax.set_ylabel('Y', fontsize=16)
    ax.set_zlabel('Z', fontsize=16)
    # ax.set_title('Skeleton and Robot End-Effector Poses', fontsize=16)
    ax.legend(fontsize=14)

    # Set font size for tick labels on each axis
    ax.tick_params(axis='both', which='major', labelsize=16)  # Major ticks
    ax.tick_params(axis='both', which='minor', labelsize=12)  # Minor ticks (if any)


def draw_orientation(ax, position, rotation_matrix, scale=0.3):
    # Transform the axes by the rotation matrix and plot
    for i in range(3):
        ax.quiver(position[0], position[1], position[2],
                   scale * rotation_matrix[0, i], scale * rotation_matrix[1, i], scale * rotation_matrix[2, i],
                   color=['black', 'black', 'black'][i], arrow_length_ratio=0.4)


# def multi_callback(sub_robot, sub_obejct, sub_shouL, sub_shouR, sub_elbowL, sub_elbowR, sub_wristL, sub_wristR):
#     global global_positions
#
#     sub_robot = transform_to_pose(sub_robot)
#     sub_object = transform_to_pose(sub_obejct)
#
#     sub_shouL = transform_to_pose(sub_shouL)
#     sub_shouR = transform_to_pose(sub_shouR)
#     sub_elbowL = transform_to_pose(sub_elbowL)
#     sub_elbowR = transform_to_pose(sub_elbowR)
#     sub_wristL = transform_to_pose(sub_wristL)
#     sub_wristR = transform_to_pose(sub_wristR)
#
#     # Transform from optitrack frame to robot frame
#     T_optitrack2robotbase = np.linalg.inv(
#         tsf.transform_optitrack_origin_to_optitrack_robot(
#             sub_robot) @ tsf.transform_optitrack_robot_to_robot_base())
#     shouL_position_init = T_optitrack2robotbase[:3, :3] @ sub_shouL[:3] + T_optitrack2robotbase[:3, 3]
#     shouR_position_init = T_optitrack2robotbase[:3, :3] @ sub_shouR[:3] + T_optitrack2robotbase[:3, 3]
#     elbowL_position_init = T_optitrack2robotbase[:3, :3] @ sub_elbowL[:3] + T_optitrack2robotbase[:3, 3]
#     elbowR_position_init = T_optitrack2robotbase[:3, :3] @ sub_elbowR[:3] + T_optitrack2robotbase[:3, 3]
#     wristL_position_init = T_optitrack2robotbase[:3, :3] @ sub_wristL[:3] + T_optitrack2robotbase[:3, 3]
#     wristR_position_init = T_optitrack2robotbase[:3, :3] @ sub_wristR[:3] + T_optitrack2robotbase[:3, 3]
#
#     # Transform object pose from optitrack frame to robot frame
#     object_pose_init = np.eye(4)
#     object_position_init = T_optitrack2robotbase[:3, :3] @ sub_object[:3] + T_optitrack2robotbase[:3, 3]
#     object_rotation_temp = R.from_quat(sub_object[3:])
#     object_pose_orientation_matrix = object_rotation_temp.as_matrix()
#     object_orientation_init = T_optitrack2robotbase[:3, :3] @ object_pose_orientation_matrix
#     object_pose_init[:3, :3] = object_orientation_init
#     object_pose_init[:3, 3] = object_position_init
#     print(object_pose_init)
#
#     # Skeleton Model
#     skeleton_joint_name, skeleton_joint, skeleton_parent_indices, skeleton_joint_local_translation = utils.read_skeleton_motion(
#         '/home/curi/Chenzui/Ergo-Manip-main/data/demo_data/demo_2_test_chenzui_only_optitrack2hotu.npy')
#     skeleton_joint = skeleton_joint[450, :]
#     global_positions, global_rotations = utils.forward_kinematics(skeleton_joint_local_translation,
#                                                                   skeleton_joint, skeleton_parent_indices)
#     global_positions[:, 2] = global_positions[:, 2] * 1.2  # Body Dimension Scaling
#
#     # Transformation
#     global_positions[4] = global_positions[3] + (elbowR_position_init - shouR_position_init)
#     global_positions[7] = global_positions[6] + (elbowL_position_init - shouL_position_init)
#     global_positions[5] = global_positions[3] + (wristR_position_init - shouR_position_init)
#     global_positions[8] = global_positions[6] + (wristL_position_init - shouL_position_init)
#
#     shou_center = (shouL_position_init + shouR_position_init) / 2
#     global_positions = global_positions + np.array([shou_center[0], shou_center[1], 0])
#     initial_wrist_position = np.array([global_positions[5],
#                                        global_positions[8]])
#     print("init", initial_wrist_position)
#
#     # Object & Robot initial EE pose
#     robot_left_pose_init = np.array([0.8, 0.4, 1.4, 0, 0, 0, 1])
#     robot_right_pose_init = np.array([0.8, -0.4, 1.4, 0, 0, 0, 1])
#
#     # Transformation
#     initial_robot_left_pose_matrix = quaternion_to_transformation_matrix(robot_left_pose_init[3:],
#                                                                          robot_left_pose_init[:3])
#     initial_robot_right_pose_matrix = quaternion_to_transformation_matrix(robot_right_pose_init[3:],
#                                                                           robot_right_pose_init[:3])
#
#     draw_skeleton_and_robot(global_positions, skeleton_parent_indices, robot_left_pose_init, robot_right_pose_init,
#                             object_pose_init)
#
#     # Optimizationopti
#     # optimized_angles = optimizing_joint_angles(global_positions[6], global_positions[3], global_positions[7],
#     #                                            global_positions[4], global_positions[8], global_positions[5])
#     optimized_angles = optimizing_joint_angles(shouL_position_init, shouR_position_init, elbowL_position_init,
#                                                elbowR_position_init, wristL_position_init, wristR_position_init)
#
#     updated_wrist_position = np.array([global_positions[5],
#                                        global_positions[8]])
#     print("update", updated_wrist_position)
#
#     object_updated_pose_matrix, object_updated_rotation = compute_updated_object_pose(object_pose_init,
#                                                                                       initial_wrist_position,
#                                                                                       updated_wrist_position)
#     print("Updated Pose:\n", object_updated_pose_matrix)
#
#     updated_robot_left_position = update_additional_points(object_pose_init, object_updated_pose_matrix,
#                                                            robot_left_pose_init[:3], object_updated_rotation)
#     updated_robot_right_position = update_additional_points(object_pose_init, object_updated_pose_matrix,
#                                                             robot_right_pose_init[:3], object_updated_rotation)
#
#     updated_robot_left_rotation = (
#             object_updated_rotation * R.from_matrix(initial_robot_left_pose_matrix[:3, :3])).as_quat()
#     updated_robot_right_rotation = (
#             object_updated_rotation * R.from_matrix(initial_robot_right_pose_matrix[:3, :3])).as_quat()
#
#     updated_robot_left_pose = np.append(updated_robot_left_position, updated_robot_left_rotation)
#     updated_robot_right_pose = np.append(updated_robot_right_position, updated_robot_right_rotation)
#     print(updated_robot_left_pose)
#     print(global_positions[8])
#     print(global_positions[5])
#
#     draw_skeleton_and_robot(global_positions, skeleton_parent_indices, updated_robot_left_pose,
#                             updated_robot_right_pose, object_updated_pose_matrix)


if __name__ == '__main__':
    # rospy.init_node('emo_hrc')
    # subscriber_robot = rospy.wait_for_message('/vrpn_client_node/robot/pose', PoseStamped)
    # subscriber_object = rospy.wait_for_message('/vrpn_client_node/object/pose', PoseStamped)
    # subscriber_shouL = rospy.wait_for_message ('/vrpn_client_node/shouL/pose', PoseStamped)
    # subscriber_shouR = rospy.wait_for_message('/vrpn_client_node/shouR/pose', PoseStamped)
    # subscriber_elbowL = rospy.wait_for_message('/vrpn_client_node/elbowL/pose', PoseStamped)
    # subscriber_elbowR = rospy.wait_for_message('/vrpn_client_node/elbowR/pose', PoseStamped)
    # subscriber_wristL = rospy.wait_for_message('/vrpn_client_node/wristL/pose', PoseStamped)
    # subscriber_wristR = rospy.wait_for_message('/vrpn_client_node/wristR/pose', PoseStamped)
    #
    # sub_robot = transform_to_pose(subscriber_robot)
    # sub_object = transform_to_pose(subscriber_object)
    #
    # sub_shouL = transform_to_pose(subscriber_shouL)
    # sub_shouR = transform_to_pose(subscriber_shouR)
    # sub_elbowL = transform_to_pose(subscriber_elbowL)
    # sub_elbowR = transform_to_pose(subscriber_elbowR)
    # sub_wristL = transform_to_pose(subscriber_wristL)
    # sub_wristR = transform_to_pose(subscriber_wristR)

    sub_robot = np.array([-0.2195, 1.11462, 0, 0, 0, 0, 1])
    sub_object = np.array([1.3, 1.3, 0, 0, 0, 0, 1])

    sub_shouL = np.array([2, 1.5, 0.25, 0, 0, 0, 1])
    sub_shouR = np.array([2, 1.5, -0.25, 0, 0, 0, 1])
    sub_elbowL = np.array([1.9, 1.3, 0.3, 0, 0, 0, 1])
    sub_elbowR = np.array([1.9, 1.3, -0.3, 0, 0, 0, 1])
    sub_wristL = np.array([1.8, 1.2, 0.3, 0, 0, 0, 1])
    sub_wristR = np.array([1.8, 1.4, -0.3, 0, 0, 0, 1])

    # Transform from optitrack frame to robot frame
    T_optitrack2robotbase = np.linalg.inv(
        tsf.transform_optitrack_origin_to_optitrack_robot(
            sub_robot) @ tsf.transform_optitrack_robot_to_robot_base())
    shouL_position_init = T_optitrack2robotbase[:3, :3] @ sub_shouL[:3] + T_optitrack2robotbase[:3, 3]
    shouR_position_init = T_optitrack2robotbase[:3, :3] @ sub_shouR[:3] + T_optitrack2robotbase[:3, 3]
    elbowL_position_init = T_optitrack2robotbase[:3, :3] @ sub_elbowL[:3] + T_optitrack2robotbase[:3, 3]
    elbowR_position_init = T_optitrack2robotbase[:3, :3] @ sub_elbowR[:3] + T_optitrack2robotbase[:3, 3]
    wristL_position_init = T_optitrack2robotbase[:3, :3] @ sub_wristL[:3] + T_optitrack2robotbase[:3, 3]
    wristR_position_init = T_optitrack2robotbase[:3, :3] @ sub_wristR[:3] + T_optitrack2robotbase[:3, 3]
    print(shouL_position_init, shouR_position_init)

    # Body dimensions
    d_ual, d_uar, d_lal, d_lar = calculate_arm_dimensions(shouL_position_init, elbowL_position_init, wristL_position_init, shouR_position_init, elbowR_position_init, wristR_position_init)

    # Transform from robot frame to each shoulder frame
    p_elbowL_init = elbowL_position_init - shouL_position_init
    p_elbowL_init = np.array([p_elbowL_init[1], -p_elbowL_init[0], p_elbowL_init[2]])
    p_wristL_init = wristL_position_init - shouL_position_init
    p_wristL_init = np.array([p_wristL_init[1], -p_wristL_init[0], p_wristL_init[2]])

    p_elbowR_init = elbowR_position_init - shouR_position_init
    p_elbowR_init = np.array([-p_elbowR_init[1], -p_elbowR_init[0], p_elbowR_init[2]])
    p_wristR_init = wristR_position_init - shouR_position_init
    p_wristR_init = np.array([-p_wristR_init[1], -p_wristR_init[0], p_wristR_init[2]])

    # Inverse kinematics for joint angles
    q_l = inverse_kinematics(p_elbowL_init, p_wristL_init, d_ual, d_lal)
    q_r = inverse_kinematics(p_elbowR_init, p_wristR_init, d_uar, d_lar)
    initial_guess = np.concatenate((q_r, q_l))

    # Transform object pose from optitrack frame to robot frame
    object_pose_init = np.eye(4)
    object_position_init = T_optitrack2robotbase[:3, :3] @ sub_object[:3] + T_optitrack2robotbase[:3, 3]
    object_rotation_temp = R.from_quat(sub_object[3:])
    object_pose_orientation_matrix = object_rotation_temp.as_matrix()
    object_orientation_init = T_optitrack2robotbase[:3, :3] @ object_pose_orientation_matrix
    object_pose_init[:3, :3] = object_orientation_init
    object_pose_init[:3, 3] = object_position_init
    print(object_pose_init)

    # Skeleton Model
    skeleton_joint_name, skeleton_joint, skeleton_parent_indices, skeleton_joint_local_translation = utils.read_skeleton_motion(
        '/home/curi/Chenzui/Ergo-Manip-main/data/demo_data/demo_2_test_chenzui_only_optitrack2hotu.npy')
    skeleton_joint = skeleton_joint[450, :]
    global_positions, global_rotations = utils.forward_kinematics(skeleton_joint_local_translation,
                                                                  skeleton_joint, skeleton_parent_indices)
    global_positions[:, 2] = global_positions[:, 2] * 1.2  # Body Dimension Scaling

    # Transformation
    global_positions[4] = global_positions[3] + (elbowR_position_init - shouR_position_init)
    global_positions[7] = global_positions[6] + (elbowL_position_init - shouL_position_init)
    global_positions[5] = global_positions[3] + (wristR_position_init - shouR_position_init)
    global_positions[8] = global_positions[6] + (wristL_position_init - shouL_position_init)

    shou_center = (shouL_position_init + shouR_position_init) / 2
    global_positions = global_positions + np.array([shou_center[0], shou_center[1], 0])
    initial_wrist_position = np.array([global_positions[5],
                                       global_positions[8]])
    print("init", initial_wrist_position)

    # Object & Robot initial EE pose
    robot_left_position_init = object_position_init + np.array([-0.8, 0.3, 0.2])
    robot_right_position_init = object_position_init + np.array([-0.8, -0.3, 0.2])
    # robot_left_pose_init = np.array([0.8, 0.4, 1.4, 0, 0, 0, 1])
    # robot_right_pose_init = np.array([0.8, -0.4, 1.4, 0, 0, 0, 1])

    robot_left_rotation_matrix_init = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    robot_right_rotation_matrix_init = np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]])

    robot_left_pose_matrix_init = np.r_[np.c_[robot_left_rotation_matrix_init, robot_left_position_init.T], np.array([[0, 0, 0, 1]])]
    robot_right_pose_matrix_init = np.r_[np.c_[robot_right_rotation_matrix_init, robot_right_position_init.T], np.array([[0, 0, 0, 1]])]

    robot_left_pose_init = np.append(robot_left_position_init, R.from_matrix(robot_left_rotation_matrix_init).as_quat())
    robot_right_pose_init = np.append(robot_right_position_init, R.from_matrix(robot_right_rotation_matrix_init).as_quat())

    # initial_robot_left_rotation_matrix = np.array([[1, 0, 0, 0.6], [0, 0, -1, 0.3], [0, 1, 0, -0.15], [0, 0, 0, 1]])
    # initial_robot_right_rotation_matrix = np.array([[-1, 0, 0, 0.6], [0, 0, 1, -0.3], [0, 1, 0, -0.15], [0, 0, 0, 1]])

    # initial_robot_left_pose_matrix = quaternion_to_transformation_matrix(robot_left_pose_init[3:],
    #                                                                      robot_left_pose_init[:3])
    # initial_robot_right_pose_matrix = quaternion_to_transformation_matrix(robot_right_pose_init[3:],
    #                                                                       robot_right_pose_init[:3])

    draw_skeleton_and_robot(global_positions, skeleton_parent_indices, robot_left_pose_init, robot_right_pose_init,
                            object_pose_init, q_r, q_l)

    # Optimization
    # optimized_angles = optimizing_joint_angles(global_positions[6], global_positions[3], global_positions[7],
    #                                            global_positions[4], global_positions[8], global_positions[5])
    optimized_angles = optimizing_joint_angles(shouL_position_init, shouR_position_init, elbowL_position_init,
                                               elbowR_position_init, wristL_position_init, wristR_position_init)

    updated_wrist_position = np.array([global_positions[5],
                                       global_positions[8]])
    print("update", updated_wrist_position)

    object_updated_pose_matrix, object_updated_rotation = compute_updated_object_pose(object_pose_init,
                                                                                      initial_wrist_position,
                                                                                      updated_wrist_position)
    print("Updated Pose:\n", object_updated_pose_matrix)

    updated_robot_left_position = update_additional_points(object_pose_init, object_updated_pose_matrix,
                                                           robot_left_pose_matrix_init[:3, 3], object_updated_rotation)
    updated_robot_right_position = update_additional_points(object_pose_init, object_updated_pose_matrix,
                                                            robot_right_pose_matrix_init[:3, 3], object_updated_rotation)

    updated_robot_left_rotation = (
            object_updated_rotation * R.from_matrix(robot_left_pose_matrix_init[:3, :3])).as_quat()
    updated_robot_right_rotation = (
            object_updated_rotation * R.from_matrix(robot_right_pose_matrix_init[:3, :3])).as_quat()

    updated_robot_left_pose = np.append(updated_robot_left_position, updated_robot_left_rotation)
    updated_robot_right_pose = np.append(updated_robot_right_position, updated_robot_right_rotation)
    # print("left_qua", updated_robot_left_pose)

    updated_robot_left_pose_matrix = quaternion_to_transformation_matrix(updated_robot_left_rotation, updated_robot_left_position)
    updated_robot_right_pose_matrix = quaternion_to_transformation_matrix(updated_robot_right_rotation, updated_robot_right_position)

    print("initial_robot_left_pose", robot_left_pose_matrix_init)
    print("initial_robot_right_pose", robot_right_pose_matrix_init)
    print("updated_robot_left_pose", updated_robot_left_pose_matrix)
    print("updated_robot_right_pose", updated_robot_right_pose_matrix)
    # print(updated_robot_left_pose)
    # print(global_positions[8])
    # print(global_positions[5])

    draw_skeleton_and_robot(global_positions, skeleton_parent_indices, updated_robot_left_pose,
                            updated_robot_right_pose, object_updated_pose_matrix, optimized_angles[:4], optimized_angles[4:])

    # Show the plot
    plt.show()


    # torso joint states
    # position: [0.0023761664051562548, -0.853187084197998, 0.5779626369476318]










