# This is a Python script for bi-manual human-robot collaborative box carrying.
import numpy as np
import rospy
import math
import message_filters
from geometry_msgs.msg import PoseStamped, JointState
from scipy.optimize import minimize
import transformation as tsf
import utils


left_pub = rospy.Publisher('/panda_left/cartesain_command_tele', PoseStamped, queue_size=1)
right_pub = rospy.Publisher('/panda_right/cartesain_command_tele', PoseStamped, queue_size=1)
torso_pub = rospy.Publisher("/curi_torso/joint/cmd_vel", JointState, queue_size=10)

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


def transform_to_odom(torso_odom):
    return np.array([
        torso_odom.pose.pose.position.x,
        torso_odom.pose.pose.position.y,
        torso_odom.pose.pose.position.z
    ])


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


def is_significant_change(current_pose, last_pose, threshold=0.1):
    # Check if the change in position exceeds the threshold
    if last_pose is None:
        return True  # Always compute if last pose is undefined
    change = np.linalg.norm(current_pose - last_pose)
    return change > threshold


def optimized_joints(q):
    # Initial_guess, constraints and bounds
    initial_guess = q
    constraints = [
        {'type': 'eq', 'fun': relative_pose_constraint}  # Fixed relative pose
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


def multi_callback(sub_robot, sub_shouL, sub_shouR, sub_elbowL, sub_elbowR, sub_wristL, sub_wristR):
    global last_relative_displacement_wrists, last_object_pose

    robot_pose = transform_to_pose(sub_robot)
    shouL_pose = transform_to_pose(sub_shouL)
    shouR_pose = transform_to_pose(sub_shouR)
    elbowL_pose = transform_to_pose(sub_elbowL)
    elbowR_pose = transform_to_pose(sub_elbowR)
    wristL_pose = transform_to_pose(sub_wristL)
    wristR_pose = transform_to_pose(sub_wristR)
    # print(robot_pose)

    # Transform from optitrack frame to robot frame
    T_optitrack2robotbase = np.linalg.inv(
        tsf.transform_optitrack_origin_to_optitrack_robot(robot_pose) @ tsf.transform_optitrack_robot_to_robot_base())
    shouL_position_init = T_optitrack2robotbase[:3, :3] @ shouL_pose[:3] + T_optitrack2robotbase[:3, 3]
    shouR_position_init = T_optitrack2robotbase[:3, :3] @ shouR_pose[:3] + T_optitrack2robotbase[:3, 3]
    elbowL_position_init = T_optitrack2robotbase[:3, :3] @ elbowL_pose[:3] + T_optitrack2robotbase[:3, 3]
    elbowR_position_init = T_optitrack2robotbase[:3, :3] @ elbowR_pose[:3] + T_optitrack2robotbase[:3, 3]
    wristL_position_init = T_optitrack2robotbase[:3, :3] @ wristL_pose[:3] + T_optitrack2robotbase[:3, 3]
    wristR_position_init = T_optitrack2robotbase[:3, :3] @ wristR_pose[:3] + T_optitrack2robotbase[:3, 3]

    # Relative displacement of contact points
    relative_displacement_wrists = euclidean_distance(wristL_pose[:3], wristR_pose[:3])
    print("relative_pose_hand:", relative_displacement_wrists)

    if is_significant_change(relative_displacement_wrists, last_relative_displacement_wrists):
        # Body dimensions
        d_ual, d_uar, d_lal, d_lar = calculate_arm_dimensions(shouL_pose, elbowL_pose, wristL_pose, shouR_pose, elbowR_pose, wristR_pose)

        global d_ual, d_uar, d_lal, d_lar

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
        global initial_guess

        # Force Manipulability ellipsoid
        eigenvalues_right, eigenvectors_right = compute_force_ellipsoid(q_r, d_uar, d_lar, arm='right')
        eigenvalues_left, eigenvectors_left = compute_force_ellipsoid(q_l, d_ual, d_lal, arm='left')

        last_relative_displacement_wrists = relative_displacement_wrists
        joints_angles = optimized_joints(initial_guess)
        print("Optimized pose calculated:", joints_angles)
    else:
        print("Using last calculated pose:", last_object_pose)






if __name__ == '__main__':
    try:
        rospy.init_node('emo_hrc')
        subscriber_robot = message_filters.Subscriber('/vrpn_client_node/robot/pose', PoseStamped)
        subscriber_shouL = message_filters.Subscriber('/vrpn_client_node/shouL/pose', PoseStamped)
        subscriber_shouR = message_filters.Subscriber('/vrpn_client_node/shouR/pose', PoseStamped)
        subscriber_elbowL = message_filters.Subscriber('/vrpn_client_node/elbowL/pose', PoseStamped)
        subscriber_elbowR = message_filters.Subscriber('/vrpn_client_node/elbowR/pose', PoseStamped)
        subscriber_wristL = message_filters.Subscriber('/vrpn_client_node/wristL/pose', PoseStamped)
        subscriber_wristR = message_filters.Subscriber('/vrpn_client_node/wristR/pose', PoseStamped)

        # Skeleton Model
        skeleton_joint_name, skeleton_joint, skeleton_parent_indices, skeleton_joint_local_translation = utils.read_skeleton_motion(
            '/home/ubuntu/Ergo-Manip/data/demo_2_test_chenzui_only_optitrack2hotu.npy')
        skeleton_joint = skeleton_joint[450, :]
        global_positions, global_rotations = utils.forward_kinematics(skeleton_joint_local_translation,
                                                                      skeleton_joint, skeleton_parent_indices)

        sync = message_filters.ApproximateTimeSynchronizer(
            [subscriber_robot, subscriber_shouL, subscriber_shouR, subscriber_elbowL, subscriber_elbowR,
             subscriber_wristL, subscriber_wristR], 10, 1)
        print("Here we go!!!")
        sync.registerCallback(multi_callback)
        rospy.spin()

    except rospy.ROSInterruptException as e:
        print(e)
