# This is a Python script for uni-manual human-robot collaborative box carrying.
import numpy as np
import rospy
import math
import message_filters
from geometry_msgs.msg import PoseArray, PoseStamped, Quaternion, Pose, PoseWithCovariance, TwistWithCovariance
from sensor_msgs.msg import JointState
from nav_msgs.msg import Odometry
import transformation as tsf

import tf   
import tf2_ros
import geometry_msgs.msg
from scipy.spatial.transform import Rotation as R
import scipy.linalg as linalg


torso_cmd = JointState()
left_pub = rospy.Publisher('/panda_left/cartesain_command_tele', PoseStamped, queue_size=1)
right_pub = rospy.Publisher('/panda_right/cartesain_command_tele', PoseStamped, queue_size=1)
torso_pub = rospy.Publisher("/curi_torso/joint/cmd_vel", JointState, queue_size=10)


def generate_robot_ee_cmd(human_state):
    robot_ee_cmd = human_state
    robot_ee_cmd[0] = human_state[0] - 0.63
    robot_ee_cmd[1] = human_state[1] + 0.08
    return robot_ee_cmd


def transform_to_pose(pose_stamped):
    pose = [pose_stamped.pose.position.x, pose_stamped.pose.position.y, pose_stamped.pose.position.z,
            pose_stamped.pose.orientation.x, pose_stamped.pose.orientation.y, pose_stamped.pose.orientation.z,
            pose_stamped.pose.orientation.w]
    return np.array(pose)


def transform_to_joint(joint_state):
    joint = [joint_state.position[0], joint_state.position[1], joint_state.position[2]]
    return np.array(joint)


def transform_to_odom(torso_odom):
    odom = [torso_odom.pose.pose.position.x, torso_odom.pose.pose.position.y, torso_odom.pose.pose.position.z]
    return np.array(odom)


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


def rotate_mat(axis, radian):
    rot_matrix = linalg.expm(np.cross(np.eye(3), axis / linalg.norm(axis) * radian))
    return rot_matrix


def dot_product_angle(v1, v2):
    if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
        print("Zero magnitude vector!")
    else:
        vector_dot_product = np.dot(v1, v2)
        arccos = np.arccos(vector_dot_product / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        angle = np.degrees(arccos)
        return angle
    return 0


def generate_joint_cmd(box_position, robot_position, torso_state):
    vec_1 = np.array([1, 0, 0])
    vec_2 = np.array([box_position[0]-robot_position[0], box_position[2]-robot_position[2], 0])
    # vec_2 = np.array([1, 1, 0])

    joint1_vel_ = rospy.get_param("joint1_vel", 0.1)
    joint3_vel_ = rospy.get_param("joint3_vel", 0.1)

    if vec_2[1] >= 0:
        ori_angle = -math.radians(dot_product_angle(vec_1, vec_2))
    else:
        ori_angle = math.radians(dot_product_angle(vec_1, vec_2))
    print(ori_angle)

    torso_joint1 = torso_state[0]

    if abs(torso_joint1 - ori_angle) <= 0.03:
        torso_joint1_vel = 0
    elif (torso_joint1 - ori_angle) > 0.03:
        torso_joint1_vel = -joint1_vel_
    else:
        torso_joint1_vel = joint1_vel_
    
    p, R = tsf.transform_torso_base_to_torso_end(torso_state)
    torso_odom = p[2]

    if abs(2*torso_odom + 0.04 - box_position[1]) <= 0.02:
        torso_joint3_vel = 0
    elif (2*torso_odom + 0.04 - box_position[1]) > 0.02:
        torso_joint3_vel = joint3_vel_
    else:
        torso_joint3_vel = -joint3_vel_

    return torso_joint1_vel, torso_joint3_vel


def multi_callback(sub_robot, sub_left, sub_right, sub_torso):
    print("torso odom:")
    robot_pose = transform_to_pose(sub_robot)
    human_pose = transform_to_pose(sub_left)
    box_pose = transform_to_pose(sub_right)
    # print(sub_right)

    torso_joint = transform_to_joint(sub_torso)
    # torso_odom = transform_to_odom(sub_torsoodom)
    # print("torso odom:", torso_odom)

    # print("torso_joint",torso_joint)
    # print("human_pose",human_pose)

    torso_joint1_cmd, torso_joint3_cmd = generate_joint_cmd(box_pose[:3], robot_pose[:3], torso_joint)
    print("torso joint1 cmd:", torso_joint1_cmd)
    print("torso joint3 cmd:", torso_joint3_cmd)

    torso_cmd.velocity = [torso_joint1_cmd, 0, torso_joint3_cmd, 0, 0, 0, 0]

    right_pose = generate_robot_ee_cmd(human_pose)
    # print("right_pose", right_pose)
    left_pose = transform_to_pose(sub_left)


    # TO arm base frame
    T_MobileBaseToLeftArmBase, T_MobileBaseToRightArmBase = tsf.transform_robot_base_to_arm_base(torso_joint)


    T = np.linalg.inv(tsf.transform_optitrack_origin_to_optitrack_robot(robot_pose) @ 
        tsf.transform_optitrack_robot_to_robot_base() @ T_MobileBaseToRightArmBase) 
    right_position_new = T[:3, :3] @ right_pose[:3] + T[:3, 3]


    right_rotation_temp = R.from_quat(right_pose[3:])
    right_pose_orientation_matrix = right_rotation_temp.as_matrix()
    right_orientation_matrix_new = T[:3, :3] @ right_pose_orientation_matrix
    right_qua_temp = R.from_matrix(right_orientation_matrix_new)
    right_pose_tele_ori_qua = right_qua_temp.as_quat()
    opti_right_pose =  np.append(right_position_new, right_pose_tele_ori_qua)
    # print("right", opti_right_pose)



    # 输出6D pose绕x轴转90度，再绕y轴转-90度。
    right_rotation_temp = R.from_quat(opti_right_pose[3:])
    right_pose_orientation_matrix = right_rotation_temp.as_matrix()
    right_rot_matrix = rotate_mat(right_pose_orientation_matrix[:,0], math.pi/2)
    right_pose_tele_ori_mtx = np.dot(right_rot_matrix,right_pose_orientation_matrix)
    right_rot_matrix_2 = rotate_mat(right_pose_tele_ori_mtx[:,1], -math.pi/2)
    right_pose_tele_ori_mtx = np.dot(right_rot_matrix_2,right_pose_tele_ori_mtx)
    right_qua_temp = R.from_matrix(right_pose_tele_ori_mtx)
    right_pose_tele_ori_qua = right_qua_temp.as_quat()
    opti_right_pose_with_orientation =  np.append(opti_right_pose[:3], right_pose_tele_ori_qua)
    print("right", opti_right_pose_with_orientation)



    # opti_left_rviz.header.frame_id = 'panda_right_link0'
    # opti_left_rviz.header.stamp =  rospy.Time.now()
    # opti_left_rviz.child_frame_id = 'link_opti_left'
    # opti_left_rviz.transform.translation.x = opti_left_pose[0]
    # opti_left_rviz.transform.translation.y = opti_left_pose[1]
    # opti_left_rviz.transform.translation.z = opti_left_pose[2]
    # opti_left_rviz.transform.rotation.x    = opti_left_pose[3]
    # opti_left_rviz.transform.rotation.y    = opti_left_pose[4]
    # opti_left_rviz.transform.rotation.z    = opti_left_pose[5]
    # opti_left_rviz.transform.rotation.w    = opti_left_pose[6]

    opti_right_rviz_tempo.header.frame_id = 'panda_right_link0'
    opti_right_rviz_tempo.header.stamp =  rospy.Time.now()
    opti_right_rviz_tempo.child_frame_id = 'link_opti_right'
    opti_right_rviz_tempo.transform.translation.x = opti_right_pose[0]
    opti_right_rviz_tempo.transform.translation.y = opti_right_pose[1]
    opti_right_rviz_tempo.transform.translation.z = opti_right_pose[2]
    opti_right_rviz_tempo.transform.rotation.x    = opti_right_pose[3]
    opti_right_rviz_tempo.transform.rotation.y    = opti_right_pose[4]
    opti_right_rviz_tempo.transform.rotation.z    = opti_right_pose[5]
    opti_right_rviz_tempo.transform.rotation.w    = opti_right_pose[6]

    opti_right_rviz.header.frame_id = 'panda_right_link0'
    opti_right_rviz.header.stamp =  rospy.Time.now()
    opti_right_rviz.child_frame_id = 'link_opti_right_with_orientation'
    opti_right_rviz.transform.translation.x = opti_right_pose_with_orientation[0]
    opti_right_rviz.transform.translation.y = opti_right_pose_with_orientation[1]
    opti_right_rviz.transform.translation.z = opti_right_pose_with_orientation[2]
    opti_right_rviz.transform.rotation.x    = opti_right_pose_with_orientation[3]
    opti_right_rviz.transform.rotation.y    = opti_right_pose_with_orientation[4]
    opti_right_rviz.transform.rotation.z    = opti_right_pose_with_orientation[5]
    opti_right_rviz.transform.rotation.w    = opti_right_pose_with_orientation[6]

    

    br2.sendTransform([opti_right_rviz, opti_right_rviz_tempo])


    # left_pose_stamped = convert_to_pose_stamped(left_pose_new, sub_left.header.frame_id, sub_left.header.stamp)
    right_pose_stamped = convert_to_pose_stamped(opti_right_pose_with_orientation, "panda_right_link0", rospy.Time.now())
    # left_pub.publish(left_pose_stamped)
    # right_pub.publish(right_pose_stamped)
    # torso_pub.publish(torso_cmd)


if __name__ == '__main__':
    try:
        rospy.init_node('ucb_hrc')

        br = tf.TransformBroadcaster()
        br2 = tf2_ros.TransformBroadcaster()

        tele_right = geometry_msgs.msg.TransformStamped()
        tele_left = geometry_msgs.msg.TransformStamped()
        t_left = geometry_msgs.msg.TransformStamped()
        t_right = geometry_msgs.msg.TransformStamped()
        # opti_left_rviz = geometry_msgs.msg.TransformStamped()
        opti_right_rviz = geometry_msgs.msg.TransformStamped()
        opti_right_rviz_tempo = geometry_msgs.msg.TransformStamped()


        subscriber_robot = message_filters.Subscriber('/vrpn_client_node/robot/pose', PoseStamped)
        subscriber_left = message_filters.Subscriber('/vrpn_client_node/left/pose', PoseStamped)
        subscriber_right = message_filters.Subscriber('/vrpn_client_node/right/pose', PoseStamped)
        subscriber_torso = message_filters.Subscriber('/curi_torso/joint_states', JointState)
        # subscriber_odom = message_filters.Subscriber('/curi_torso/odom', Odometry)

        sync = message_filters.ApproximateTimeSynchronizer([subscriber_robot, subscriber_left, subscriber_right, subscriber_torso], 10, 1)
        print("Here we go!!!")
        sync.registerCallback(multi_callback)


        # get pose/joint information from ROS topic
        # human = ...
        # box = ...
        # robot = ...
        # torso = ...

        # calculate the command pos of the robot end effector in optitrack frame
        # robot_ee_cmd = generate_robot_ee_cmd(human, box)
        # robot_ee_cmd = human

        # transform to the arm frame of the CURI robot
        # T_L, T_R = tsf.transform_optitrack_origin_to_arm_base(robot, torso)
        # robot_ee_cmd_new = np.linalg.inv(T_L) * robot_ee_cmd
        # print(robot_ee_cmd_new)

        rospy.spin()
        
    except rospy.ROSInterruptException as e:
        print(e)



