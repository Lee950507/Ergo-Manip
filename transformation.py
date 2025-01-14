import numpy as np
import math
from scipy.spatial.transform import Rotation as R


def quaternion2matrix(q):
    x = q[0]
    y = q[1]
    z = q[2]
    w = q[3]

    return np.array([[1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
                     [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
                     [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y]])


def transform_optitrack_origin_to_optitrack_robot(robot_state):
    translation = robot_state[:3]
    R = quaternion2matrix(robot_state[3:7])
    T_OptitrackOriginToOptitrackRobot = np.r_[np.c_[R, translation.T], np.array([[0, 0, 0, 1]])]

    return T_OptitrackOriginToOptitrackRobot


def transform_optitrack_robot_to_robot_base():
    # translation = np.array([[-0.2195, 0, 1.11462]])
    R = np.linalg.inv(np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]))
    translation = np.array([[0.2195, -1.11462, 0]])
    # R = np.array([[0, 0, 1], [-1, 0, 0], [0, 1, 0]])
    T_OptitrackRobotToRobotBase = np.r_[np.c_[R, translation.T], np.array([[0, 0, 0, 1]])]

    return T_OptitrackRobotToRobotBase


def transform_torso_base_to_torso_end(theta):
    p = np.empty([3])
    p[0] = 0.25 * (math.sin(theta[1]) + math.cos(theta[1] + theta[2])) * math.cos(theta[0])
    p[1] = 0.25 * (math.sin(theta[1]) + math.cos(theta[1] + theta[2])) * math.sin(theta[0])
    p[2] = 0.215 + 0.25 * (math.cos(theta[1]) - math.sin(theta[1] + theta[2]))
    R = np.array(
        [[math.cos(theta[0]), -math.sin(theta[0]), 0], [math.sin(theta[0]), math.cos(theta[0]), 0],
         [0, 0, 1]])

    return p, R


def transform_torso_end_to_arm_base(data, vec):
    R_1 = np.array([[math.cos(data[0]), 0, math.sin(data[0])], [0, 1, 0],
                    [-math.sin(data[0]), 0, math.cos(data[0])]])
    R_2 = np.array([[1, 0, 0], [0, math.cos(data[1]), -math.sin(data[1])],
                    [0, math.sin(data[1]), math.cos(data[1])]])
    R_3 = np.array(
        [[math.cos(data[2]), -math.sin(data[2]), 0], [math.sin(data[2]), math.cos(data[2]), 0],
         [0, 0, 1]])
    R_4 = np.array([[math.cos(data[3]), 0, math.sin(data[3])], [0, 1, 0],
                    [-math.sin(data[3]), 0, math.cos(data[3])]])
    T = np.around(np.array([R_4 @ R_1 @ R_2 @ R_3]).reshape(-1, 3), decimals=6)
    T = np.r_[np.c_[T, vec.T], np.array([[0, 0, 0, 1]])]

    return T


def transform_robot_base_to_arm_base(torso_state):
    # Fixed Base Transformation
    T_MobileBaseToTorsoBase = np.array([[1, 0, 0, 0.2375], [0, 1, 0, 0], [0, 0, 1, 0.53762], [0, 0, 0, 1]])
    T_TorsoEndToLeftArmBase = transform_torso_end_to_arm_base(
        np.array([math.pi / 2, -math.pi / 4, -math.pi / 6, -math.pi / 18]), np.array([[-0.08537, 0.07009, 0.2535]]))
    T_TorsoEndToRightArmBase = transform_torso_end_to_arm_base(
        np.array([math.pi / 2, math.pi / 4, math.pi / 6, -math.pi / 18]), np.array([[-0.08537, -0.07009, 0.2535]]))

    # Current Joint States of Torso
    p, R = transform_torso_base_to_torso_end(torso_state)
    T_TorsoBaseToTorsoEnd = np.r_[np.c_[R, p.T], np.array([[0, 0, 0, 1]])]

    # Transformation Matrix from CURI base to Left/Right Arm Base under the torso configuration
    T_MobileBaseToLeftArmBase = T_MobileBaseToTorsoBase @ T_TorsoBaseToTorsoEnd @ T_TorsoEndToLeftArmBase
    # print(T_CURIBaseToLeftArmBase)

    T_MobileBaseToRightArmBase = T_MobileBaseToTorsoBase @ T_TorsoBaseToTorsoEnd @ T_TorsoEndToRightArmBase
    # print(T_CURIBaseToRightArmBase)

    return T_MobileBaseToLeftArmBase, T_MobileBaseToRightArmBase


def transform_optitrack_origin_to_arm_base(robot_state, torso_state):
    T_RobotBaseToLeftArmBase, T_RobotBaseToRightArmBase = transform_robot_base_to_arm_base(torso_state)
    T_OptitrackOriginToLeftArmBase = transform_optitrack_origin_to_optitrack_robot(
        robot_state) @ transform_optitrack_robot_to_robot_base() @ T_RobotBaseToLeftArmBase
    T_OptitrackOriginToRightArmBase = transform_optitrack_origin_to_optitrack_robot(
        robot_state) @ transform_optitrack_robot_to_robot_base() @ T_RobotBaseToRightArmBase

    return T_OptitrackOriginToLeftArmBase, T_OptitrackOriginToRightArmBase


def quart_to_rpy(quart):
    roll = math.atan2(2 * (quart[3] * quart[0]  + quart[1] * quart[2]), 1 - 2 * (quart[0]  * quart[0]  + quart[1] * quart[1]))
    pitch = math.asin(2 * (quart[3] * quart[1] - quart[0] * quart[2]))
    yaw = math.atan2(2 * (quart[3] * quart[2] + quart[0] * quart[1]), 1 - 2 * (quart[2] * quart[2] + quart[1] * quart[1]))
    rpy = np.array([roll, pitch, yaw])
    return rpy


def pose_generation_under_robot_arm_base(left_pose, right_pose, joint_state):
    T_MobileBaseToLeftArmBase, T_MobileBaseToRightArmBase = transform_robot_base_to_arm_base(joint_state)
        
    T_r = np.linalg.inv(T_MobileBaseToRightArmBase) 
    right_position_arm_base = T_r[:3, :3] @ right_pose[:3] + T_r[:3, 3]

    right_rotation_temp = R.from_quat(right_pose[3:])
    right_pose_orientation_matrix = right_rotation_temp.as_matrix()
    right_orientation_matrix_new = T_r[:3, :3] @ right_pose_orientation_matrix
    right_qua_temp = R.from_matrix(right_orientation_matrix_new)
    right_orientation_arm_base = right_qua_temp.as_quat()

    right_pose_arm_base =  np.append(right_position_arm_base, right_orientation_arm_base)
    # print("right", right_pose_arm_base)

    T_l = np.linalg.inv(T_MobileBaseToLeftArmBase) 
    left_position_arm_base = T_l[:3, :3] @ left_pose[:3] + T_l[:3, 3]

    left_rotation_temp = R.from_quat(left_pose[3:])
    left_pose_orientation_matrix = left_rotation_temp.as_matrix()
    left_orientation_matrix_new = T_l[:3, :3] @ left_pose_orientation_matrix
    left_qua_temp = R.from_matrix(left_orientation_matrix_new)
    left_orientation_arm_base = left_qua_temp.as_quat()

    left_pose_arm_base =  np.append(left_position_arm_base, left_orientation_arm_base)
    # print("left", left_pose_arm_base)
    
    return left_pose_arm_base, right_pose_arm_base


def generate_pose_under_robot_base(opti_robot, opti_box):
    T_optiorigin2robotbase = np.linalg.inv(transform_optitrack_origin_to_optitrack_robot(opti_robot) @ transform_optitrack_robot_to_robot_base())
    box_position = T_optiorigin2robotbase[:3, :3] @ opti_box[:3] + T_optiorigin2robotbase[:3, 3]

    box_rotation_temp = R.from_quat(opti_box[3:])
    box_pose_orientation_matrix = box_rotation_temp.as_matrix()
    box_orientation_matrix_new = T_optiorigin2robotbase[:3, :3] @ box_pose_orientation_matrix
    box_qua_temp = R.from_matrix(box_orientation_matrix_new)
    box_orientation = box_qua_temp.as_quat()

    box_pose =  np.append(box_position, box_orientation)
    # print("box_pose", box_pose)
    return box_pose


if __name__ == '__main__':
    ori_arm_base = np.array([0.75116071, -0.52792446, 0.0067883143, -0.396241422])

    T_MobileBaseToLeftArmBase, T_MobileBaseToRightArmBase = transform_robot_base_to_arm_base(np.array([-0.0248294342, -0.7418483, 0.5442205]))
        
    T = T_MobileBaseToRightArmBase
    # T = np.linalg.inv(T_MobileBaseToRightArmBase)

    right_rotation_temp = R.from_quat(ori_arm_base)
    right_pose_orientation_matrix = right_rotation_temp.as_matrix()
    right_orientation_matrix_new = T[:3, :3] @ right_pose_orientation_matrix
    right_qua_temp = R.from_matrix(right_orientation_matrix_new)
    right_orientation_robot_base = right_qua_temp.as_quat()
    print(right_orientation_robot_base)
