import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from scipy.spatial.transform import Rotation as R


def read_skeleton_motion(file_path):
    skeleton = np.load(file_path, allow_pickle=True).item()
    skeleton_joint_name = skeleton['skeleton_tree']['node_names']
    skeleton_joint_local_translation = skeleton['skeleton_tree']['local_translation']['arr']
    skeleton_parent_indices = skeleton['skeleton_tree']['parent_indices']['arr']
    skeleton_joint = skeleton['rotation']['arr']
    return skeleton, skeleton_joint_name, skeleton_joint, skeleton_parent_indices, skeleton_joint_local_translation


def quaternion_to_rotation_matrix(quaternion):
    """Convert a quaternion to a rotation matrix."""
    return R.from_quat(quaternion).as_matrix()


def forward_kinematics(joint_positions, joint_rotations, parent_indices):
    """Convert all joint positions and orientations to a global reference frame."""
    num_joints = len(joint_positions)
    global_positions = np.zeros_like(joint_positions)
    global_rotations = [np.eye(3) for _ in range(num_joints)]

    for i in range(num_joints):
        if parent_indices[i] == -1:
            global_positions[i] = joint_positions[i]
            global_rotations[i] = quaternion_to_rotation_matrix(joint_rotations[i])
        else:
            parent_index = parent_indices[i]
            parent_rotation = global_rotations[parent_index]
            parent_position = global_positions[parent_index]

            rotation_matrix = quaternion_to_rotation_matrix(joint_rotations[i])
            global_rotations[i] = parent_rotation @ rotation_matrix
            global_positions[i] = parent_position + parent_rotation @ joint_positions[i]

    return global_positions, global_rotations


# def calculate_right_limb_lengths(global_positions):
#     """Calculate right limb lengths from joint positions."""
#     shoulder_index = 3
#     elbow_index = 4
#     wrist_index = 5
#
#     upper_arm_length = np.linalg.norm(global_positions[elbow_index] - global_positions[shoulder_index])
#     lower_arm_length = np.linalg.norm(global_positions[wrist_index] - global_positions[elbow_index])
#
#     return upper_arm_length, lower_arm_length
#
#
# def calculate_left_limb_lengths(global_positions):
#     """Calculate left limb lengths from joint positions."""
#     shoulder_index = 6
#     elbow_index = 7
#     wrist_index = 8
#
#     upper_arm_length = np.linalg.norm(global_positions[elbow_index] - global_positions[shoulder_index])
#     lower_arm_length = np.linalg.norm(global_positions[wrist_index] - global_positions[elbow_index])
#
#     return upper_arm_length, lower_arm_length

def calculate_manipulability(jacobian):
    """Calculate the manipulability ellipsoid from the Jacobian."""
    M = jacobian @ jacobian.T
    eigenvalues, eigenvectors = np.linalg.eigh(M)
    return eigenvalues, eigenvectors


def plot_skeleton(ax, global_positions, parent_indices):
    for i in range(len(global_positions)):
        if parent_indices[i] != -1:
            parent_pos = global_positions[parent_indices[i]]
            joint_pos = global_positions[i]
            ax.plot([parent_pos[0], joint_pos[0]], [parent_pos[1], joint_pos[1]], [parent_pos[2], joint_pos[2]], 'k-')


def plot_ellipsoid(ax, eigenvalues, eigenvectors, center, color):
    # Use only the last three eigenvalues and corresponding eigenvectors
    eigenvalues = eigenvalues[-3:]
    eigenvectors = eigenvectors[-3:, -3:]

    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v)) / 10
    y = np.outer(np.sin(u), np.sin(v)) / 10
    z = np.outer(np.ones_like(u), np.cos(v)) / 10

    for i in range(len(x)):
        for j in range(len(x)):
            [x[i, j], y[i, j], z[i, j]] = np.dot(eigenvectors,
                                                 np.array([x[i, j], y[i, j], z[i, j]]) * np.sqrt(eigenvalues))
            x[i, j] += center[0]
            y[i, j] += center[1]
            z[i, j] += center[2]

    ax.plot_surface(x, y, z, color=color, alpha=0.3)


def update(frame):
    ax.clear()
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(0, 2)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    joint_rotations = skeleton_joint[frame]

    global_positions, global_rotations = forward_kinematics(skeleton_joint_local_translation, joint_rotations,
                                                            skeleton_parent_indices)
    plot_skeleton(ax, global_positions, skeleton_parent_indices)

    right_hand_index = 5
    left_hand_index = 8

    jacobian_right = np.squeeze(jacobians_right[frame])
    eigenvalues_right, eigenvectors_right = calculate_manipulability(jacobian_right)
    plot_ellipsoid(ax, eigenvalues_right, eigenvectors_right, global_positions[right_hand_index], 'b')

    jacobian_left = np.squeeze(jacobians_left[frame])
    eigenvalues_left, eigenvectors_left = calculate_manipulability(jacobian_left)
    plot_ellipsoid(ax, eigenvalues_left, eigenvectors_left, global_positions[left_hand_index], 'r')


if __name__ == '__main__':
    skeleton, skeleton_joint_name, skeleton_joint, skeleton_parent_indices, skeleton_joint_local_translation = read_skeleton_motion(
        '/home/ubuntu/Ergo-Manip/data/demo_2_test_chenzui_only_optitrack2hotu.npy')

    skeleton_joint = skeleton_joint[::20, :, :]

    jacobians_left = np.load('/home/ubuntu/Ergo-Manip/data/jacobian_data/jacobians_left_hand_2.npy', allow_pickle=True)[::5]
    jacobians_right = np.load('/home/ubuntu/Ergo-Manip/data/jacobian_data/jacobians_right_hand_2.npy', allow_pickle=True)[::5]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ani = FuncAnimation(fig, update, frames=len(skeleton_joint), repeat=True)
    plt.show()
