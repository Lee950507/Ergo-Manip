import numpy as np
import read_skeleton_motion as read
import manipulability_calculator as manip
import plot
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation

def update(frame):
    ax.clear()
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(0, 2)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    joint_rotations = skeleton_joint[frame]
    global_positions, global_rotations = manip.forward_kinematics(skeleton_joint_local_translation, joint_rotations, skeleton_parent_indices)
    plot.plot_skeleton(ax, global_positions, skeleton_parent_indices)
    right_hand_index = 5
    left_hand_index = 8
    jacobian_right = np.squeeze(jacobians_right[frame])
    eigenvalues_right, eigenvectors_right = manip.calculate_manipulability(jacobian_right)
    plot.plot_ellipsoid(ax, eigenvalues_right, eigenvectors_right, global_positions[right_hand_index], 'b')
    jacobian_left = np.squeeze(jacobians_left[frame])
    eigenvalues_left, eigenvectors_left = manip.calculate_manipulability(jacobian_left)
    plot.plot_ellipsoid(ax, eigenvalues_left, eigenvectors_left, global_positions[left_hand_index], 'r')

if __name__ == '__main__':
    skeleton_joint_name, skeleton_joint, skeleton_parent_indices, skeleton_joint_local_translation = read.read_skeleton_motion('/home/ubuntu/Ergo-Manip/data/demo_2_test_chenzui_only_optitrack2hotu.npy')
    skeleton_joint = skeleton_joint[::40, :, :]
    jacobians_left = np.load('/data/jacobian_data/jacobians_left_hand_2.npy', allow_pickle=True)[::10]
    jacobians_right = np.load('/data/jacobian_data/jacobians_right_hand_2.npy', allow_pickle=True)[::10]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ani = FuncAnimation(fig, update, frames=len(skeleton_joint), repeat=True)
    plt.show()