import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import reba_calculator.pose2degree.upper_arm_degree as UpperArm
import reba_calculator.degree2reba.upper_arm_reba_score as UpperArmReba
import reba_calculator.pose2degree.lower_arm_degree as LowerArm
import reba_calculator.degree2reba.lower_arm_reba_score as LowerArmReba
import reba_calculator.pose2degree.trunk_degree as Trunk
import reba_calculator.degree2reba.trunk_reba_score as TrunkReba
import manipulability_calculator as manip
import read_skeleton_motion as read
import plot

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

    plot.plot_skeleton(ax, global_positions, skeleton_parent_indices)

    # Define a structure to map joint types to their corresponding functions and indices
    joint_analysis = {
        'upper_arm': {
            'indices': [skeleton_joint_name.index('right_upper_arm'), skeleton_joint_name.index('left_upper_arm')],
            'degree_func': UpperArm.UpperArmDegree(global_positions).upper_arm_degrees,
            'reba_func': UpperArmReba.UAREBA,
            'score_func': lambda reba: reba.upper_arm_reba_score(),
            'color_func': plot.ua_get_color
        },
        'lower_arm': {
            'indices': [skeleton_joint_name.index('right_lower_arm'), skeleton_joint_name.index('left_lower_arm')],
            'degree_func': LowerArm.LADegrees(global_positions).lower_arm_degree,
            'reba_func': LowerArmReba.LAREBA,
            'score_func': lambda reba: reba.lower_arm_score(),
            'color_func': plot.la_get_color
        },
        'trunk': {
            'indices': [skeleton_joint_name.index('pelvis')],
            'degree_func': Trunk.TrunkDegree(global_positions, global_rotations).trunk_degrees,
            'reba_func': TrunkReba.TrunkREBA,
            'score_func': lambda reba: reba.trunk_reba_score(),
            'color_func': plot.trunk_get_color
        }
    }

    # Process each joint type with its corresponding REBA analysis and coloring
    for joint_type, settings in joint_analysis.items():
        degrees = settings['degree_func']()
        reba = settings['reba_func'](degrees)
        scores = settings['score_func'](reba)
        for i, idx in enumerate(settings['indices']):
            score = scores[i]
            color = settings['color_func'](score)
            ax.scatter(*global_positions[idx], color=color, s=100)  # Use scatter for single points

if __name__ == '__main__':
    skeleton_joint_name, skeleton_joint, skeleton_parent_indices, skeleton_joint_local_translation = read.read_skeleton_motion('/home/ubuntu/Ergo-Manip/data/demo_2_test_chenzui_only_optitrack2hotu.npy')
    skeleton_joint = skeleton_joint[::40, :, :]
    jacobians_left = np.load('/data/jacobian_data/jacobians_left_hand_2.npy', allow_pickle=True)[::10]
    jacobians_right = np.load('/data/jacobian_data/jacobians_right_hand_2.npy', allow_pickle=True)[::10]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ani = FuncAnimation(fig, update, frames=len(skeleton_joint), repeat=True)
    # ani.save('/home/ubuntu/Ergo-Manip/data/gif/demo_2_andrew.gif', writer='imagemagick', fps=3)
    plt.show()