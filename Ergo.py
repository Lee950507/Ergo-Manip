import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import reba_calculator.pose2degree.upper_arm_degree as UpperArm
import reba_calculator.degree2reba.upper_arm_reba_score as UpperArmReba
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
    ua_d = UpperArm.UpperArmDegree(global_positions)
    ua_degrees = ua_d.upper_arm_degrees()
    ua_reba = UpperArmReba.UAREBA(ua_degrees)
    ua_scores = ua_reba.upper_arm_reba_score()

    plot.plot_skeleton(ax, global_positions, skeleton_parent_indices)

    # Color the upper arm joints based on the REBA scores
    upper_arm_indices = [skeleton_joint_name.index('right_upper_arm'), skeleton_joint_name.index('left_upper_arm')]
    for idx in upper_arm_indices:
        score = ua_scores
        color = plot.get_color(score[0])
        ax.scatter(*global_positions[idx], color=color, s=100)  # Use scatter for single points


if __name__ == '__main__':
    # Read and process data
    skeleton_joint_name, skeleton_joint, skeleton_parent_indices, skeleton_joint_local_translation = read.read_skeleton_motion(
        '/home/ubuntu/Ergo-Manip/data/demo_2_test_chenzui_only_optitrack2hotu.npy')
    skeleton_joint = skeleton_joint[::10]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ani = FuncAnimation(fig, update, frames=len(skeleton_joint), repeat=True)
    plt.show()