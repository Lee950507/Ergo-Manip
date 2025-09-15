import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import main_opt_static as mos
import transformation as tsf
import utils


def trans_global2shoulder(shoulder, elbow, wrist, arm='left'):
    if arm == 'left':
        elbow_new = elbow - shoulder
        elbow_new = np.array([elbow_new[1], -elbow_new[0], elbow_new[2]])
        wrist_new = wrist - shoulder
        wrist_new = np.array([wrist_new[1], -wrist_new[0], wrist_new[2]])
    if arm == 'right':
        elbow_new = elbow - shoulder
        elbow_new = np.array([-elbow_new[1], -elbow_new[0], elbow_new[2]])
        wrist_new = wrist - shoulder
        wrist_new = np.array([-wrist_new[1], -wrist_new[0], wrist_new[2]])
    return elbow_new, wrist_new


# Load data
optimized_joints = np.load(
    "/home/ubuntu/Ergo-Manip/data/data_CSEF_comparison_0911/20250911_CSEF/posture 1/optimized_joint_angles.npy",
    allow_pickle=True)
recorded_data = np.load(
    "/home/ubuntu/Ergo-Manip/data/data_CSEF_comparison_0911/20250911_CSEF/posture 1/recorded_human_positions.npy",
    allow_pickle=True).item()

# Extract data components
timestamps = np.array(recorded_data['timestamps'])[::10]
shoulder_positions = np.array(recorded_data['shoulder_positions'])[::10]
elbow_positions = np.array(recorded_data['elbow_positions'])[::10]
wrist_positions = np.array(recorded_data['wrist_positions'])[::10]

sub_robot = np.array([-0.2195, 1.11462, 0, 0, 0, 0, 1])
T_optitrack2robotbase = np.linalg.inv(
    tsf.transform_optitrack_origin_to_optitrack_robot(
        sub_robot) @ tsf.transform_optitrack_robot_to_robot_base())

# Array to store the calculated joint angles from human data
calculated_joints = []
human_wrist_positions = []  # Store human wrist positions in shoulder frame
optimized_wrist_positions = []  # Store optimized wrist positions from FK
ergonomic_scores_calculated = []
ergonomic_scores_optimized = []

# Process each time step
for i in range(len(timestamps)):
    shoulder_pos = shoulder_positions[i, :3]
    elbow_pos = elbow_positions[i, :3]
    wrist_pos = wrist_positions[i, :3]

    # Transform to shoulder frame
    _, wrist_in_shoulder = trans_global2shoulder(shoulder_pos, elbow_pos, wrist_pos, arm='right')
    human_wrist_positions.append(wrist_in_shoulder)

    d_ual, d_uar, d_lal, d_lar = mos.calculate_arm_dimensions(shoulder_pos, elbow_pos,
                                                              wrist_pos, shoulder_pos,
                                                              elbow_pos, wrist_pos)
    p_elbowR_init, p_wristR_init = trans_global2shoulder(shoulder_pos, elbow_pos, wrist_pos, arm='right')

    q = mos.inverse_kinematics(p_elbowR_init, p_wristR_init, d_uar, d_lar)
    calculated_joints.append(q)

    # Calculate ergonomic scores
    if hasattr(utils, 'calculate_upper_limb_score_with_joint_angles'):
        ergonomic_scores_calculated.append(utils.calculate_upper_limb_score_with_joint_angles(q))

# Calculate wrist positions from optimized joint angles using forward kinematics
for i in range(len(optimized_joints)):
    if i < len(timestamps):
        shoulder_pos = T_optitrack2robotbase[:3, :3] @ shoulder_positions[i, :3] + T_optitrack2robotbase[:3, 3]
        elbow_pos = T_optitrack2robotbase[:3, :3] @ elbow_positions[i, :3] + T_optitrack2robotbase[:3, 3]
        wrist_pos = T_optitrack2robotbase[:3, :3] @ wrist_positions[i, :3] + T_optitrack2robotbase[:3, 3]
        d_ual, d_uar, d_lal, d_lar = mos.calculate_arm_dimensions(shoulder_pos, elbow_pos,
                                                                  wrist_pos, shoulder_pos,
                                                                  elbow_pos, wrist_pos)

        # Use forward kinematics to get optimized wrist position
        _, optimized_wrist_pos = mos.forward_kinematics(optimized_joints[i], d_uar, d_lar)
        optimized_wrist_positions.append(optimized_wrist_pos)

    # Calculate ergonomic scores for optimized joints
    if hasattr(utils, 'calculate_upper_limb_score_with_joint_angles'):
        ergonomic_scores_optimized.append(utils.calculate_upper_limb_score_with_joint_angles(optimized_joints[i]))

calculated_joints = np.array(calculated_joints)
human_wrist_positions = np.array(human_wrist_positions)
optimized_wrist_positions = np.array(optimized_wrist_positions)

# Create normalized time arrays (0-1) for both datasets
time_calc_normalized = np.linspace(0, 1, len(calculated_joints))
time_opt_normalized = np.linspace(0, 1, len(optimized_joints))

# Plot joint angle comparison with normalized time
plt.figure(figsize=(12, 8))
joint_names = ['Joint 1', 'Joint 2', 'Joint 3', 'Joint 4']

for i in range(4):  # Assuming 4 joints
    plt.subplot(2, 2, i + 1)
    plt.plot(time_calc_normalized, calculated_joints[:, i], 'b-', label='Human Joints')
    plt.plot(time_opt_normalized, optimized_joints[:, i], 'r-', label='Optimized Joints')
    plt.title(f'{joint_names[i]} Comparison')
    plt.xlabel('Normalized Time (0-1)')
    plt.ylabel('Joint Angle (rad)')
    plt.grid(True)
    plt.legend()

plt.tight_layout()
plt.savefig('joint_comparison_normalized.png', dpi=300)
plt.show()

# Plot wrist position trajectories in shoulder frame
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot human wrist trajectory
ax.plot(human_wrist_positions[:, 0],
        human_wrist_positions[:, 1],
        human_wrist_positions[:, 2],
        'b-', label='Human Wrist', linewidth=2)

# Plot optimized wrist trajectory
if len(optimized_wrist_positions) > 0:
    ax.plot(optimized_wrist_positions[:, 0],
            optimized_wrist_positions[:, 1],
            optimized_wrist_positions[:, 2],
            'r-', label='Optimized Wrist', linewidth=2)

# Set labels and legend
ax.set_xlabel('X Position (m)')
ax.set_ylabel('Y Position (m)')
ax.set_zlabel('Z Position (m)')
ax.set_title('Wrist Position Trajectories in Shoulder Frame')
ax.legend()

# Set equal aspect ratio for better visualization
max_range = np.array([
    np.max([human_wrist_positions[:, 0].max(), optimized_wrist_positions[:, 0].max()]) -
    np.min([human_wrist_positions[:, 0].min(), optimized_wrist_positions[:, 0].min()]),
    np.max([human_wrist_positions[:, 1].max(), optimized_wrist_positions[:, 1].max()]) -
    np.min([human_wrist_positions[:, 1].min(), optimized_wrist_positions[:, 1].min()]),
    np.max([human_wrist_positions[:, 2].max(), optimized_wrist_positions[:, 2].max()]) -
    np.min([human_wrist_positions[:, 2].min(), optimized_wrist_positions[:, 2].min()])
]).max() / 2.0

mid_x = (human_wrist_positions[:, 0].mean() + optimized_wrist_positions[:, 0].mean()) / 2
mid_y = (human_wrist_positions[:, 1].mean() + optimized_wrist_positions[:, 1].mean()) / 2
mid_z = (human_wrist_positions[:, 2].mean() + optimized_wrist_positions[:, 2].mean()) / 2
ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)

plt.savefig('wrist_trajectory_comparison.png', dpi=300)
plt.show()

# Plot ergonomic scores if available
if ergonomic_scores_calculated and ergonomic_scores_optimized:
    plt.figure(figsize=(10, 6))

    # Normalize time for ergonomic scores
    time_ergo_calc = np.linspace(0, 1, len(ergonomic_scores_calculated))
    time_ergo_opt = np.linspace(0, 1, len(ergonomic_scores_optimized))

    plt.plot(time_ergo_calc, ergonomic_scores_calculated, 'b-', label='Human Ergonomic Score')
    plt.plot(time_ergo_opt, ergonomic_scores_optimized, 'r-', label='Optimized Ergonomic Score')
    plt.title('Ergonomic Score Comparison')
    plt.xlabel('Normalized Time (0-1)')
    plt.ylabel('Ergonomic Score')
    plt.grid(True)
    plt.legend()
    plt.savefig('ergonomic_comparison_normalized.png', dpi=300)
    plt.show()

    # Print mean ergonomic scores
    mean_ergo_calculated = np.mean(ergonomic_scores_calculated)
    mean_ergo_optimized = np.mean(ergonomic_scores_optimized)
    print(f"Mean Human Ergonomic Score: {mean_ergo_calculated:.4f}")
    print(f"Mean Optimized Ergonomic Score: {mean_ergo_optimized:.4f}")
    print(
        f"Improvement: {(mean_ergo_optimized - mean_ergo_calculated):.4f} ({(mean_ergo_optimized - mean_ergo_calculated) / mean_ergo_calculated * 100:.2f}%)")