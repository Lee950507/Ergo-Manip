import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
from scipy.optimize import minimize
import time
from matplotlib import cm


# Forward kinematics function
def forward_kinematics(q, d_ua, d_la):
    T1 = np.array([[math.cos(q[0]), 0, math.sin(q[0])], [0, 1, 0], [-math.sin(q[0]), 0, math.cos(q[0])]])
    T2 = np.array([[1, 0, 0], [0, math.cos(q[1]), -math.sin(q[1])], [0, math.sin(q[1]), math.cos(q[1])]])
    T3 = np.array([[math.cos(q[2]), -math.sin(q[2]), 0], [math.sin(q[2]), math.cos(q[2]), 0], [0, 0, 1]])
    T4 = np.array([[1, 0, 0], [0, math.cos(q[3]), -math.sin(q[3])], [0, math.sin(q[3]), math.cos(q[3])]])
    p_elbow = T1 @ T2 @ T3 @ d_ua
    p_hand = T1 @ T2 @ T3 @ (d_ua + T4 @ d_la)
    return p_elbow, p_hand


# Define ergonomic assessment function
def ergonomic_assessment(q, q_opt, weights):
    # Weighted Euclidean distance-based assessment
    return np.sqrt(np.sum(weights * (q - q_opt) ** 2))


# Define Signed Ergonomics Field (SEF)
def calculate_sef(q, q_opt, weights, comfort_threshold):
    ergo_value = ergonomic_assessment(q, q_opt, weights)
    return ergo_value - comfort_threshold


# Calculate gradient of SEF with respect to joint angles
def calculate_sef_gradient(q, q_opt, weights, comfort_threshold, delta=1e-6):
    # Numerical gradient calculation
    grad = np.zeros_like(q)
    sef_q = calculate_sef(q, q_opt, weights, comfort_threshold)

    for i in range(len(q)):
        q_plus = q.copy()
        q_plus[i] += delta
        sef_plus = calculate_sef(q_plus, q_opt, weights, comfort_threshold)
        grad[i] = (sef_plus - sef_q) / delta

    return grad


# Function to ensure joint limits are respected
def enforce_joint_limits(q, bounds):
    q_limited = np.copy(q)
    for i in range(len(q)):
        q_limited[i] = max(bounds[i][0], min(bounds[i][1], q[i]))
    return q_limited


# Plan trajectory using SEF-guided gradient descent
def plan_trajectory_with_sef(q_start, q_goal, q_opt, weights, comfort_threshold, bounds, max_steps=200, step_size=0.05):
    trajectory = [q_start]
    q_current = q_start.copy()

    # Parameters to balance goal attraction and ergonomic comfort
    alpha = 0.7  # Weight for moving toward goal
    beta = 0.3  # Weight for improving ergonomic comfort

    for step in range(max_steps):
        # Vector pointing toward goal
        goal_direction = q_goal - q_current
        goal_distance = np.linalg.norm(goal_direction)

        # If we're close enough to the goal, end trajectory
        if goal_distance < 0.05:
            trajectory.append(q_goal)
            break

        # Normalize goal direction
        if goal_distance > 0:
            goal_direction = goal_direction / goal_distance

        # Get SEF gradient (direction of increasing discomfort)
        sef_gradient = calculate_sef_gradient(q_current, q_opt, weights, comfort_threshold)
        sef_gradient_norm = np.linalg.norm(sef_gradient)

        # Normalize SEF gradient if non-zero
        if sef_gradient_norm > 0:
            sef_gradient = sef_gradient / sef_gradient_norm

        # Combine goal direction and negative SEF gradient (toward more comfort)
        combined_direction = alpha * goal_direction - beta * sef_gradient
        combined_norm = np.linalg.norm(combined_direction)

        if combined_norm > 0:
            # Normalize and take a step
            combined_direction = combined_direction / combined_norm
            q_next = q_current + step_size * combined_direction
        else:
            # If directions cancel out, prioritize goal direction
            q_next = q_current + step_size * goal_direction

        # Ensure joint limits are respected
        q_next = enforce_joint_limits(q_next, bounds)

        # Add to trajectory and update current position
        trajectory.append(q_next)
        q_current = q_next

        # Adaptive step size: reduce as we get closer to goal
        step_size = min(0.05, goal_distance * 0.2)

    return np.array(trajectory)


# Visualize trajectory in task space and SEF values
def visualize_trajectory_and_sef(trajectory, d_ua, d_la, q_opt, weights, comfort_threshold):
    # Calculate SEF values along the trajectory
    sef_values = []
    for q in trajectory:
        sef_value = calculate_sef(q, q_opt, weights, comfort_threshold)
        sef_values.append(sef_value)

    sef_values = np.array(sef_values)

    # Map joint space trajectory to task space
    elbow_positions = []
    hand_positions = []

    for q in trajectory:
        p_elbow, p_hand = forward_kinematics(q, d_ua, d_la)
        elbow_positions.append(p_elbow)
        hand_positions.append(p_hand)

    elbow_positions = np.array(elbow_positions)
    hand_positions = np.array(hand_positions)

    # Create a figure with two subplots: 3D trajectory and SEF values
    fig = plt.figure(figsize=(18, 8))

    # 3D Trajectory plot
    ax1 = fig.add_subplot(121, projection='3d')

    # Plot arm configurations at key points
    num_configs_to_show = min(10, len(trajectory))
    indices = np.linspace(0, len(trajectory) - 1, num_configs_to_show, dtype=int)

    # Colors based on SEF values
    norm = plt.Normalize(min(sef_values), max(sef_values))
    cmap = cm.RdBu_r

    # Plot hand trajectory with color based on SEF values
    for i in range(len(trajectory) - 1):
        segment_color = cmap(norm(sef_values[i]))
        ax1.plot(hand_positions[i:i + 2, 0], hand_positions[i:i + 2, 1], hand_positions[i:i + 2, 2],
                 color=segment_color, linewidth=3)

    # Different colors for arm configurations based on SEF
    for i, idx in enumerate(indices):
        q = trajectory[idx]
        p_elbow = elbow_positions[idx]
        p_hand = hand_positions[idx]

        # Color based on SEF value
        config_color = cmap(norm(sef_values[idx]))

        # Plot arm links
        ax1.plot([0, p_elbow[0], p_hand[0]],
                 [0, p_elbow[1], p_hand[1]],
                 [0, p_elbow[2], p_hand[2]],
                 color=config_color, alpha=0.7, linewidth=2)

        # Plot joints
        ax1.scatter([0], [0], [0], color=config_color, s=50)  # Shoulder
        ax1.scatter([p_elbow[0]], [p_elbow[1]], [p_elbow[2]], color=config_color, s=50)  # Elbow
        ax1.scatter([p_hand[0]], [p_hand[1]], [p_hand[2]], color=config_color, s=50)  # Hand

    # Mark start and end points
    ax1.scatter([hand_positions[0, 0]], [hand_positions[0, 1]], [hand_positions[0, 2]],
                color='blue', s=100, marker='o', label='Start')
    ax1.scatter([hand_positions[-1, 0]], [hand_positions[-1, 1]], [hand_positions[-1, 2]],
                color='green', s=100, marker='*', label='Goal')

    # Add shoulder position
    ax1.scatter([0], [0], [0], color='black', s=100, marker='^', label='Shoulder')

    # Set labels and title
    ax1.set_xlabel('X Position')
    ax1.set_ylabel('Y Position')
    ax1.set_zlabel('Z Position')
    ax1.set_title('SEF-Guided Trajectory in Task Space')

    # Equal aspect ratio
    max_range = max([
        hand_positions[:, 0].max() - hand_positions[:, 0].min(),
        hand_positions[:, 1].max() - hand_positions[:, 1].min(),
        hand_positions[:, 2].max() - hand_positions[:, 2].min()
    ])

    mid_x = (hand_positions[:, 0].max() + hand_positions[:, 0].min()) * 0.5
    mid_y = (hand_positions[:, 1].max() + hand_positions[:, 1].min()) * 0.5
    mid_z = (hand_positions[:, 2].max() + hand_positions[:, 2].min()) * 0.5

    ax1.set_xlim(mid_x - max_range, mid_x + max_range)
    ax1.set_ylim(mid_y - max_range, mid_y + max_range)
    ax1.set_zlim(mid_z - max_range, mid_z + max_range)

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax1, label='SEF Value')

    # Add legend
    ax1.legend()

    # SEF Values plot
    ax2 = fig.add_subplot(122)

    steps = np.arange(len(sef_values))

    # Plot SEF values with color gradient
    points = ax2.scatter(steps, sef_values, c=sef_values, cmap=cmap, s=50)

    # Connect points with lines
    ax2.plot(steps, sef_values, 'k-', alpha=0.5)

    # Add comfort threshold line
    ax2.axhline(y=0, color='r', linestyle='--', label='Comfort Threshold')

    # Find threshold crossing points
    crossings = []
    for i in range(1, len(sef_values)):
        if (sef_values[i - 1] < 0 and sef_values[i] >= 0) or (sef_values[i - 1] >= 0 and sef_values[i] < 0):
            crossings.append(i)

    # Mark threshold crossings
    if crossings:
        ax2.scatter([steps[i] for i in crossings], [sef_values[i] for i in crossings],
                    color='red', s=100, marker='x', label='Comfort Threshold Crossing')

    # Shade comfortable and uncomfortable regions
    for i in range(1, len(sef_values)):
        if sef_values[i] < 0:  # Comfortable region
            ax2.fill_between([i - 1, i], [sef_values[i - 1], sef_values[i]], 0,
                             color='blue', alpha=0.1)
        else:  # Uncomfortable region
            ax2.fill_between([i - 1, i], [sef_values[i - 1], sef_values[i]], 0,
                             color='red', alpha=0.1)

    # Mark start and end
    ax2.scatter([0], [sef_values[0]], color='blue', s=100, marker='o', label='Start')
    ax2.scatter([len(sef_values) - 1], [sef_values[-1]], color='green', s=100, marker='*', label='Goal')

    # Add annotations for key points
    min_sef_idx = np.argmin(sef_values)
    max_sef_idx = np.argmax(sef_values)

    ax2.annotate(f'Min SEF: {sef_values[min_sef_idx]:.2f}',
                 xy=(min_sef_idx, sef_values[min_sef_idx]),
                 xytext=(min_sef_idx - 10, sef_values[min_sef_idx] - 0.2),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))

    ax2.annotate(f'Max SEF: {sef_values[max_sef_idx]:.2f}',
                 xy=(max_sef_idx, sef_values[max_sef_idx]),
                 xytext=(max_sef_idx - 10, sef_values[max_sef_idx] + 0.2),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))

    # Set labels and title
    ax2.set_xlabel('Step Number')
    ax2.set_ylabel('SEF Value')
    ax2.set_title('SEF Values Along Trajectory')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend()

    # Add joint angle analysis
    joint_names = ['Shoulder Abduction/Adduction', 'Shoulder Flexion/Extension',
                   'Shoulder Internal/External Rotation', 'Elbow Flexion/Extension']

    # Create separate plot for joint angles
    fig2, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

    for i in range(4):
        axes[i].plot(steps, trajectory[:, i], 'b-', linewidth=2)
        axes[i].axhline(y=q_opt[i], color='g', linestyle='--', label='Optimal')
        axes[i].set_ylabel(f'q{i + 1}')
        axes[i].set_title(joint_names[i])
        axes[i].grid(True, linestyle='--', alpha=0.7)

    # Shade the SEF values on the bottom plot to see correlation
    ax_twin = axes[3].twinx()

    for i in range(1, len(sef_values)):
        if sef_values[i] < 0:  # Comfortable region
            ax_twin.fill_between([i - 1, i], 0, 1,
                                 color='blue', alpha=0.1, transform=ax_twin.get_xaxis_transform())
        else:  # Uncomfortable region
            ax_twin.fill_between([i - 1, i], 0, 1,
                                 color='red', alpha=0.1, transform=ax_twin.get_xaxis_transform())

    ax_twin.set_ylim(0, 1)
    ax_twin.set_ylabel('Comfort Region')

    axes[3].set_xlabel('Step Number')
    fig2.tight_layout()
    fig2.savefig('fig/joint_angles_trajectory.png', dpi=300)

    plt.tight_layout()
    fig.savefig('fig/trajectory_and_sef.png', dpi=300)
    plt.show()


def main():
    # Define model parameters
    d_ua = np.array([0, 0, -0.3])  # Upper arm length (shoulder to elbow)
    d_la = np.array([0, 0.25, 0])  # Lower arm length (elbow to hand)

    # Define optimal joint configuration
    q_opt = np.array([0, 0, 0, -np.pi / 6])

    # Define weights for joints (higher weights mean more important joints)
    weights = np.array([1.0, 1.5, 1.0, 2.0])

    # Define comfort threshold
    comfort_threshold = 0.1

    # Define joint limits
    bounds = [
        (-math.pi / 18, 17 * math.pi / 18),
        (-np.pi / 3, 17 * math.pi / 18),
        (-np.pi / 3, np.pi / 2),
        (-np.pi / 2, np.pi / 3)
    ]

    # Define start and goal configurations
    q_start = np.array([math.pi / 2, -math.pi / 6, math.pi / 4, -math.pi / 4])
    q_goal = q_opt
    # q_goal = np.array([math.pi / 4, math.pi / 4, 0, 0])

    print("Planning trajectory using SEF guidance...")
    trajectory = plan_trajectory_with_sef(q_start, q_goal, q_opt, weights, comfort_threshold, bounds)

    print(f"Trajectory planned with {len(trajectory)} steps")
    print("Visualizing trajectory and SEF values...")
    visualize_trajectory_and_sef(trajectory, d_ua, d_la, q_opt, weights, comfort_threshold)


if __name__ == "__main__":
    main()