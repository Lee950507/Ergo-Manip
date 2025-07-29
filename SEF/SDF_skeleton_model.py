import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import math
from scipy.optimize import minimize
import time


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


# Inverse Kinematics using optimization
def inverse_kinematics(target_pos, d_ua, d_la, q_init=None, bounds=None):
    if q_init is None:
        q_init = np.array([0, 0, 0, 0])

    if bounds is None:
        bounds = [
            (-math.pi / 18, 17 * math.pi / 18),
            (-np.pi / 3, 17 * math.pi / 18),
            (-np.pi / 3, np.pi / 2),
            (-np.pi / 2, np.pi / 3)
        ]

    def objective(q):
        _, p_hand = forward_kinematics(q, d_ua, d_la)
        return np.sum((p_hand - target_pos) ** 2)

    result = minimize(objective, q_init, method='L-BFGS-B', bounds=bounds, options={'gtol': 1e-6, 'ftol': 1e-6})
    return result.x if result.success else None


# Create joint space SEF
def create_joint_space_sef(q_opt, weights, comfort_threshold, resolution=20):
    # Define joint limits based on optimal configuration to center it
    joint_range_width = [
        np.pi,  # range for q1
        np.pi,  # range for q2
        np.pi,  # range for q3
        np.pi,  # range for q4
    ]

    # Create 2D slices of the 4D joint space for visualization
    # Center the ranges on optimal configuration
    q1_range = np.linspace(q_opt[0] - joint_range_width[0] / 2, q_opt[0] + joint_range_width[0] / 2, resolution)
    q2_range = np.linspace(q_opt[1] - joint_range_width[1] / 2, q_opt[1] + joint_range_width[1] / 2, resolution)
    Q1, Q2 = np.meshgrid(q1_range, q2_range)

    # Calculate SEF for q1-q2 slice
    SEF_q1_q2 = np.zeros_like(Q1)
    for i in range(resolution):
        for j in range(resolution):
            q = np.array([Q1[i, j], Q2[i, j], q_opt[2], q_opt[3]])
            SEF_q1_q2[i, j] = calculate_sef(q, q_opt, weights, comfort_threshold)

    # Create q2-q3 slice
    q2_range = np.linspace(q_opt[1] - joint_range_width[1] / 2, q_opt[1] + joint_range_width[1] / 2, resolution)
    q3_range = np.linspace(q_opt[2] - joint_range_width[2] / 2, q_opt[2] + joint_range_width[2] / 2, resolution)
    Q2, Q3 = np.meshgrid(q2_range, q3_range)

    # Calculate SEF for q2-q3 slice
    SEF_q2_q3 = np.zeros_like(Q2)
    for i in range(resolution):
        for j in range(resolution):
            q = np.array([q_opt[0], Q2[i, j], Q3[i, j], q_opt[3]])
            SEF_q2_q3[i, j] = calculate_sef(q, q_opt, weights, comfort_threshold)

    # Create q3-q4 slice
    q3_range = np.linspace(q_opt[2] - joint_range_width[2] / 2, q_opt[2] + joint_range_width[2] / 2, resolution)
    q4_range = np.linspace(q_opt[3] - joint_range_width[3] / 2, q_opt[3] + joint_range_width[3] / 2, resolution)
    Q3, Q4 = np.meshgrid(q3_range, q4_range)

    # Calculate SEF for q3-q4 slice
    SEF_q3_q4 = np.zeros_like(Q3)
    for i in range(resolution):
        for j in range(resolution):
            q = np.array([q_opt[0], q_opt[1], Q3[i, j], Q4[i, j]])
            SEF_q3_q4[i, j] = calculate_sef(q, q_opt, weights, comfort_threshold)

    return {
        'q1_q2': (q1_range, q2_range, SEF_q1_q2),
        'q2_q3': (q2_range, q3_range, SEF_q2_q3),
        'q3_q4': (q3_range, q4_range, SEF_q3_q4)
    }


# Create task space SEF
def create_task_space_sef(q_opt, weights, comfort_threshold, d_ua, d_la, resolution=15):
    # Calculate workspace boundaries based on arm kinematics
    arm_length = np.linalg.norm(d_ua) + np.linalg.norm(d_la)

    # Get optimal positions
    p_elbow_opt, p_hand_opt = forward_kinematics(q_opt, d_ua, d_la)

    # Create centered grid in task space around optimal hand position
    # Make sure the optimal position is included in the grid
    range_size = arm_length * 0.8
    x_range = np.linspace(p_hand_opt[0] - range_size / 2, p_hand_opt[0] + range_size / 2, resolution)
    y_range = np.linspace(p_hand_opt[1] - range_size / 2, p_hand_opt[1] + range_size / 2, resolution)
    z_range = np.linspace(p_hand_opt[2] - range_size / 2, p_hand_opt[2] + range_size / 2, resolution)

    # Ensure optimal position is exactly on a grid point
    x_idx = np.argmin(np.abs(x_range - p_hand_opt[0]))
    y_idx = np.argmin(np.abs(y_range - p_hand_opt[1]))
    z_idx = np.argmin(np.abs(z_range - p_hand_opt[2]))
    x_range[x_idx] = p_hand_opt[0]
    y_range[y_idx] = p_hand_opt[1]
    z_range[z_idx] = p_hand_opt[2]

    # Initialize SEF arrays for each plane
    SEF_xy = np.full((resolution, resolution), np.nan)  # Z middle plane
    SEF_yz = np.full((resolution, resolution), np.nan)  # X middle plane
    SEF_xz = np.full((resolution, resolution), np.nan)  # Y middle plane

    # For 3D visualization
    points_x = []
    points_y = []
    points_z = []
    points_sef = []

    # Mid-indices for planes
    mid_z = z_idx
    mid_x = x_idx
    mid_y = y_idx

    # Calculate SEF for XY plane (at optimal Z)
    z_optimal = z_range[mid_z]
    for i, x in enumerate(x_range):
        for j, y in enumerate(y_range):
            target_pos = np.array([x, y, z_optimal])

            # Skip points outside reasonable workspace
            if np.linalg.norm(target_pos) > arm_length * 0.95:
                continue

            try:
                # Try multiple initial guesses for better coverage
                # First try optimal configuration as initial guess
                q = inverse_kinematics(target_pos, d_ua, d_la, q_init=q_opt)

                # If that fails, try a few random initializations
                if q is None:
                    for _ in range(3):
                        q_rand = np.array([
                            np.random.uniform(-math.pi / 18, 17 * math.pi / 18),
                            np.random.uniform(-np.pi / 3, 17 * math.pi / 18),
                            np.random.uniform(-np.pi / 3, np.pi / 2),
                            np.random.uniform(-np.pi / 2, np.pi / 3)
                        ])
                        q = inverse_kinematics(target_pos, d_ua, d_la, q_init=q_rand)
                        if q is not None:
                            break

                if q is not None:
                    sef_value = calculate_sef(q, q_opt, weights, comfort_threshold)
                    SEF_xy[j, i] = sef_value  # Note: j, i for proper display orientation

                    # Save for 3D visualization
                    points_x.append(x)
                    points_y.append(y)
                    points_z.append(z_optimal)
                    points_sef.append(sef_value)
            except:
                continue

    # Calculate SEF for YZ plane (at optimal X)
    x_optimal = x_range[mid_x]
    for j, y in enumerate(y_range):
        for k, z in enumerate(z_range):
            target_pos = np.array([x_optimal, y, z])

            # Skip points outside reasonable workspace
            if np.linalg.norm(target_pos) > arm_length * 0.95:
                continue

            try:
                # Try multiple initial guesses
                q = inverse_kinematics(target_pos, d_ua, d_la, q_init=q_opt)

                if q is None:
                    for _ in range(3):
                        q_rand = np.array([
                            np.random.uniform(-math.pi / 18, 17 * math.pi / 18),
                            np.random.uniform(-np.pi / 3, 17 * math.pi / 18),
                            np.random.uniform(-np.pi / 3, np.pi / 2),
                            np.random.uniform(-np.pi / 2, np.pi / 3)
                        ])
                        q = inverse_kinematics(target_pos, d_ua, d_la, q_init=q_rand)
                        if q is not None:
                            break

                if q is not None:
                    sef_value = calculate_sef(q, q_opt, weights, comfort_threshold)
                    SEF_yz[j, k] = sef_value

                    # Save for 3D visualization
                    points_x.append(x_optimal)
                    points_y.append(y)
                    points_z.append(z)
                    points_sef.append(sef_value)
            except:
                continue

    # Calculate SEF for XZ plane (at optimal Y)
    y_optimal = y_range[mid_y]
    for i, x in enumerate(x_range):
        for k, z in enumerate(z_range):
            target_pos = np.array([x, y_optimal, z])

            # Skip points outside reasonable workspace
            if np.linalg.norm(target_pos) > arm_length * 0.95:
                continue

            try:
                # Try multiple initial guesses
                q = inverse_kinematics(target_pos, d_ua, d_la, q_init=q_opt)

                if q is None:
                    for _ in range(3):
                        q_rand = np.array([
                            np.random.uniform(-math.pi / 18, 17 * math.pi / 18),
                            np.random.uniform(-np.pi / 3, 17 * math.pi / 18),
                            np.random.uniform(-np.pi / 3, np.pi / 2),
                            np.random.uniform(-np.pi / 2, np.pi / 3)
                        ])
                        q = inverse_kinematics(target_pos, d_ua, d_la, q_init=q_rand)
                        if q is not None:
                            break

                if q is not None:
                    sef_value = calculate_sef(q, q_opt, weights, comfort_threshold)
                    SEF_xz[i, k] = sef_value

                    # Save for 3D visualization
                    points_x.append(x)
                    points_y.append(y_optimal)
                    points_z.append(z)
                    points_sef.append(sef_value)
            except:
                continue

    # Special handling for optimal point and surroundings
    # Ensure we have the optimal point properly represented
    if np.isnan(SEF_xy[y_idx, x_idx]):
        SEF_xy[y_idx, x_idx] = -comfort_threshold  # Will be negative (comfortable)
    if np.isnan(SEF_yz[y_idx, z_idx]):
        SEF_yz[y_idx, z_idx] = -comfort_threshold
    if np.isnan(SEF_xz[x_idx, z_idx]):
        SEF_xz[x_idx, z_idx] = -comfort_threshold

    # Add more points around optimal position for better 3D visualization
    radius = 0.05  # Small radius around optimal point
    num_extra_points = 50

    for _ in range(num_extra_points):
        # Random direction from optimal point
        rand_dir = np.random.randn(3)
        rand_dir = rand_dir / np.linalg.norm(rand_dir)

        # Random distance from optimal point (concentrated near optimal)
        rand_dist = radius * np.random.beta(2, 5)  # Beta distribution for concentration near 0

        # New point
        new_point = p_hand_opt + rand_dir * rand_dist

        # Calculate SEF for this point
        try:
            q = inverse_kinematics(new_point, d_ua, d_la, q_init=q_opt)
            if q is not None:
                sef_value = calculate_sef(q, q_opt, weights, comfort_threshold)

                # Add to point lists
                points_x.append(new_point[0])
                points_y.append(new_point[1])
                points_z.append(new_point[2])
                points_sef.append(sef_value)
        except:
            continue

    # Add more random points throughout workspace for better coverage
    num_random_points = 500
    for _ in range(num_random_points):
        # Random spherical coordinates
        r = np.random.uniform(0.1, 0.9) * arm_length
        theta = np.random.uniform(0, 2 * np.pi)
        phi = np.random.uniform(0, np.pi)

        # Convert to Cartesian
        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(phi)

        target_pos = np.array([x, y, z])

        try:
            q = inverse_kinematics(target_pos, d_ua, d_la, q_init=q_opt)
            if q is not None:
                sef_value = calculate_sef(q, q_opt, weights, comfort_threshold)

                # Save for 3D visualization
                points_x.append(x)
                points_y.append(y)
                points_z.append(z)
                points_sef.append(sef_value)
        except:
            continue

    # Ensure the optimal point is in our visualization data
    points_x.append(p_hand_opt[0])
    points_y.append(p_hand_opt[1])
    points_z.append(p_hand_opt[2])
    points_sef.append(-comfort_threshold * 0.9)  # Slightly less than threshold to show as comfortable

    return {
        'xy_plane': (x_range, y_range, SEF_xy, z_optimal),
        'yz_plane': (y_range, z_range, SEF_yz, x_optimal),
        'xz_plane': (x_range, z_range, SEF_xz, y_optimal),
        '3d_points': (np.array(points_x), np.array(points_y), np.array(points_z), np.array(points_sef)),
        'optimal_positions': (p_elbow_opt, p_hand_opt)
    }


# Visualize SEF in joint space
def plot_joint_space_sef(joint_space_sef, q_opt):
    fig = plt.figure(figsize=(18, 6))

    # Plot q1-q2 slice
    q1_range, q2_range, SEF_q1_q2 = joint_space_sef['q1_q2']
    ax1 = fig.add_subplot(131)
    Q1, Q2 = np.meshgrid(q1_range, q2_range)
    contour = ax1.contourf(Q1, Q2, SEF_q1_q2, 20, cmap=cm.RdBu_r)
    # Mark optimal position
    ax1.plot(q_opt[0], q_opt[1], 'r*', markersize=10)
    # Draw zero-level contour
    ax1.contour(Q1, Q2, SEF_q1_q2, levels=[0], colors='k', linewidths=2)

    ax1.set_xlabel('Shoulder Abduction/Adduction (q1)')
    ax1.set_ylabel('Shoulder Flexion/Extension (q2)')
    ax1.set_title('SEF in q1-q2 Joint Space')
    fig.colorbar(contour, ax=ax1, label='SEF Value')

    # Plot q2-q3 slice
    q2_range, q3_range, SEF_q2_q3 = joint_space_sef['q2_q3']
    ax2 = fig.add_subplot(132)
    Q2, Q3 = np.meshgrid(q2_range, q3_range)
    contour = ax2.contourf(Q2, Q3, SEF_q2_q3, 20, cmap=cm.RdBu_r)
    # Mark optimal position
    ax2.plot(q_opt[1], q_opt[2], 'r*', markersize=10)
    # Draw zero-level contour
    ax2.contour(Q2, Q3, SEF_q2_q3, levels=[0], colors='k', linewidths=2)

    ax2.set_xlabel('Shoulder Flexion/Extension (q2)')
    ax2.set_ylabel('Shoulder Internal/External Rotation (q3)')
    ax2.set_title('SEF in q2-q3 Joint Space')
    fig.colorbar(contour, ax=ax2, label='SEF Value')

    # Plot q3-q4 slice
    q3_range, q4_range, SEF_q3_q4 = joint_space_sef['q3_q4']
    ax3 = fig.add_subplot(133)
    Q3, Q4 = np.meshgrid(q3_range, q4_range)
    contour = ax3.contourf(Q3, Q4, SEF_q3_q4, 20, cmap=cm.RdBu_r)
    # Mark optimal position
    ax3.plot(q_opt[2], q_opt[3], 'r*', markersize=10)
    # Draw zero-level contour
    ax3.contour(Q3, Q4, SEF_q3_q4, levels=[0], colors='k', linewidths=2)

    ax3.set_xlabel('Shoulder Internal/External Rotation (q3)')
    ax3.set_ylabel('Elbow Flexion/Extension (q4)')
    ax3.set_title('SEF in q3-q4 Joint Space')
    fig.colorbar(contour, ax=ax3, label='SEF Value')

    plt.tight_layout()
    plt.savefig('joint_space_sef.png', dpi=300)
    plt.show()


# Visualize SEF in task space
def plot_task_space_sef(task_space_sef):
    # Extract data
    x_range, y_range, SEF_xy, z_mid = task_space_sef['xy_plane']
    y_range, z_range, SEF_yz, x_mid = task_space_sef['yz_plane']
    x_range, z_range, SEF_xz, y_mid = task_space_sef['xz_plane']
    points_x, points_y, points_z, points_sef = task_space_sef['3d_points']
    p_elbow_opt, p_hand_opt = task_space_sef['optimal_positions']

    # Create 2D plots for each plane
    fig = plt.figure(figsize=(18, 6))

    # X-Y plane
    ax1 = fig.add_subplot(131)
    X, Y = np.meshgrid(x_range, y_range)
    contour = ax1.contourf(X, Y, SEF_xy, 20, cmap=cm.RdBu_r)
    ax1.contour(X, Y, SEF_xy, levels=[0], colors='k', linewidths=2)
    # Mark optimal hand position
    ax1.plot(p_hand_opt[0], p_hand_opt[1], 'r*', markersize=10, label='Optimal Hand Position')
    ax1.set_xlabel('X Position')
    ax1.set_ylabel('Y Position')
    ax1.set_title(f'SEF in X-Y Plane (Z={z_mid:.2f})')
    fig.colorbar(contour, ax=ax1, label='SEF Value')
    ax1.legend()

    # Y-Z plane
    ax2 = fig.add_subplot(132)
    Y, Z = np.meshgrid(y_range, z_range)
    contour = ax2.contourf(Y, Z, SEF_yz.T, 20, cmap=cm.RdBu_r)  # Transpose for correct orientation
    ax2.contour(Y, Z, SEF_yz.T, levels=[0], colors='k', linewidths=2)
    # Mark optimal hand position
    ax2.plot(p_hand_opt[1], p_hand_opt[2], 'r*', markersize=10, label='Optimal Hand Position')
    ax2.set_xlabel('Y Position')
    ax2.set_ylabel('Z Position')
    ax2.set_title(f'SEF in Y-Z Plane (X={x_mid:.2f})')
    fig.colorbar(contour, ax=ax2, label='SEF Value')
    ax2.legend()

    # X-Z plane
    ax3 = fig.add_subplot(133)
    X, Z = np.meshgrid(x_range, z_range)
    contour = ax3.contourf(X, Z, SEF_xz.T, 20, cmap=cm.RdBu_r)  # Transpose for correct orientation
    ax3.contour(X, Z, SEF_xz.T, levels=[0], colors='k', linewidths=2)
    # Mark optimal hand position
    ax3.plot(p_hand_opt[0], p_hand_opt[2], 'r*', markersize=10, label='Optimal Hand Position')
    ax3.set_xlabel('X Position')
    ax3.set_ylabel('Z Position')
    ax3.set_title(f'SEF in X-Z Plane (Y={y_mid:.2f})')
    fig.colorbar(contour, ax=ax3, label='SEF Value')
    ax3.legend()

    plt.tight_layout()
    plt.savefig('task_space_sef_2d.png', dpi=300)
    plt.show()

    # 3D visualization of ergonomic comfort regions
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Create masks for comfortable and uncomfortable regions
    comfortable_mask = points_sef < 0
    uncomfortable_mask = points_sef >= 0

    # Plot comfortable regions in blue
    if np.any(comfortable_mask):
        sef_comfortable = points_sef[comfortable_mask]
        norm_comfortable = plt.Normalize(min(sef_comfortable), 0)
        colors_comfortable = cm.Blues_r(norm_comfortable(sef_comfortable))

        ax.scatter(
            points_x[comfortable_mask],
            points_y[comfortable_mask],
            points_z[comfortable_mask],
            c=colors_comfortable,
            s=35,
            alpha=0.7,
            label='Comfortable Regions (SEF < 0)'
        )

    # Plot uncomfortable regions in red
    if np.any(uncomfortable_mask):
        sef_uncomfortable = points_sef[uncomfortable_mask]
        norm_uncomfortable = plt.Normalize(0, max(sef_uncomfortable))
        colors_uncomfortable = cm.Reds(norm_uncomfortable(sef_uncomfortable))

        ax.scatter(
            points_x[uncomfortable_mask],
            points_y[uncomfortable_mask],
            points_z[uncomfortable_mask],
            c=colors_uncomfortable,
            s=20,
            alpha=0.5,
            label='Uncomfortable Regions (SEF ≥ 0)'
        )

    # Plot the optimal arm configuration
    ax.scatter([p_elbow_opt[0]], [p_elbow_opt[1]], [p_elbow_opt[2]],
               color='green', s=100, marker='o', label='Optimal Elbow Position')
    ax.scatter([p_hand_opt[0]], [p_hand_opt[1]], [p_hand_opt[2]],
               color='red', s=200, marker='*', label='Optimal Hand Position')

    # Draw a line representing the arm in optimal configuration
    ax.plot([0, p_elbow_opt[0], p_hand_opt[0]],
            [0, p_elbow_opt[1], p_hand_opt[1]],
            [0, p_elbow_opt[2], p_hand_opt[2]],
            'k-', linewidth=2, label='Arm in Optimal Position')

    # Add a marker for the shoulder (origin)
    ax.scatter([0], [0], [0], color='black', s=100, marker='^', label='Shoulder')

    # Set labels and title
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_zlabel('Z Position')
    ax.set_title('3D Visualization of SEF in Task Space')

    # Set equal aspect ratio to have proper spherical visualization
    max_range = np.array([
        points_x.max() - points_x.min(),
        points_y.max() - points_y.min(),
        points_z.max() - points_z.min()
    ]).max() / 2.0

    mid_x = (points_x.max() + points_x.min()) / 2
    mid_y = (points_y.max() + points_y.min()) / 2
    mid_z = (points_z.max() + points_z.min()) / 2

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    # Add legend at a good position
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

    plt.tight_layout()
    plt.savefig('task_space_sef_3d.png', dpi=300)
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

    print("Computing joint space SEF...")
    joint_space_sef = create_joint_space_sef(q_opt, weights, comfort_threshold)
    plot_joint_space_sef(joint_space_sef, q_opt)

    print("Computing task space SEF (this may take a while)...")
    start_time = time.time()
    task_space_sef = create_task_space_sef(q_opt, weights, comfort_threshold, d_ua, d_la)
    print(f"Task space SEF computation completed in {time.time() - start_time:.2f} seconds")
    plot_task_space_sef(task_space_sef)


if __name__ == "__main__":
    main()