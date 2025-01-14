import numpy as np

def plot_skeleton(ax, global_positions, parent_indices, color='k'):
    for i in range(len(global_positions)):
        if parent_indices[i] != -1:
            parent_pos = global_positions[parent_indices[i]]
            joint_pos = global_positions[i]
            ax.plot([parent_pos[0], joint_pos[0]], [parent_pos[1], joint_pos[1]], [parent_pos[2], joint_pos[2]], color)

def plot_box(ax, corners, color='b'):
    """
    Plot the cuboid given its corners.
    :param ax: The Matplotlib 3D axis to plot on.
    :param corners: Array of 8 corners of the cuboid (Nx3).
    """
    # Define edges of the box (pairs of corners)
    edges = [
        (0, 1), (1, 3), (3, 2), (2, 0),  # Bottom face
        (4, 5), (5, 7), (7, 6), (6, 4),  # Top face
        (0, 4), (1, 5), (2, 6), (3, 7)   # Vertical edges
    ]

    # Plot each edge
    for edge in edges:
        start, end = edge
        ax.plot([corners[start][0], corners[end][0]],
                [corners[start][1], corners[end][1]],
                [corners[start][2], corners[end][2]], color=color)

def plot_ellipsoid(ax, eigenvalues, eigenvectors, center, color):
    eigenvalues = eigenvalues[-3:]
    eigenvectors = eigenvectors[-3:, -3:]
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v)) / 40
    y = np.outer(np.sin(u), np.sin(v)) / 40
    z = np.outer(np.ones_like(u), np.cos(v)) / 40
    for i in range(len(x)):
        for j in range(len(x)):
            [x[i, j], y[i, j], z[i, j]] = np.dot(eigenvectors, np.array([x[i, j], y[i, j], z[i, j]]) * np.sqrt(eigenvalues))
            x[i, j] += center[0]
            y[i, j] += center[1]
            z[i, j] += center[2]
    ax.plot_surface(x, y, z, color=color, alpha=0.3)

def ua_get_color(score):
    if 1 <= score <= 2:
        return 'green'
    elif 3 <= score <= 4:
        return 'orange'
    elif 5 <= score <= 6:
        return 'red'
    else:
        return 'purple'  # Other scores in purple

def la_get_color(score):
    if 1 <= score < 2:
        return 'green'
    elif 2 <= score:
        return 'orange'
    else:
        return 'purple'  # Other scores in purple

def trunk_get_color(score):
    if 1 <= score <= 2:
        return 'green'
    elif 3 <= score <= 4:
        return 'orange'
    elif 5 <= score <= 6:
        return 'red'
    else:
        return 'purple'  # Other scores in purple

def overall_arm_get_color(score):
    if 1 <= score <= 2:
        return 'green'
    elif 3 <= score <= 4:
        return 'orange'
    elif 5 <= score <= 9:
        return 'red'
    else:
        return 'purple'  # Other scores in purple