import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import manipulability_calculator as manip
import numpy as np

# Function to plot an ellipse based on eigenvalues and eigenvectors
def plot_ellipse(ax, center, width, height, angle, edge_color):
    ellipse = Ellipse(xy=center, width=2 * width, height=2 * height, angle=angle,
                      edgecolor=edge_color, facecolor='none', linewidth=1.5)
    ax.add_patch(ellipse)

# Example data
jacobians_left = np.load('/data/jacobian_data/jacobians_right_hand_2.npy', allow_pickle=True)[:330:10]

fig, axs = plt.subplots(3, 1, figsize=(6, 6))  # 1 row, 3 columns of subplots
titles = ['Manipulability Ellipses in XY Plane', 'Manipulability Ellipses in XZ Plane', 'Manipulability Ellipses in YZ Plane']
ylabels = ['$M^F_{xy}$', '$M^F_{xz}$', '$M^F_{yz}$']
planes = [(0, 1), (0, 2), (1, 2)]  # Tuple pairs to select planes: (X, Y), (X, Z), (Y, Z)

for ax, title, ylabel, (dim1, dim2) in zip(axs, titles, ylabels, planes):
    ax.set_title(title)
    ax.set_aspect('equal')
    ax.set_xlim(-1, len(jacobians_left))
    ax.set_ylim(-2, 2)
    ax.set_xlabel('t')
    ax.set_ylabel(ylabel)
    ax.grid(False)

    # Plot each manipulability ellipse
    for i, J in enumerate(jacobians_left):
        eigenvalues, eigenvectors = manip.calculate_manipulability(np.squeeze(J, axis=0))
        major_axis_length = np.sqrt(max(eigenvalues))
        minor_axis_length = np.sqrt(min(eigenvalues))
        major_index = eigenvalues.argmax()
        minor_index = eigenvalues.argmin()

        # Calculate angle of the major axis in the plane
        angle = np.degrees(np.arctan2(eigenvectors[dim2, major_index], eigenvectors[dim1, major_index]))

        # Center position varies with i along the dim1 axis (0 for X, 1 for Y, 2 for Z)
        center = (i, 0)  # This spreads ellipses along the X-axis based on index

        # Plot ellipse in the respective plane
        plot_ellipse(ax, center, major_axis_length, minor_axis_length, angle, 'blue')

plt.savefig('/home/ubuntu/Ergo-Manip/data/fig/demo_2_manip_right.png', format='png', dpi=300)

plt.tight_layout()
plt.show()