import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load the ergo trajectory data
data = np.loadtxt('/home/ubuntu/Ergo-Manip/SEF/data/pose3_ergo.txt')
data_stra = np.loadtxt('/home/ubuntu/Ergo-Manip/SEF/data/pose3_stra.txt')

# Extract the relevant columns
time = data[:, 0]  # 1st column (time in millionths of a second)
x = data[:, 1]     # 2nd column (x position)
y = data[:, 2]     # 3rd column (y position)
z = data[:, 3]     # 4th column (z position)

# Extract the trajectory from the point of first motion
period = 10000
# start_idx = 57500
# start_idx = 65000
start_idx = 75000
time_moving = data[:, 0][start_idx:start_idx + period]
x_moving = data[:, 1][start_idx:start_idx + period]
y_moving = data[:, 2][start_idx:start_idx + period]
z_moving = data[:, 3][start_idx:start_idx + period]

# start_idx_stra = 87000
# start_idx_stra = 60000
start_idx_stra = 87000
time_moving_stra = data_stra[:, 0][start_idx_stra:start_idx_stra + period]
x_moving_stra = data_stra[:, 1][start_idx_stra:start_idx_stra + period]
y_moving_stra = data_stra[:, 2][start_idx_stra:start_idx_stra + period]
z_moving_stra = data_stra[:, 3][start_idx_stra:start_idx_stra + period]

# Create a 3D plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Make the background white
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

# Turn off the grid
# ax.grid(False)

# Plot the ergo trajectory in orange with larger line width
ax.plot(x_moving, y_moving, z_moving, color='#0343DF', label='SEF-Based', linewidth=3)

# Plot the straight trajectory in blue with larger line width
ax.plot(x_moving_stra, y_moving_stra, z_moving_stra, color='#F97306', label='Point-to-Point', linewidth=3)

# Set labels and title with larger font size
# ax.set_xlabel('X', fontsize=24)
# ax.set_ylabel('Y', fontsize=24)
# ax.set_zlabel('Z', fontsize=24)

# Make tick labels larger
ax.tick_params(axis='x', labelsize=20)
ax.tick_params(axis='y', labelsize=20)
ax.tick_params(axis='z', labelsize=20)

# Add a legend with larger font
ax.legend(fontsize=24)

# Show the plot
plt.tight_layout()
plt.show()