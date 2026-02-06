import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import interp1d

# 加载数据
optimized_robot_pos = np.load('/data/joint_comparison_0918/test/optimized_robot_positions.npy', allow_pickle=True)
recorded_human_data = np.load('/data/joint_comparison_0918/test/recorded_human_positions.npy', allow_pickle=True)
real_robot_raw = np.loadtxt("/home/clover/catkin_ws/record/test/python_curi_dual_arm_ic_2025_9_18_15_03_39/tcp_actual_position_rpy_l.txt")

# 提取手腕位置数据
wrist_positions = np.array(recorded_human_data.item()['wrist_positions'])
robot_positions = np.array(optimized_robot_pos)

# 处理实际机器人数据
# 提取第3-5列(索引2-4)并截取指定范围的数据
real_robot_pos = real_robot_raw[57474:66301, 1:4]

print("手腕位置数据形状:", wrist_positions.shape)
print("优化机器人位置数据形状:", robot_positions.shape)
print("实际机器人位置数据形状:", real_robot_pos.shape)

# 1. 减去初始点，使三组数据都从(0,0,0)开始
normalized_wrist = wrist_positions - wrist_positions[0]
normalized_robot = robot_positions - robot_positions[0]
normalized_real_robot = real_robot_pos - real_robot_pos[0]

# 2. 创建时间归一化数组 (0到1)
wrist_time = np.linspace(0, 1, len(normalized_wrist))
robot_time = np.linspace(0, 1, len(normalized_robot))
real_robot_time = np.linspace(0, 1, len(normalized_real_robot))

# 3. 选择一个标准长度进行重采样
standard_length = 1000
standard_time = np.linspace(0, 1, standard_length)

# 4. 为每个维度创建插值函数
wrist_interp_x = interp1d(wrist_time, normalized_wrist[:, 0], kind='cubic')
wrist_interp_y = interp1d(wrist_time, normalized_wrist[:, 1], kind='cubic')
wrist_interp_z = interp1d(wrist_time, normalized_wrist[:, 2], kind='cubic')

robot_interp_x = interp1d(robot_time, normalized_robot[:, 0], kind='cubic')
robot_interp_y = interp1d(robot_time, normalized_robot[:, 1], kind='cubic')
robot_interp_z = interp1d(robot_time, normalized_robot[:, 2], kind='cubic')

real_robot_interp_x = interp1d(real_robot_time, normalized_real_robot[:, 0], kind='cubic')
real_robot_interp_y = interp1d(real_robot_time, normalized_real_robot[:, 1], kind='cubic')
real_robot_interp_z = interp1d(real_robot_time, normalized_real_robot[:, 2], kind='cubic')

# 5. 重采样到标准时间轴
wrist_resampled = np.column_stack([
    wrist_interp_x(standard_time),
    wrist_interp_y(standard_time),
    wrist_interp_z(standard_time)
])

robot_resampled = np.column_stack([
    robot_interp_x(standard_time),
    robot_interp_y(standard_time),
    robot_interp_z(standard_time)
])

real_robot_resampled = np.column_stack([
    real_robot_interp_x(standard_time),
    real_robot_interp_y(standard_time),
    real_robot_interp_z(standard_time)
])

# 6. 计算差异
# 人类手腕 vs 优化机器人
human_vs_optimized_diff = np.linalg.norm(wrist_resampled - robot_resampled, axis=1)
# 人类手腕 vs 实际机器人
human_vs_real_diff = np.linalg.norm(wrist_resampled - real_robot_resampled, axis=1)
# 优化机器人 vs 实际机器人
optimized_vs_real_diff = np.linalg.norm(robot_resampled - real_robot_resampled, axis=1)

# 统计数据
print("\n====== 差异统计 ======")
print("人类手腕 vs 优化机器人:")
print(f"  平均差异: {np.mean(human_vs_optimized_diff):.4f} 米")
print(f"  最大差异: {np.max(human_vs_optimized_diff):.4f} 米")
print(f"  最小差异: {np.min(human_vs_optimized_diff):.4f} 米")
print(f"  标准差: {np.std(human_vs_optimized_diff):.4f} 米")

print("\n人类手腕 vs 实际机器人:")
print(f"  平均差异: {np.mean(human_vs_real_diff):.4f} 米")
print(f"  最大差异: {np.max(human_vs_real_diff):.4f} 米")
print(f"  最小差异: {np.min(human_vs_real_diff):.4f} 米")
print(f"  标准差: {np.std(human_vs_real_diff):.4f} 米")

print("\n优化机器人 vs 实际机器人:")
print(f"  平均差异: {np.mean(optimized_vs_real_diff):.4f} 米")
print(f"  最大差异: {np.max(optimized_vs_real_diff):.4f} 米")
print(f"  最小差异: {np.min(optimized_vs_real_diff):.4f} 米")
print(f"  标准差: {np.std(optimized_vs_real_diff):.4f} 米")

# 绘制位置差异随标准化时间的变化
plt.figure(figsize=(12, 6))
plt.plot(standard_time, human_vs_optimized_diff, 'b-', label='人类 vs 优化机器人')
plt.plot(standard_time, human_vs_real_diff, 'r-', label='人类 vs 实际机器人')
plt.plot(standard_time, optimized_vs_real_diff, 'g-', label='优化机器人 vs 实际机器人')
plt.title('三组轨迹之间的位置差异')
plt.xlabel('标准化时间 (0-1)')
plt.ylabel('欧氏距离 (米)')
plt.grid(True)
plt.legend()
# plt.savefig('three_trajectories_position_diff.png')
plt.show()

# 绘制轨迹对比的3D图
fig = plt.figure(figsize=(14, 12))
ax = fig.add_subplot(111, projection='3d')

# 绘制重采样后的轨迹
ax.plot(wrist_resampled[:, 0], wrist_resampled[:, 1], wrist_resampled[:, 2],
        'r-', linewidth=2, label='人类手腕轨迹')
ax.plot(robot_resampled[:, 0], robot_resampled[:, 1], robot_resampled[:, 2],
        'b-', linewidth=2, label='优化机器人轨迹')
ax.plot(real_robot_resampled[:, 0], real_robot_resampled[:, 1], real_robot_resampled[:, 2],
        'g-', linewidth=2, label='实际机器人轨迹')

# 添加点以标记起点和终点
ax.scatter(wrist_resampled[0, 0], wrist_resampled[0, 1], wrist_resampled[0, 2],
           c='r', marker='o', s=100, label='人类起点')
ax.scatter(wrist_resampled[-1, 0], wrist_resampled[-1, 1], wrist_resampled[-1, 2],
           c='r', marker='x', s=100, label='人类终点')

ax.scatter(robot_resampled[0, 0], robot_resampled[0, 1], robot_resampled[0, 2],
           c='b', marker='o', s=100, label='优化机器人起点')
ax.scatter(robot_resampled[-1, 0], robot_resampled[-1, 1], robot_resampled[-1, 2],
           c='b', marker='x', s=100, label='优化机器人终点')

ax.scatter(real_robot_resampled[0, 0], real_robot_resampled[0, 1], real_robot_resampled[0, 2],
           c='g', marker='o', s=100, label='实际机器人起点')
ax.scatter(real_robot_resampled[-1, 0], real_robot_resampled[-1, 1], real_robot_resampled[-1, 2],
           c='g', marker='x', s=100, label='实际机器人终点')

ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.set_title('三组轨迹对比 (标准化时间)')
ax.legend()
# plt.savefig('three_trajectories_comparison_3d.png')
plt.show()

# 分别绘制X、Y、Z三个方向的轨迹对比
fig, axs = plt.subplots(3, 1, figsize=(12, 15))

# X方向
axs[0].plot(standard_time, wrist_resampled[:, 0], 'r-', label='人类手腕')
axs[0].plot(standard_time, robot_resampled[:, 0], 'b-', label='优化机器人')
axs[0].plot(standard_time, real_robot_resampled[:, 0], 'g-', label='实际机器人')
axs[0].set_title('X方向轨迹对比')
axs[0].set_ylabel('X位置 (m)')
axs[0].grid(True)
axs[0].legend()

# Y方向
axs[1].plot(standard_time, wrist_resampled[:, 1], 'r-', label='人类手腕')
axs[1].plot(standard_time, robot_resampled[:, 1], 'b-', label='优化机器人')
axs[1].plot(standard_time, real_robot_resampled[:, 1], 'g-', label='实际机器人')
axs[1].set_title('Y方向轨迹对比')
axs[1].set_ylabel('Y位置 (m)')
axs[1].grid(True)
axs[1].legend()

# Z方向
axs[2].plot(standard_time, wrist_resampled[:, 2], 'r-', label='人类手腕')
axs[2].plot(standard_time, robot_resampled[:, 2], 'b-', label='优化机器人')
axs[2].plot(standard_time, real_robot_resampled[:, 2], 'g-', label='实际机器人')
axs[2].set_title('Z方向轨迹对比')
axs[2].set_xlabel('标准化时间 (0-1)')
axs[2].set_ylabel('Z位置 (m)')
axs[2].grid(True)
axs[2].legend()

plt.tight_layout()
# plt.savefig('three_trajectories_comparison_xyz.png')
plt.show()

# 计算并绘制速度剖面对比
# 计算速度 (位移变化率)
wrist_velocity = np.zeros((standard_length-1, 3))
robot_velocity = np.zeros((standard_length-1, 3))
real_robot_velocity = np.zeros((standard_length-1, 3))

for i in range(3):  # x, y, z 三个方向
    wrist_velocity[:, i] = np.diff(wrist_resampled[:, i]) / np.diff(standard_time)
    robot_velocity[:, i] = np.diff(robot_resampled[:, i]) / np.diff(standard_time)
    real_robot_velocity[:, i] = np.diff(real_robot_resampled[:, i]) / np.diff(standard_time)

# 计算速度的欧氏范数
wrist_speed = np.linalg.norm(wrist_velocity, axis=1)
robot_speed = np.linalg.norm(robot_velocity, axis=1)
real_robot_speed = np.linalg.norm(real_robot_velocity, axis=1)

# 绘制速度对比
plt.figure(figsize=(12, 6))
plt.plot(standard_time[:-1], wrist_speed, 'r-', label='人类手腕速度')
plt.plot(standard_time[:-1], robot_speed, 'b-', label='优化机器人速度')
plt.plot(standard_time[:-1], real_robot_speed, 'g-', label='实际机器人速度')
plt.title('速度对比 (标准化时间)')
plt.xlabel('标准化时间 (0-1)')
plt.ylabel('速度 (m/s)')
plt.grid(True)
plt.legend()
# plt.savefig('three_trajectories_velocity_comparison.png')
plt.show()

# 额外比较：轨迹长度对比
wrist_length = np.sum(np.sqrt(np.sum(np.diff(wrist_resampled, axis=0)**2, axis=1)))
robot_length = np.sum(np.sqrt(np.sum(np.diff(robot_resampled, axis=0)**2, axis=1)))
real_robot_length = np.sum(np.sqrt(np.sum(np.diff(real_robot_resampled, axis=0)**2, axis=1)))

print("\n====== 轨迹长度比较 ======")
print(f"人类手腕轨迹长度: {wrist_length:.4f} 米")
print(f"优化机器人轨迹长度: {robot_length:.4f} 米")
print(f"实际机器人轨迹长度: {real_robot_length:.4f} 米")
print(f"人类/优化机器人长度比: {wrist_length/robot_length:.4f}")
print(f"人类/实际机器人长度比: {wrist_length/real_robot_length:.4f}")
print(f"优化机器人/实际机器人长度比: {robot_length/real_robot_length:.4f}")