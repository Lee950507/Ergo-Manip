import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec


def forward_kinematics(q1, q2, l1, l2):
    """正向运动学：从关节角度计算末端执行器位置"""
    x = l1 * np.cos(q1) + l2 * np.cos(q1 + q2)
    y = l1 * np.sin(q1) + l2 * np.sin(q1 + q2)
    return x, y


def calculate_joint_sef(q1, q2, q1_opt, q2_opt, comfort_threshold, weights=None):
    """计算关节空间的SEF值"""
    if weights is None:
        weights = [1, 1]
    distance = np.sqrt(weights[0] * (q1 - q1_opt) ** 2 + weights[1] * (q2 - q2_opt) ** 2)
    return distance - comfort_threshold


def calculate_sef_gradient(q1, q2, q1_opt, q2_opt, weights=None):
    """计算SEF的梯度"""
    if weights is None:
        weights = [1, 1]

    # 计算SEF对q1和q2的偏导数
    dSEF_dq1 = 2 * weights[0] * (q1 - q1_opt)
    dSEF_dq2 = 2 * weights[1] * (q2 - q2_opt)

    # 计算梯度的长度以便归一化
    grad_length = np.sqrt(dSEF_dq1 ** 2 + dSEF_dq2 ** 2) + 1e-10  # 避免除以0

    # 返回归一化的梯度
    return np.array([dSEF_dq1 / grad_length, dSEF_dq2 / grad_length])


def calculate_joint_space_path(start_q, end_q, q1_opt, q2_opt, comfort_threshold,
                               weights=None, step_size=0.05, max_steps=1000,
                               goal_weight=0.5, random_factor=0.05):
    """
    使用混合了目标吸引力和SEF梯度的方法计算关节空间的路径

    参数:
    - start_q: 起始关节角度 [q1, q2]
    - end_q: 目标关节角度 [q1, q2]
    - q1_opt, q2_opt: 最优关节角度
    - comfort_threshold: 舒适阈值
    - weights: 关节权重
    - step_size: 每步移动的距离
    - max_steps: 最大步数
    - goal_weight: 目标吸引力权重
    - random_factor: 随机扰动因子，防止陷入局部最小值

    返回:
    - path: 关节空间路径的点列表
    - sef_values: 路径上每点的SEF值
    """
    if weights is None:
        weights = [1, 1]

    # 初始化路径和当前位置
    current_q = np.array(start_q)
    path = [current_q.copy()]
    sef_values = [calculate_joint_sef(current_q[0], current_q[1], q1_opt, q2_opt, comfort_threshold, weights)]

    for _ in range(max_steps):
        # 计算到目标的向量
        to_goal = np.array(end_q) - current_q
        distance_to_goal = np.linalg.norm(to_goal)

        # 如果足够接近目标，则结束
        if distance_to_goal < step_size:
            path.append(np.array(end_q))
            sef_values.append(calculate_joint_sef(end_q[0], end_q[1], q1_opt, q2_opt, comfort_threshold, weights))
            break

        # 计算目标吸引力方向（归一化）
        goal_direction = to_goal / (distance_to_goal + 1e-10)

        # 计算SEF梯度
        sef_gradient = calculate_sef_gradient(current_q[0], current_q[1], q1_opt, q2_opt, weights)

        # 混合方向：目标吸引力和SEF梯度的负方向
        mixed_direction = goal_weight * goal_direction - (1 - goal_weight) * sef_gradient

        # 添加随机扰动以避免局部最小值
        random_noise = random_factor * np.random.normal(0, 1, 2)
        mixed_direction += random_noise

        # 归一化方向
        direction_norm = np.linalg.norm(mixed_direction)
        if direction_norm > 0:
            mixed_direction = mixed_direction / direction_norm

        # 更新位置
        current_q = current_q + step_size * mixed_direction

        # 保存路径和SEF值
        path.append(current_q.copy())
        sef_values.append(calculate_joint_sef(current_q[0], current_q[1], q1_opt, q2_opt, comfort_threshold, weights))

    return np.array(path), np.array(sef_values)


def calculate_cartesian_path(joint_path, l1, l2):
    """计算关节空间路径对应的笛卡尔空间轨迹"""
    cartesian_path = []
    for q in joint_path:
        x, y = forward_kinematics(q[0], q[1], l1, l2)
        cartesian_path.append([x, y])
    return np.array(cartesian_path)


def draw_robot_arm(ax, q1, q2, l1, l2, color='k', linewidth=2, add_labels=False):
    """在指定轴上绘制机械臂"""
    # 计算肘部和末端位置
    elbow_x = l1 * np.cos(q1)
    elbow_y = l1 * np.sin(q1)
    end_x, end_y = forward_kinematics(q1, q2, l1, l2)

    # 绘制连杆
    ax.plot([0, elbow_x], [0, elbow_y], color=color, linewidth=linewidth)
    ax.plot([elbow_x, end_x], [elbow_y, end_y], color=color, linewidth=linewidth)

    # 标记关节
    ax.plot(0, 0, 'o', color=color, markersize=6)  # 基座
    ax.plot(elbow_x, elbow_y, 'o', color=color, markersize=6)  # 肘部
    ax.plot(end_x, end_y, 'o', color=color, markersize=6)  # 末端

    if add_labels:
        ax.text(0, 0, ' Base', fontsize=10)
        ax.text(elbow_x, elbow_y, ' Joint 2', fontsize=10)
        ax.text(end_x, end_y, ' End-effector', fontsize=10)


def create_joint_sef_field(q1_min, q1_max, q2_min, q2_max, q1_opt, q2_opt,
                           comfort_threshold, weights, resolution=100):
    """创建关节空间的SEF场"""
    q1_vals = np.linspace(q1_min, q1_max, resolution)
    q2_vals = np.linspace(q2_min, q2_max, resolution)
    Q1, Q2 = np.meshgrid(q1_vals, q2_vals)

    # 计算每个点的SEF值
    SEF = np.zeros_like(Q1)
    for i in range(resolution):
        for j in range(resolution):
            SEF[i, j] = calculate_joint_sef(Q1[i, j], Q2[i, j], q1_opt, q2_opt,
                                            comfort_threshold, weights)

    return Q1, Q2, SEF


def visualize_paths(joint_path, cartesian_path, sef_values, q1_opt, q2_opt,
                    l1, l2, comfort_threshold, weights, q_ranges, anim_save=None):
    """可视化关节空间和笛卡尔空间的路径"""
    q1_min, q1_max, q2_min, q2_max = q_ranges

    # 创建关节空间的SEF场
    Q1, Q2, joint_sef = create_joint_sef_field(q1_min, q1_max, q2_min, q2_max,
                                               q1_opt, q2_opt, comfort_threshold,
                                               weights, resolution=100)

    # 计算最优点的笛卡尔坐标
    x_opt, y_opt = forward_kinematics(q1_opt, q2_opt, l1, l2)

    # 创建图形
    fig = plt.figure(figsize=(18, 10))
    gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 0.05])

    # 关节空间子图
    ax_joint = plt.subplot(gs[0, 0])

    # 可视化关节空间的SEF
    cmap = plt.cm.RdBu_r
    max_abs = max(abs(np.min(joint_sef)), abs(np.max(joint_sef)))
    norm = plt.Normalize(-max_abs, max_abs)

    contour = ax_joint.contourf(Q1, Q2, joint_sef, 20, cmap=cmap, norm=norm, alpha=0.8)
    zero_contour = ax_joint.contour(Q1, Q2, joint_sef, [0], colors='green', linewidths=2)
    ax_joint.clabel(zero_contour, inline=True, fontsize=8, fmt='%1.1f')

    # 绘制关节空间路径
    start_q = joint_path[0]
    end_q = joint_path[-1]

    ax_joint.plot(joint_path[:, 0], joint_path[:, 1], 'r-', linewidth=2, label='Planned Path')
    ax_joint.plot(start_q[0], start_q[1], 'bo', markersize=8, label='Start')
    ax_joint.plot(end_q[0], end_q[1], 'go', markersize=8, label='End')
    ax_joint.plot(q1_opt, q2_opt, 'mo', markersize=8, label='Optimal Point')

    ax_joint.set_xlabel('q1 (rad)')
    ax_joint.set_ylabel('q2 (rad)')
    ax_joint.set_title('Joint Space Path with SEF')
    ax_joint.grid(True, linestyle='--', alpha=0.6)
    ax_joint.legend(loc='upper right')

    # 笛卡尔空间子图
    ax_cart = plt.subplot(gs[0, 1])

    # 定义工作空间边界
    workspace_radius = l1 + l2
    theta = np.linspace(0, 2 * np.pi, 100)

    # 内边界 (|l1-l2|)
    inner_radius = abs(l1 - l2)
    inner_circle = Circle((0, 0), inner_radius, fill=False, linestyle='--', color='gray', label='Inner Boundary')
    ax_cart.add_patch(inner_circle)

    # 外边界 (l1+l2)
    outer_circle = Circle((0, 0), workspace_radius, fill=False, linestyle='--', color='black', label='Outer Boundary')
    ax_cart.add_patch(outer_circle)

    # 绘制笛卡尔空间轨迹
    start_cart = cartesian_path[0]
    end_cart = cartesian_path[-1]

    # 初始化空的轨迹线和机械臂，会在动画中更新
    path_line, = ax_cart.plot([], [], 'r-', linewidth=2, label='Cartesian Path')
    ax_cart.plot(start_cart[0], start_cart[1], 'bo', markersize=8, label='Start')
    ax_cart.plot(end_cart[0], end_cart[1], 'go', markersize=8, label='End')
    ax_cart.plot(x_opt, y_opt, 'mo', markersize=8, label='Optimal Position')

    ax_cart.set_xlabel('X')
    ax_cart.set_ylabel('Y')
    ax_cart.set_title('Cartesian Space Trajectory')
    ax_cart.set_xlim([-workspace_radius * 1.2, workspace_radius * 1.2])
    ax_cart.set_ylim([-workspace_radius * 1.2, workspace_radius * 1.2])
    ax_cart.set_aspect('equal')
    ax_cart.grid(True, linestyle='--', alpha=0.6)
    ax_cart.legend(loc='upper right')

    # 添加颜色条
    cax = plt.subplot(gs[1, :])
    cbar = plt.colorbar(contour, cax=cax, orientation='horizontal')
    cbar.set_label('SEF Value')

    plt.tight_layout()

    # 设置动画
    def init():
        path_line.set_data([], [])
        return path_line,

    # 存储机械臂线条的引用
    arm_lines = []

    def animate(i):
        # 清除之前的机械臂绘制
        for line in arm_lines:
            line.remove()
        arm_lines.clear()

        # 更新路径
        path_line.set_data(cartesian_path[:i + 1, 0], cartesian_path[:i + 1, 1])

        # 绘制当前姿态的机械臂
        q1, q2 = joint_path[i]
        elbow_x = l1 * np.cos(q1)
        elbow_y = l1 * np.sin(q1)
        end_x, end_y = cartesian_path[i]

        line1, = ax_cart.plot([0, elbow_x], [0, elbow_y], 'b-', linewidth=2)
        line2, = ax_cart.plot([elbow_x, end_x], [elbow_y, end_y], 'b-', linewidth=2)
        point1, = ax_cart.plot(0, 0, 'ko', markersize=6)
        point2, = ax_cart.plot(elbow_x, elbow_y, 'ko', markersize=6)
        point3, = ax_cart.plot(end_x, end_y, 'ko', markersize=6)

        arm_lines.extend([line1, line2, point1, point2, point3])

        return [path_line] + arm_lines

    # 创建动画
    num_frames = len(joint_path)
    ani = FuncAnimation(fig, animate, frames=num_frames, init_func=init,
                        interval=50, blit=True, repeat=True)

    if anim_save:
        # 保存动画为GIF或MP4
        ani.save(anim_save, writer='pillow', fps=20)

    plt.tight_layout()
    plt.show()

    # 返回动画对象以防止被垃圾回收
    return ani


def main():
    # 定义参数
    l1, l2 = 1.0, 0.8
    q1_opt, q2_opt = np.pi / 4, -np.pi / 3
    comfort_threshold = 0.5
    weights = [1, 1]

    # 定义关节范围
    q1_min, q1_max = -np.pi, np.pi
    q2_min, q2_max = -np.pi, np.pi
    q_ranges = [q1_min, q1_max, q2_min, q2_max]

    # 定义起点和终点
    start_q = [-0.8, 0.6]  # 远离最优点的起点
    end_q = [0.8, -0.8]  # 远离最优点的终点

    # 计算关节空间路径和SEF值
    joint_path, sef_values = calculate_joint_space_path(
        start_q, end_q, q1_opt, q2_opt, comfort_threshold, weights,
        step_size=0.05, max_steps=1000, goal_weight=0.5, random_factor=0.05
    )

    # 计算笛卡尔空间轨迹
    cartesian_path = calculate_cartesian_path(joint_path, l1, l2)

    # 可视化路径
    ani = visualize_paths(joint_path, cartesian_path, sef_values, q1_opt, q2_opt,
                          l1, l2, comfort_threshold, weights, q_ranges)


if __name__ == "__main__":
    main()