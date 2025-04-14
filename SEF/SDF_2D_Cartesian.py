import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import Circle


def forward_kinematics(q1, q2, l1, l2):
    """
    正向运动学：从关节角度计算末端执行器位置

    参数:
    - q1, q2: 关节角度
    - l1, l2: 连杆长度

    返回:
    - (x, y): 笛卡尔坐标
    """
    x = l1 * np.cos(q1) + l2 * np.cos(q1 + q2)
    y = l1 * np.sin(q1) + l2 * np.sin(q1 + q2)
    return x, y


def inverse_kinematics(x, y, l1, l2):
    """
    逆运动学：从末端执行器位置计算关节角度

    参数:
    - x, y: 笛卡尔坐标
    - l1, l2: 连杆长度

    返回:
    - ((q1_sol1, q2_sol1), (q1_sol2, q2_sol2)): 两组可能的解
    """
    # 计算肘部的两种可能位置（肘上和肘下）
    r_squared = x ** 2 + y ** 2
    r = np.sqrt(r_squared)

    # 检查位置是否在工作空间内
    if r > l1 + l2 or r < abs(l1 - l2):
        return None  # 点不在工作空间内

    # 计算第二个关节角
    cos_q2 = (r_squared - l1 ** 2 - l2 ** 2) / (2 * l1 * l2)
    cos_q2 = np.clip(cos_q2, -1.0, 1.0)  # 处理数值误差

    sin_q2_pos = np.sqrt(1 - cos_q2 ** 2)  # 肘上解
    sin_q2_neg = -sin_q2_pos  # 肘下解

    q2_sol1 = np.arctan2(sin_q2_pos, cos_q2)
    q2_sol2 = np.arctan2(sin_q2_neg, cos_q2)

    # 计算第一个关节角
    k1 = l1 + l2 * cos_q2
    k2 = l2 * sin_q2_pos
    gamma = np.arctan2(k2, k1)
    alpha = np.arctan2(y, x)
    q1_sol1 = alpha - gamma

    k2 = l2 * sin_q2_neg
    gamma = np.arctan2(k2, k1)
    q1_sol2 = alpha - gamma

    return ((q1_sol1, q2_sol1), (q1_sol2, q2_sol2))


def calculate_joint_sef(q1, q2, q1_opt, q2_opt, comfort_threshold, weights):
    """计算关节空间的SEF值"""
    distance = np.sqrt(weights[0] * (q1 - q1_opt) ** 2 + weights[1] * (q2 - q2_opt) ** 2)
    return distance - comfort_threshold


def calculate_cartesian_sef(l1, l2, q1_opt, q2_opt, comfort_threshold=0.5, weights=None, resolution=100):
    """
    计算笛卡尔空间中的SEF

    参数:
    - l1, l2: 连杆长度
    - q1_opt, q2_opt: 最优关节角度
    - comfort_threshold: 舒适阈值
    - weights: 关节权重
    - resolution: 网格分辨率
    """
    if weights is None:
        weights = [1, 1]

    # 计算最优末端执行器位置
    x_opt, y_opt = forward_kinematics(q1_opt, q2_opt, l1, l2)

    # 计算工作空间范围
    workspace_radius = l1 + l2

    # 创建笛卡尔空间网格
    x_min, x_max = -workspace_radius, workspace_radius
    y_min, y_max = -workspace_radius, workspace_radius

    x_vals = np.linspace(x_min, x_max, resolution)
    y_vals = np.linspace(y_min, y_max, resolution)
    X, Y = np.meshgrid(x_vals, y_vals)

    # 初始化SEF场
    cartesian_sef = np.full_like(X, np.nan)

    # 对每个网格点计算SEF
    for i in range(resolution):
        for j in range(resolution):
            x, y = X[i, j], Y[i, j]

            # 计算逆运动学
            ik_solutions = inverse_kinematics(x, y, l1, l2)

            if ik_solutions is not None:
                # 计算两种解的SEF值
                sef_values = []
                for q1, q2 in ik_solutions:
                    # 计算关节空间的SEF
                    joint_sef = calculate_joint_sef(q1, q2, q1_opt, q2_opt, comfort_threshold, weights)
                    sef_values.append(joint_sef)

                # 选择SEF值最小的解（最舒适的配置）
                cartesian_sef[i, j] = min(sef_values)

    return X, Y, cartesian_sef, (x_opt, y_opt)


def draw_robot_arm(ax, q1, q2, l1, l2, color='k', linewidth=2, add_labels=True):
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


def plot_cartesian_sef(X, Y, SEF, optimal_point, l1, l2, q1_opt, q2_opt, title):
    """
    绘制笛卡尔空间的SEF

    参数:
    - X, Y: 笛卡尔空间网格
    - SEF: SEF值
    - optimal_point: 最优点的笛卡尔坐标
    - l1, l2: 连杆长度
    - q1_opt, q2_opt: 最优关节角度
    - title: 图表标题
    """
    # 创建图形
    fig, ax = plt.subplots(figsize=(12, 10))

    # 定义色图
    cmap = plt.cm.RdBu_r

    # 找出数据范围并设置对称的色标
    valid_data = SEF[~np.isnan(SEF)]
    if len(valid_data) > 0:
        max_abs = max(abs(np.nanmin(valid_data)), abs(np.nanmax(valid_data)))
        norm = plt.Normalize(-max_abs, max_abs)
    else:
        norm = plt.Normalize(-1, 1)

    # 填充等高线
    contour = ax.contourf(X, Y, SEF, 20, cmap=cmap, norm=norm, alpha=0.8)
    cbar = fig.colorbar(contour, ax=ax, label='SEF Value')

    # 添加0值等高线
    zero_contour = ax.contour(X, Y, SEF, [0], colors='green', linewidths=2)
    ax.clabel(zero_contour, inline=True, fontsize=10, fmt='%1.1f')

    # 标记最优点
    x_opt, y_opt = optimal_point
    ax.plot(x_opt, y_opt, 'go', markersize=10, label='Optimal Point')

    # 添加工作空间边界
    theta = np.linspace(0, 2 * np.pi, 100)
    # 内边界 (|l1-l2|)
    inner_radius = abs(l1 - l2)
    inner_circle = Circle((0, 0), inner_radius, fill=False, linestyle='--', color='gray',
                          label='Inner Workspace Boundary')
    ax.add_patch(inner_circle)

    # 外边界 (l1+l2)
    outer_radius = l1 + l2
    outer_circle = Circle((0, 0), outer_radius, fill=False, linestyle='--', color='black',
                          label='Outer Workspace Boundary')
    ax.add_patch(outer_circle)

    # 绘制最优配置下的机械臂
    draw_robot_arm(ax, q1_opt, q2_opt, l1, l2, color='blue', linewidth=3, add_labels=True)

    # 设置图形属性
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(title)
    ax.set_xlim([-outer_radius * 1.2, outer_radius * 1.2])
    ax.set_ylim([-outer_radius * 1.2, outer_radius * 1.2])
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='best')

    plt.tight_layout()
    plt.show()


def main():
    # 定义连杆长度
    l1, l2 = 1.0, 0.8

    # 定义最优关节角度
    q1_opt, q2_opt = np.pi / 4, -np.pi / 3

    # 定义舒适阈值和权重
    comfort_threshold = 0.5
    weights = [1, 1]  # q1和q2的权重相同

    # 计算笛卡尔空间的SEF
    X, Y, cartesian_sef, optimal_point = calculate_cartesian_sef(
        l1, l2, q1_opt, q2_opt, comfort_threshold, weights
    )

    # 可视化
    plot_cartesian_sef(X, Y, cartesian_sef, optimal_point, l1, l2, q1_opt, q2_opt,
                       "Cartesian Space SEF for Two-Link Robotic Arm")


if __name__ == "__main__":
    main()