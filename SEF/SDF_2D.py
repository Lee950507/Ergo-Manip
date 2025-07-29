import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import math


def calculate_sef_single_optimal(Q1, Q2, q1_opt, q2_opt, comfort_threshold=0.5, weights=None):
    """
    计算单一最优点情况下的SEF

    参数:
    - Q1, Q2: 网格点的坐标
    - q1_opt, q2_opt: 最优点的坐标
    - comfort_threshold: 舒适阈值，小于此阈值的距离为负值
    - weights: 关节权重，默认为[1, 1]表示两个关节权重相同
    """
    if weights is None:
        weights = [1, 1]

    # 计算加权距离
    distance = np.sqrt(weights[0] * (Q1 - q1_opt) ** 2 + weights[1] * (Q2 - q2_opt) ** 2)

    # 转换为带符号的场：小于阈值的为负，大于阈值的为正
    return distance - comfort_threshold


def calculate_sdf_circle(Q1, Q2, center_x, center_y, radius):
    """
    计算圆形的SDF（Signed Distance Function）

    参数:
    - Q1, Q2: 网格点的坐标
    - center_x, center_y: 圆心坐标
    - radius: 圆的半径
    """
    # 计算到圆心的距离
    distance = np.sqrt((Q1 - center_x) ** 2 + (Q2 - center_y) ** 2)

    # SDF值：距离减去半径（内部为负，外部为正）
    return distance - radius


def calculate_sef_optimal_region(Q1, Q2, q1_up, q1_down, q2_up, q2_down):
    """
    计算最优区域情况下的SEF，区域内为负，区域外为正

    参数:
    - Q1, Q2: 网格点的坐标
    - q1_up, q1_down: q1的最优范围上下限
    - q2_up, q2_down: q2的最优范围上下限
    """
    # 创建掩码，区分区域内和区域外的点
    inside_mask = (q1_up <= Q1) & (Q1 <= q1_down) & (q2_up <= Q2) & (Q2 <= q2_down)

    # 初始化SEF场
    sef = np.zeros_like(Q1)

    # 对于区域内的点，计算到边界的最短距离并取负值
    if np.any(inside_mask):
        dist_to_q1_up = Q1 - q1_up
        dist_to_q1_down = q1_down - Q1
        dist_to_q2_up = Q2 - q2_up
        dist_to_q2_down = q2_down - Q2

        min_dist = np.minimum(
            np.minimum(dist_to_q1_up, dist_to_q1_down),
            np.minimum(dist_to_q2_up, dist_to_q2_down)
        )

        sef[inside_mask] = -min_dist[inside_mask]

    # 对于区域外的点，计算到最优区域边界的最短距离
    if np.any(~inside_mask):
        dx = np.maximum(0, q1_up - Q1, Q1 - q1_down)
        dy = np.maximum(0, q2_up - Q2, Q2 - q2_down)

        sef[~inside_mask] = np.sqrt(dx[~inside_mask] ** 2 + dy[~inside_mask] ** 2)

    return sef


def plot_sef(Q1, Q2, SEF, title, optimal_params=None):
    """
    绘制SEF的3D表面图和等高线图，显示等高线的数值并增大标签字体
    """
    # 设置更大的字体大小
    plt.rcParams.update({'font.size': 14})

    # 定义色图，从蓝色（负值）到红色（正值）
    cmap = plt.cm.RdBu_r

    # 找出数据范围，进行对称的色标
    max_abs = max(abs(SEF.min()), abs(SEF.max()))
    norm = plt.Normalize(-max_abs, max_abs)

    # 3D表面图
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(Q1, Q2, SEF, cmap=cmap, norm=norm, linewidth=0, antialiased=True)

    ax.set_xlabel('q1', fontsize=16)
    ax.set_ylabel('q2', fontsize=16)
    ax.set_zlabel('SEF Value', fontsize=16)
    ax.set_title(title, fontsize=18)

    # 添加z=0平面以突出显示"表面"
    x_min, x_max = Q1.min(), Q1.max()
    y_min, y_max = Q2.min(), Q2.max()
    xx, yy = np.meshgrid([x_min, x_max], [y_min, y_max])
    zz = np.zeros_like(xx)
    ax.plot_surface(xx, yy, zz, alpha=0.2, color='green')

    # 添加坐标轴和z平面交点的标注
    ax.text(x_min, y_min, 0, "SEF=0", color='darkgreen', fontsize=14,
            horizontalalignment='center', verticalalignment='center')

    plt.tight_layout()
    plt.show()

    # 等高线图，突出显示0值等高线以及其他等高线的值
    plt.figure(figsize=(12, 12))

    # 定义更多的等高线级别，以便显示更多的值
    levels = np.linspace(-max_abs, max_abs, 21)  # 20个间隔，21个级别

    contour_filled = plt.contourf(Q1, Q2, SEF, levels=levels, cmap=cmap, norm=norm)

    # 添加带标签的等高线
    # 选择部分等高线进行标记以避免过度拥挤
    contour_levels = np.linspace(-max_abs, max_abs, 11)  # 减少等高线数量，避免标签重叠
    contours = plt.contour(Q1, Q2, SEF, levels=contour_levels, colors='black', linewidths=0.8)
    plt.clabel(contours, inline=True, fontsize=28, fmt='%.1f')

    # 添加0值等高线并加粗
    zero_contour = plt.contour(Q1, Q2, SEF, [0], colors='green', linewidths=3)
    plt.clabel(zero_contour, inline=True, fontsize=28, fmt='%.1f')

    # 标记最优点或区域
    if optimal_params:
        if optimal_params['type'] == 'point':
            plt.plot(optimal_params['q1_opt'], optimal_params['q2_opt'], 'go',
                     markersize=12)
        elif optimal_params['type'] == 'region':
            q1_up = optimal_params['q1_up']
            q1_down = optimal_params['q1_down']
            q2_up = optimal_params['q2_up']
            q2_down = optimal_params['q2_down']

            # 画出最优区域
            rect = plt.Rectangle((q1_up, q2_up), q1_down - q1_up, q2_down - q2_up,
                                 fill=False, edgecolor='green', linewidth=3)
            plt.gca().add_patch(rect)

    plt.xlabel('q1', fontsize=30)
    plt.ylabel('q2', fontsize=30)
    plt.tick_params(axis='x', labelsize=30)
    plt.tick_params(axis='y', labelsize=30)
    plt.title(f'{title} (Contour)', fontsize=18)
    plt.tight_layout()
    plt.savefig("SEF_2D.png", dpi=800)
    plt.show()


def plot_sdf(Q1, Q2, SDF, title, center=None, radius=None):
    """
    绘制SDF的等高线图
    """
    # 设置更大的字体大小
    plt.rcParams.update({'font.size': 18})

    # 定义色图，从蓝色（负值）到红色（正值）
    cmap = plt.cm.RdBu_r

    # 找出数据范围，进行对称的色标
    max_abs = max(abs(SDF.min()), abs(SDF.max()))
    norm = plt.Normalize(-max_abs, max_abs)

    plt.figure(figsize=(12, 12))

    # 定义等高线级别
    levels = np.linspace(-max_abs, max_abs, 21)  # 20个间隔，21个级别

    contour_filled = plt.contourf(Q1, Q2, SDF, levels=levels, cmap=cmap, norm=norm)

    # 添加带标签的等高线
    contour_levels = np.linspace(-max_abs, max_abs, 11)  # 减少等高线数量，避免标签重叠
    contours = plt.contour(Q1, Q2, SDF, levels=contour_levels, colors='black', linewidths=0.8)
    plt.clabel(contours, inline=True, fontsize=28, fmt='%.1f')

    # 添加0值等高线并加粗
    zero_contour = plt.contour(Q1, Q2, SDF, [0], colors='green', linewidths=3)
    plt.clabel(zero_contour, inline=True, fontsize=28, fmt='%.1f')

    # 如果提供了圆心和半径，则绘制圆形
    if center is not None and radius is not None:
        circle = plt.Circle(center, radius, fill=False, edgecolor='green', linewidth=3)
        plt.gca().add_patch(circle)

    plt.xlabel('x', fontsize=30)
    plt.ylabel('y', fontsize=30)
    plt.tick_params(axis='x', labelsize=30)
    plt.tick_params(axis='y', labelsize=30)
    plt.title(title, fontsize=18)
    plt.tight_layout()
    plt.savefig("SDF_2D.png", dpi=800)
    plt.show()


def main():
    # 定义关节空间的范围
    q1_min, q1_max = -np.pi, np.pi
    q2_min, q2_max = -np.pi, np.pi

    # 创建网格
    resolution = 100
    q1 = np.linspace(q1_min, q1_max, resolution)
    q2 = np.linspace(q2_min, q2_max, resolution)
    Q1, Q2 = np.meshgrid(q1, q2)

    # 情况1：单一最优点
    q1_opt, q2_opt = math.pi / 3, math.pi / 6
    comfort_threshold = 0.5
    weights = [1, 2]  # q2的权重是q1的两倍

    SEF_single = calculate_sef_single_optimal(Q1, Q2, q1_opt, q2_opt, comfort_threshold, weights)
    optimal_params_single = {'type': 'point', 'q1_opt': q1_opt, 'q2_opt': q2_opt}
    plot_sef(Q1, Q2, SEF_single, "SEF - Single Optimal Point", optimal_params_single)

    # 添加一个SDF的2D平面图
    # 为SDF创建一个新的网格，范围为[-2, 2]
    x_min, x_max = -2, 2
    y_min, y_max = -2, 2
    resolution = 100
    x = np.linspace(x_min, x_max, resolution)
    y = np.linspace(y_min, y_max, resolution)
    X, Y = np.meshgrid(x, y)

    # 定义一个圆形SDF
    circle_center = (0.5, 0.5)
    circle_radius = 0.5

    SDF_circle = calculate_sdf_circle(X, Y, circle_center[0], circle_center[1], circle_radius)
    plot_sdf(X, Y, SDF_circle,'', circle_center, circle_radius)

    # # 如果需要，可以取消注释以下代码，展示最优区域的情况
    # # 情况2：最优区域
    # q1_up, q1_down = -0.5, 0.5
    # q2_up, q2_down = -0.5, 0.5
    #
    # SEF_region = calculate_sef_optimal_region(Q1, Q2, q1_up, q1_down, q2_up, q2_down)
    # optimal_params_region = {
    #     'type': 'region',
    #     'q1_up': q1_up, 'q1_down': q1_down,
    #     'q2_up': q2_up, 'q2_down': q2_down
    # }
    # plot_sef(Q1, Q2, SEF_region, "SEF - Optimal Region", optimal_params_region)


if __name__ == "__main__":
    main()