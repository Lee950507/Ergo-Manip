import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


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
    绘制SEF的3D表面图和等高线图
    """
    # 定义色图，从蓝色（负值）到红色（正值）
    cmap = plt.cm.RdBu_r

    # 找出数据范围，进行对称的色标
    max_abs = max(abs(SEF.min()), abs(SEF.max()))
    norm = plt.Normalize(-max_abs, max_abs)

    # 3D表面图
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(Q1, Q2, SEF, cmap=cmap, norm=norm, linewidth=0, antialiased=True)

    ax.set_xlabel('q1')
    ax.set_ylabel('q2')
    ax.set_zlabel('SEF Value')
    ax.set_title(title)

    # 添加z=0平面以突出显示"表面"
    x_min, x_max = Q1.min(), Q1.max()
    y_min, y_max = Q2.min(), Q2.max()
    xx, yy = np.meshgrid([x_min, x_max], [y_min, y_max])
    zz = np.zeros_like(xx)
    ax.plot_surface(xx, yy, zz, alpha=0.2, color='green')

    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.tight_layout()
    plt.show()

    # 等高线图，突出显示0值等高线
    plt.figure(figsize=(12, 10))

    contour = plt.contourf(Q1, Q2, SEF, 20, cmap=cmap, norm=norm)
    plt.colorbar(contour)

    # 添加0值等高线并加粗
    zero_contour = plt.contour(Q1, Q2, SEF, [0], colors='green', linewidths=2)
    plt.clabel(zero_contour, inline=True, fontsize=10, fmt='%1.1f')

    # 标记最优点或区域
    if optimal_params:
        if optimal_params['type'] == 'point':
            plt.plot(optimal_params['q1_opt'], optimal_params['q2_opt'], 'go',
                     markersize=10, label='Optimal Point')
            plt.legend()
        elif optimal_params['type'] == 'region':
            q1_up = optimal_params['q1_up']
            q1_down = optimal_params['q1_down']
            q2_up = optimal_params['q2_up']
            q2_down = optimal_params['q2_down']

            # 画出最优区域
            rect = plt.Rectangle((q1_up, q2_up), q1_down - q1_up, q2_down - q2_up,
                                 fill=False, edgecolor='green', linewidth=2, label='Optimal Region')
            plt.gca().add_patch(rect)
            plt.legend()

    plt.xlabel('q1')
    plt.ylabel('q2')
    plt.title(f'{title} (Contour)')
    plt.grid(True)
    plt.tight_layout()
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
    q1_opt, q2_opt = 0, 0
    comfort_threshold = 1.0
    weights = [1, 1]  # 两个关节的权重相同

    SEF_single = calculate_sef_single_optimal(Q1, Q2, q1_opt, q2_opt, comfort_threshold, weights)
    optimal_params_single = {'type': 'point', 'q1_opt': q1_opt, 'q2_opt': q2_opt}
    plot_sef(Q1, Q2, SEF_single, "SEF - Single Optimal Point", optimal_params_single)

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