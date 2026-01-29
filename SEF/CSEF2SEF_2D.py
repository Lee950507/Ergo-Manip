import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# ---------------- Utilities ----------------

def wrap_to_pi(x):
    return (x + np.pi) % (2*np.pi) - np.pi

def angle_diff(a, b):
    return wrap_to_pi(a - b)

def forward_kinematics(q1, q2, l1=1.0, l2=0.8):
    x = l1*np.cos(q1) + l2*np.cos(q1 + q2)
    y = l1*np.sin(q1) + l2*np.sin(q1 + q2)
    return x, y

def _angle_in_bounds(q, qmin, qmax):
    """判断角 q（都视为 wrap 后）是否在 [qmin,qmax]（允许跨越 π 裂缝）"""
    q = wrap_to_pi(q); qmin = wrap_to_pi(qmin); qmax = wrap_to_pi(qmax)
    if qmin <= qmax:
        return (q >= qmin) & (q <= qmax)
    else:
        return (q >= qmin) | (q <= qmax)

def estimate_task_bounds_from_joint_window(l1, l2, joint_bounds, res=80, margin_ratio=0.03):
    """在关节窗口上均匀采样，通过 FK 估计任务空间可视范围（带少量边距）"""
    (q1min, q1max), (q2min, q2max) = joint_bounds
    q1s = np.linspace(q1min, q1max, res)
    q2s = np.linspace(q2min, q2max, res)
    Q1, Q2 = np.meshgrid(q1s, q2s, indexing='xy')
    x, y = forward_kinematics(Q1.ravel(), Q2.ravel(), l1, l2)
    x_min, x_max = float(np.min(x)), float(np.max(x))
    y_min, y_max = float(np.min(y)), float(np.max(y))
    # 边距
    dx = x_max - x_min; dy = y_max - y_min
    x_pad = margin_ratio * (dx if dx > 1e-9 else 1.0)
    y_pad = margin_ratio * (dy if dy > 1e-9 else 1.0)
    return (x_min - x_pad, x_max + x_pad), (y_min - y_pad, y_max + y_pad)

# ---------------- Analytic IK assignment with joint-window filter (L1 metric) ----------------

def cartesian_sef_via_ik_joint_window_L1(
    l1, l2,
    q_opt, weights, comfort_threshold,
    x_bounds, x_res,
    joint_bounds
):
    """
    对任务空间网格上的每个点：
      - 解析 IK 得到两支解（肘上/肘下）；
      - 把不在 joint_bounds 内的解丢弃；
      - 对保留分支用 L1 距离：phi = w1*|dq1| + w2*|dq2| - tau（dq 使用 angle_diff）；
      - 取两支的最小值；无解或两支均被丢弃则为 NaN。
    """
    q_opt = np.asarray(q_opt, dtype=float).reshape(2)
    w1, w2 = float(weights[0]), float(weights[1])
    (xmin, xmax), (ymin, ymax) = x_bounds

    xs = np.linspace(xmin, xmax, int(x_res))
    ys = np.linspace(ymin, ymax, int(x_res))
    X, Y = np.meshgrid(xs, ys, indexing='xy')

    r2 = X*X + Y*Y
    outer_R = l1 + l2
    inner_R = abs(l1 - l2)
    reachable = (r2 <= (outer_R + 1e-12)**2) & (r2 >= (inner_R - 1e-12)**2)

    # 解析 IK 两分支（矢量化）
    cos_q2 = (r2 - l1*l1 - l2*l2) / (2.0*l1*l2)
    cos_q2 = np.clip(cos_q2, -1.0, 1.0)
    sin_q2_abs = np.sqrt(np.maximum(0.0, 1.0 - cos_q2*cos_q2))

    q2_up   = np.arctan2(+sin_q2_abs, cos_q2)
    q2_down = np.arctan2(-sin_q2_abs, cos_q2)

    k1 = l1 + l2*cos_q2
    alpha = np.arctan2(Y, X)
    gamma_up   = np.arctan2(l2*(+sin_q2_abs), k1)
    gamma_down = np.arctan2(l2*(-sin_q2_abs), k1)

    q1_up   = alpha - gamma_up
    q1_down = alpha - gamma_down

    # wrap 到 [-π,π]
    q1_up_w, q2_up_w = wrap_to_pi(q1_up), wrap_to_pi(q2_up)
    q1_dn_w, q2_dn_w = wrap_to_pi(q1_down), wrap_to_pi(q2_down)

    # 关节窗口过滤
    (q1min, q1max), (q2min, q2max) = joint_bounds
    in_up   = _angle_in_bounds(q1_up_w, q1min, q1max) & _angle_in_bounds(q2_up_w, q2min, q2max)
    in_down = _angle_in_bounds(q1_dn_w, q1min, q1max) & _angle_in_bounds(q2_dn_w, q2min, q2max)

    # L1 度量（用环空间角差）
    dq1_up = angle_diff(q1_up_w, q_opt[0]); dq2_up = angle_diff(q2_up_w, q_opt[1])
    dq1_dn = angle_diff(q1_dn_w, q_opt[0]); dq2_dn = angle_diff(q2_dn_w, q_opt[1])

    phi_up = w1*np.abs(dq1_up) + w2*np.abs(dq2_up) - float(comfort_threshold)
    phi_dn = w1*np.abs(dq1_dn) + w2*np.abs(dq2_dn) - float(comfort_threshold)

    # 仅窗口内有效
    phi_up = np.where(in_up, phi_up, np.nan)
    phi_dn = np.where(in_down, phi_dn, np.nan)

    # 两分支取最小 + 工作空间掩膜
    SEF = np.nanmin(np.stack([phi_up, phi_dn], axis=0), axis=0)
    SEF = np.where(reachable, SEF, np.nan)

    return X, Y, SEF, dict(reachable=reachable, in_up=in_up, in_down=in_down)

# ---------------- Visualization ----------------

def plot_sef(X, Y, SEF, l1, l2, title, q_star=None, stars=None):
    fig, ax = plt.subplots(figsize=(10, 8))

    valid = SEF[~np.isnan(SEF)]
    if valid.size > 0:
        vmax = float(max(abs(np.nanmin(valid)), abs(np.nanmax(valid))))
        vmax = vmax if vmax > 1e-12 else 1.0
        norm = plt.Normalize(-vmax, vmax)
    else:
        norm = plt.Normalize(-1, 1)

    cf = ax.contourf(X, Y, SEF, levels=30, cmap='RdBu_r', norm=norm)
    fig.colorbar(cf, ax=ax, label='SEF (L1)')

    try:
        ax.contour(X, Y, SEF, levels=[0.0], colors='lime', linewidths=2)
    except Exception:
        pass

    # 工作空间边界
    outer_R = l1 + l2
    inner_R = abs(l1 - l2)
    ax.add_patch(Circle((0, 0), inner_R, fill=False, linestyle='--', color='gray'))
    ax.add_patch(Circle((0, 0), outer_R, fill=False, linestyle='--', color='black'))

    if q_star is not None:
        # 标注理论最优点在任务空间的位置
        x_star, y_star = forward_kinematics(q_star[0], q_star[1], l1, l2)
        ax.plot(x_star, y_star, marker='*', color='gold', markersize=14, markeredgecolor='k', label='predicted min (q*)')

    if stars:
        for (qx, qy, name, color) in stars:
            xx, yy = forward_kinematics(qx, qy, l1, l2)
            ax.plot(xx, yy, marker='o', color=color, markersize=8, label=name)

    ax.set_xlabel('X'); ax.set_ylabel('Y')
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(title)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='best')
    plt.tight_layout()
    plt.show()

# ---------------- Example with your exact conditions ----------------

def main():
    # 参数设置（按你提供）
    l1, l2 = 1.0, 0.8
    q_opt = np.array([np.pi / 4, -np.pi / 3])    # (π/4, -π/3)
    q_current = np.array([-np.pi / 3, -np.pi / 2]) # (π/3,  π/2)
    weights = np.array([1.0, 1.0])
    comfort_threshold = 0.5

    q1_range = np.pi / 4  # ±30°
    q2_range = np.pi / 4  # ±30°

    # 关节窗口（中心 q_current）
    joint_bounds = (
        (float(q_current[0] - q1_range), float(q_current[0] + q1_range)),  # [π/6, π/2]
        (float(q_current[1] - q2_range), float(q_current[1] + q2_range)),  # [π/3, 2π/3]
    )
    print("Joint window:")
    print(f" q1 in [{joint_bounds[0][0]:.3f}, {joint_bounds[0][1]:.3f}]")
    print(f" q2 in [{joint_bounds[1][0]:.3f}, {joint_bounds[1][1]:.3f}]")

    # 理论最优 q*：q1*=clip(q1_opt)，q2*取窗口端点中 angle_diff 距离最小者
    q1_min, q1_max = joint_bounds[0]
    q2_min, q2_max = joint_bounds[1]
    q1_star = np.clip(q_opt[0], q1_min, q1_max)
    # 比较两个端点到 q2_opt 的环空间距离
    d_min = abs(angle_diff(q2_min, q_opt[1]))
    d_max = abs(angle_diff(q2_max, q_opt[1]))
    q2_star = q2_min if d_min <= d_max else q2_max
    q_star = np.array([q1_star, q2_star])
    print(f"Predicted minimizer in joint window: q* = [{q_star[0]:.3f}, {q_star[1]:.3f}] "
          f"(should be [π/4, π/3] ≈ [{np.pi/4:.3f}, {np.pi/3:.3f}])")

    # 用窗口采样估计任务空间可视范围，再进行 IK 赋值（L1）
    x_bounds = estimate_task_bounds_from_joint_window(l1, l2, joint_bounds, res=120, margin_ratio=0.03)
    x_res = 240  # 网格分辨率，适当提高保证定位精度
    X, Y, SEF, info = cartesian_sef_via_ik_joint_window_L1(
        l1, l2,
        q_opt, weights, comfort_threshold,
        x_bounds, x_res,
        joint_bounds
    )

    # 在理论最优点处做数值检查
    d1 = abs(angle_diff(q_star[0], q_opt[0]))
    d2 = abs(angle_diff(q_star[1], q_opt[1]))
    sef_star_true = weights[0]*d1 + weights[1]*d2 - comfort_threshold
    # 找到 q* 的 FK 对应网格索引
    x_s, y_s = forward_kinematics(q_star[0], q_star[1], l1, l2)
    ix = int(np.argmin(np.abs(X[0, :] - x_s)))
    iy = int(np.argmin(np.abs(Y[:, 0] - y_s)))
    sef_star_grid = SEF[iy, ix]
    print(f"Check at q* (grid vs. true): SEF_grid = {sef_star_grid:.6f}, SEF_true = {sef_star_true:.6f}")

    # 画图并标注 q*、q_current、q_opt 的末端位置
    stars = [
        (q_current[0], q_current[1], 'q_current', 'red'),
        (q_opt[0], q_opt[1], 'q_opt', 'green'),
    ]
    plot_sef(X, Y, SEF, l1, l2,
             title='Task-space SEF via analytic IK + joint-window filter (L1 metric)',
             q_star=q_star, stars=stars)

if __name__ == "__main__":
    main()