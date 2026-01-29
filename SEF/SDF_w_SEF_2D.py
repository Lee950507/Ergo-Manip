import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, List

import matplotlib.pyplot as plt

try:
    from scipy.spatial import cKDTree as KDTree
    SCIPY_OK = True
except Exception:
    KDTree = None
    SCIPY_OK = False


# -------------------------- Basic utilities --------------------------

def wrap_to_pi(x: np.ndarray) -> np.ndarray:
    return (x + np.pi) % (2 * np.pi) - np.pi


def unwrap_path(path: np.ndarray, revolute_mask: np.ndarray) -> np.ndarray:
    path = np.asarray(path, dtype=float)
    out = path.copy()
    if not np.any(revolute_mask):
        return out
    for d, is_rev in enumerate(revolute_mask):
        if not is_rev:
            continue
        diffs = np.diff(out[:, d])
        diffs_wrapped = (diffs + np.pi) % (2 * np.pi) - np.pi
        out[1:, d] = out[0, d] + np.cumsum(diffs_wrapped)
    return out


def lift_query_to_unwrapped(q_wrapped: np.ndarray, anchor_unwrapped: np.ndarray, revolute_mask: np.ndarray) -> np.ndarray:
    q = q_wrapped.copy()
    q_lifted = q.copy()
    if np.any(revolute_mask):
        delta = anchor_unwrapped - q
        n = np.zeros_like(q)
        n[revolute_mask] = np.round(delta[revolute_mask] / (2 * np.pi))
        q_lifted = q + n * (2 * np.pi)
    return q_lifted


def make_metric(weight: Optional[np.ndarray], D: int) -> Tuple[np.ndarray, np.ndarray]:
    if weight is None:
        weight = np.ones(D)
    weight = np.asarray(weight, dtype=float)
    assert weight.shape == (D,)
    assert np.all(weight > 0)
    W = np.diag(weight)
    L = np.diag(np.sqrt(weight))
    return W, L


# -------------------------- CSDF: distance + projection onto polyline --------------------------

@dataclass
class CSDF:
    demo_path: np.ndarray
    revolute_mask: Optional[np.ndarray]
    weight: Optional[np.ndarray] = None
    window: int = 6
    use_kdtree: bool = True

    def __post_init__(self):
        self.demo_path = np.asarray(self.demo_path, dtype=float)
        assert self.demo_path.ndim == 2 and self.demo_path.shape[0] >= 2
        self.N, self.D = self.demo_path.shape

        if self.revolute_mask is None:
            self.revolute_mask = np.zeros(self.D, dtype=bool)
        else:
            self.revolute_mask = np.asarray(self.revolute_mask, dtype=bool)
            assert self.revolute_mask.shape == (self.D,)

        self.W, self.L = make_metric(self.weight, self.D)

        self.path_wrapped = self.demo_path.copy()
        if np.any(self.revolute_mask):
            self.path_wrapped[:, self.revolute_mask] = wrap_to_pi(self.path_wrapped[:, self.revolute_mask])
        self.path_unwrapped = unwrap_path(self.path_wrapped, self.revolute_mask)

        self.seg_p0 = self.path_unwrapped[:-1, :]
        self.seg_p1 = self.path_unwrapped[1:, :]
        self.seg_v = self.seg_p1 - self.seg_p0
        self.num_segments = self.seg_p0.shape[0]

        Pw = self.path_wrapped @ self.L.T
        self.kdtree = KDTree(Pw) if (self.use_kdtree and SCIPY_OK) else None
        if self.kdtree is None:
            self.Pw = Pw

    def _nearest_waypoint_index(self, q_wrapped: np.ndarray) -> int:
        qT = (q_wrapped @ self.L.T).reshape(1, -1)
        if self.kdtree is not None:
            _, idx = self.kdtree.query(qT, k=1)
            return int(idx)
        diffs = self.Pw - qT
        d2 = np.sum(diffs * diffs, axis=1)
        return int(np.argmin(d2))

    def _distance_to_segment(self, q_unwrapped: np.ndarray, p0: np.ndarray, p1: np.ndarray):
        v = p1 - p0
        vWv = float(v @ self.W @ v)
        if vWv <= 1e-12:
            delta = q_unwrapped - p0
            d2 = float(delta @ self.W @ delta)
            return np.sqrt(max(0.0, d2)), p0.copy(), 0.0
        t = float((q_unwrapped - p0) @ self.W @ v) / vWv
        t_clamped = max(0.0, min(1.0, t))
        y = p0 + t_clamped * v
        delta = q_unwrapped - y
        d2 = float(delta @ self.W @ delta)
        return np.sqrt(max(0.0, d2)), y, t_clamped

    def project(self, q: np.ndarray):
        q = np.asarray(q, dtype=float).reshape(-1)
        assert q.shape == (self.D,)
        idx = self._nearest_waypoint_index(q)
        anchor_unwrapped = self.path_unwrapped[idx]
        q_unwrapped = lift_query_to_unwrapped(q, anchor_unwrapped, self.revolute_mask)

        k0 = max(0, idx - self.window)
        k1 = min(self.num_segments - 1, idx + self.window)
        best_d = np.inf
        best_y = None
        best_t = 0.0
        best_seg = k0

        for s in range(k0, k1 + 1):
            d, y, t = self._distance_to_segment(q_unwrapped, self.seg_p0[s], self.seg_p1[s])
            if d < best_d:
                best_d = d
                best_y = y
                best_seg = s
                best_t = t

        d = float(best_d)
        y_unwrapped = best_y
        if d > 1e-12:
            delta = q_unwrapped - y_unwrapped
            grad = (self.W @ delta) / d
        else:
            grad = np.zeros(self.D)

        y_wrapped = y_unwrapped.copy()
        if np.any(self.revolute_mask):
            y_wrapped[self.revolute_mask] = wrap_to_pi(y_wrapped[self.revolute_mask])

        info = dict(segment=best_seg, t=best_t, y_unwrapped=y_unwrapped, q_unwrapped=q_unwrapped)
        return y_wrapped, d, grad, info

    def distance(self, q: np.ndarray) -> float:
        return self.project(q)[1]


def compute_sdf_grid_from_csdf(csdf: CSDF, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    ny, nx = X.shape
    Z = np.zeros_like(X, dtype=float)
    for r in range(ny):
        for c in range(nx):
            q = np.array([X[r, c], Y[r, c]], dtype=float)
            Z[r, c] = csdf.distance(q)
    return Z


# -------------------------- 2R arm: SEF on the same grid (analytic IK + joint window, L1) --------------------------

def angle_diff(a, b):
    return wrap_to_pi(a - b)


def forward_kinematics(q1, q2, l1=1.0, l2=0.8):
    x = l1 * np.cos(q1) + l2 * np.cos(q1 + q2)
    y = l1 * np.sin(q1) + l2 * np.sin(q1 + q2)
    return x, y


def _angle_in_bounds(q, qmin, qmax):
    q = wrap_to_pi(q); qmin = wrap_to_pi(qmin); qmax = wrap_to_pi(qmax)
    if qmin <= qmax:
        return (q >= qmin) & (q <= qmax)
    else:
        return (q >= qmin) | (q <= qmax)


def sef_on_grid_via_ik_window_L1(
    X, Y,
    l1, l2,
    q_opt, weights, comfort_threshold,
    joint_bounds
) -> np.ndarray:
    q_opt = np.asarray(q_opt, dtype=float).reshape(2)
    w1, w2 = float(weights[0]), float(weights[1])

    r2 = X*X + Y*Y
    outer_R = l1 + l2
    inner_R = abs(l1 - l2)
    reachable = (r2 <= (outer_R + 1e-12)**2) & (r2 >= (inner_R - 1e-12)**2)

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

    q1_up_w, q2_up_w = wrap_to_pi(q1_up), wrap_to_pi(q2_up)
    q1_dn_w, q2_dn_w = wrap_to_pi(q1_down), wrap_to_pi(q2_down)

    (q1min, q1max), (q2min, q2max) = joint_bounds
    in_up   = _angle_in_bounds(q1_up_w, q1min, q1max) & _angle_in_bounds(q2_up_w, q2min, q2max)
    in_down = _angle_in_bounds(q1_dn_w, q1min, q1max) & _angle_in_bounds(q2_dn_w, q2min, q2max)

    dq1_up = angle_diff(q1_up_w, q_opt[0]); dq2_up = angle_diff(q2_up_w, q_opt[1])
    dq1_dn = angle_diff(q1_dn_w, q_opt[0]); dq2_dn = angle_diff(q2_dn_w, q_opt[1])

    phi_up = w1*np.abs(dq1_up) + w2*np.abs(dq2_up) - float(comfort_threshold)
    phi_dn = w1*np.abs(dq1_dn) + w2*np.abs(dq2_dn) - float(comfort_threshold)

    phi_up = np.where(in_up, phi_up, np.nan)
    phi_dn = np.where(in_down, phi_dn, np.nan)

    SEF = np.nanmin(np.stack([phi_up, phi_dn], axis=0), axis=0)
    SEF = np.where(reachable, SEF, np.nan)
    return SEF


# -------------------------- Composite: use only intersection (where SEF is valid) --------------------------

def compose_on_intersection(SDF, SEF,
                            w_sdf=1.0, w_sef=1.0,
                            normalize=True, q=0.9,
                            clamp_sef_negative=False):
    """
    仅在 SEF 有值的位置上合成复合场。
    - 掩膜 M = isfinite(SEF)
    - SDF、SEF 先按掩膜提取后再归一化（避免把掩膜外的值影响尺度）
    - Composite = w_sdf * S_norm + w_sef * E_norm；掩膜外设为 NaN
    """
    mask = np.isfinite(SEF)
    S_masked = np.where(mask, SDF, np.nan)
    E_masked = np.where(mask, SEF, np.nan)
    if clamp_sef_negative:
        E_masked = np.where(mask, np.maximum(E_masked, 0.0), np.nan)

    if normalize:
        def ref_scale(arr, qv):
            v = arr[np.isfinite(arr)]
            if v.size == 0:
                return 1.0
            ref = np.quantile(np.abs(v), qv)
            return float(ref if ref > 1e-12 else 1.0)
        s_ref = ref_scale(S_masked, q)
        e_ref = ref_scale(E_masked, q)
        S_norm = S_masked / s_ref
        E_norm = E_masked / e_ref
    else:
        S_norm, E_norm = S_masked, E_masked

    Composite = w_sdf * S_norm + w_sef * E_norm
    # 掩膜外维持 NaN
    Composite[~mask] = np.nan
    return Composite, mask


# -------------------------- Visualization --------------------------

def plot_maps_intersection(X, Y, SDF, SEF, Composite, mask, demo_path, l1, l2,
                           titles=('Task-space SDF (full grid)', 'SEF (no fill, NaN outside coverage)', 'Composite (only where SEF exists)')):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)

    def draw_overlays(ax):
        outer_R = l1 + l2
        inner_R = abs(l1 - l2)
        ax.add_patch(plt.Circle((0, 0), inner_R, fill=False, linestyle='--', color='gray', alpha=0.6))
        ax.add_patch(plt.Circle((0, 0), outer_R, fill=False, linestyle='--', color='black', alpha=0.6))
        ax.plot(demo_path[:, 0], demo_path[:, 1], color='orange', lw=2.0, label='Demo path')
        ax.legend(loc='best')

    # SDF（全范围）
    ax = axes[0]
    v = SDF[np.isfinite(SDF)]
    vmax = float(np.quantile(v, 0.95)) if v.size > 0 else 1.0
    cf = ax.contourf(X, Y, SDF, levels=32, cmap='viridis', vmin=0.0, vmax=vmax)
    fig.colorbar(cf, ax=ax, label='distance')
    ax.set_title(titles[0]); ax.set_aspect('equal'); ax.grid(True, alpha=0.3); draw_overlays(ax)

    # SEF（不填补）
    ax = axes[1]
    v = SEF[np.isfinite(SEF)]
    vmax = float(max(abs(np.nanmin(v)), abs(np.nanmax(v)))) if v.size > 0 else 1.0
    cf = ax.contourf(X, Y, SEF, levels=32, cmap='RdBu_r', vmin=-vmax, vmax=+vmax)
    try:
        ax.contour(X, Y, SEF, levels=[0.0], colors='lime', linewidths=2.0)
    except Exception:
        pass
    fig.colorbar(cf, ax=ax, label='SEF (L1)')
    ax.set_title(titles[1]); ax.set_aspect('equal'); ax.grid(True, alpha=0.3); draw_overlays(ax)

    # Composite（仅 SEF 区域）
    ax = axes[2]
    cf = ax.contourf(X, Y, Composite, levels=32, cmap='magma')
    fig.colorbar(cf, ax=ax, label='Composite')
    # 叠加 SEF 掩膜轮廓
    try:
        ax.contour(X, Y, mask.astype(float), levels=[0.5], colors='cyan', linewidths=1.8)
    except Exception:
        pass
    ax.set_title(titles[2]); ax.set_aspect('equal'); ax.grid(True, alpha=0.3); draw_overlays(ax)

    for ax in axes:
        ax.set_xlabel('x'); ax.set_ylabel('y')

    plt.show()


# -------------------------- Main --------------------------

def main():
    np.set_printoptions(precision=3, suppress=True)

    # 1) 示教轨迹（保持你的定义）
    D = 2
    revolute_mask = np.array([True, True], dtype=bool)
    t = np.linspace(-1.6, 1.6, 260)
    demo = np.zeros((t.size, D))
    demo[:, 0] = t
    demo[:, 1] = 0.9 * np.sin(2.2 * t)
    demo[:, 0] = wrap_to_pi(demo[:, 0])
    demo[:, 1] = wrap_to_pi(demo[:, 1])

    weight = np.array([1.0, 1.0])
    csdf = CSDF(demo_path=demo, revolute_mask=revolute_mask, weight=weight, window=8, use_kdtree=True)

    # 2) 网格范围 [-3,3] × [-3,3]
    bounds = ((-3.0, 3.0), (-3.0, 3.0))
    res = 241
    xs = np.linspace(bounds[0][0], bounds[0][1], res)
    ys = np.linspace(bounds[1][0], bounds[1][1], res)
    X, Y = np.meshgrid(xs, ys, indexing='xy')

    # 3) SDF（全范围）
    print("Computing SDF on grid ...")
    SDF = compute_sdf_grid_from_csdf(csdf, X, Y)

    # 4) SEF（同一网格上，IK + 关节窗口过滤，L1；不进行 NaN 填补）
    l1, l2 = 1.0, 0.8
    q_opt = np.array([np.pi / 4, -np.pi / 3])
    q_current = np.array([np.pi / 2,  np.pi / 2])
    weights_sef = np.array([1.0, 1.0])
    comfort_threshold = 0.5
    q1_range = np.pi / 6
    q2_range = np.pi / 6
    joint_bounds = (
        (float(q_current[0] - q1_range), float(q_current[0] + q1_range)),
        (float(q_current[1] - q2_range), float(q_current[1] + q2_range)),
    )
    print("Computing SEF on same grid ...")
    SEF = sef_on_grid_via_ik_window_L1(
        X, Y, l1, l2,
        q_opt, weights_sef, comfort_threshold,
        joint_bounds
    )

    # 5) 仅在 SEF 有值的区域上构建复合场
    Composite, mask = compose_on_intersection(
        SDF, SEF,
        w_sdf=1.0, w_sef=1.0,
        normalize=True, q=0.9,
        clamp_sef_negative=False
    )

    # 6) 可视化：SDF（全范围）、SEF（不填补）、Composite（仅 SEF 区域）
    plot_maps_intersection(
        X, Y, SDF, SEF, Composite, mask,
        demo_path=demo,
        l1=l1, l2=l2
    )


if __name__ == "__main__":
    main()