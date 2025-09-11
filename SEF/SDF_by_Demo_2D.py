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


# -------------------------- CSDF: distance + gradient + projection onto polyline --------------------------

@dataclass
class CSDF:
    demo_path: np.ndarray                 # (N, D) 示教轨迹，按示教顺序排列（现用于任务空间轨迹）
    revolute_mask: Optional[np.ndarray]   # (D,) 旋转关节为 True（保持原参数，不改动）
    weight: Optional[np.ndarray] = None   # (D,) 马氏度量对角权重
    window: int = 6                       # 最近段搜索窗口
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

        # Wrapped and unwrapped path
        self.path_wrapped = self.demo_path.copy()
        if np.any(self.revolute_mask):
            self.path_wrapped[:, self.revolute_mask] = wrap_to_pi(self.path_wrapped[:, self.revolute_mask])
        self.path_unwrapped = unwrap_path(self.path_wrapped, self.revolute_mask)

        # Segments (in unwrapped space)
        self.seg_p0 = self.path_unwrapped[:-1, :]
        self.seg_p1 = self.path_unwrapped[1:, :]
        self.seg_v = self.seg_p1 - self.seg_p0
        self.num_segments = self.seg_p0.shape[0]

        # KD-tree on wrapped coords with metric transform
        Pw = self.path_wrapped @ self.L.T
        self.kdtree = KDTree(Pw) if (self.use_kdtree and SCIPY_OK) else None
        if self.kdtree is None:
            self.Pw = Pw  # fallback for brute-force

    def _nearest_waypoint_index(self, q_wrapped: np.ndarray) -> int:
        qT = (q_wrapped @ self.L.T).reshape(1, -1)
        if self.kdtree is not None:
            _, idx = self.kdtree.query(qT, k=1)
            return int(idx)
        diffs = self.Pw - qT
        d2 = np.sum(diffs * diffs, axis=1)
        return int(np.argmin(d2))

    def _distance_to_segment(self, q_unwrapped: np.ndarray, p0: np.ndarray, p1: np.ndarray) -> Tuple[float, np.ndarray, float]:
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
        """
        最近点投影 + 距离 + 梯度 + 段索引与段内参数
        Returns: y_wrapped, d, grad, info={segment, t, y_unwrapped, q_unwrapped}
        """
        q = np.asarray(q, dtype=float).reshape(-1)
        assert q.shape == (self.D,)
        idx = self._nearest_waypoint_index(q)
        anchor_unwrapped = self.path_unwrapped[idx]
        q_unwrapped = lift_query_to_unwrapped(q, anchor_unwrapped, self.revolute_mask)

        k0 = max(0, idx - self.window)
        k1 = min(self.num_segments - 1, idx + self.window)
        best = (np.inf, None, None, None)
        best_seg = k0
        best_t = 0.0
        for s in range(k0, k1 + 1):
            d, y, t = self._distance_to_segment(q_unwrapped, self.seg_p0[s], self.seg_p1[s])
            if d < best[0]:
                best = (d, y, t, s)
                best_seg = s
                best_t = t

        d, y_unwrapped, _, seg_idx = best
        if d > 1e-12:
            delta = q_unwrapped - y_unwrapped
            grad = (self.W @ delta) / d
        else:
            grad = np.zeros(self.D)

        y_wrapped = y_unwrapped.copy()
        if np.any(self.revolute_mask):
            y_wrapped[self.revolute_mask] = wrap_to_pi(y_wrapped[self.revolute_mask])

        info = dict(segment=seg_idx, t=best_t, y_unwrapped=y_unwrapped, q_unwrapped=q_unwrapped)
        return y_wrapped, d, grad, info

    def distance(self, q: np.ndarray) -> float:
        return self.project(q)[1]


# -------------------------- Adaptive directional field --------------------------

@dataclass
class AdaptiveDirectionalField:
    """
    任务空间命名版（算法不变）：
      F(x) = k_tan * alpha_tan(d) * u_tan(x) + k_norm * alpha_norm(d) * n_hat(x)
    """
    csdf: CSDF
    k_tan: float = 1.0
    k_norm: float = 1.0
    d_tan: float = 0.35
    d_norm: float = 0.35
    p_tan: float = 2.0
    p_norm: float = 1.0

    def _alpha_tan(self, d: float) -> float:
        x = max(d, 0.0) / max(1e-6, self.d_tan)
        return 1.0 / (1.0 + x ** self.p_tan)

    def _alpha_norm(self, d: float) -> float:
        x = max(d, 0.0) / max(1e-6, self.d_norm)
        return np.tanh(x) ** self.p_norm

    def field(self, q: np.ndarray) -> np.ndarray:
        _, d, grad, info = self.csdf.project(q)

        # Unit tangent from closest segment
        v = self.csdf.seg_v[info["segment"]]
        v_norm = np.linalg.norm(v)
        u_tan = np.zeros_like(v) if v_norm < 1e-12 else v / v_norm

        # Unit inward normal from gradient
        n_dir = -grad
        n_norm = np.linalg.norm(n_dir)
        n_hat = np.zeros_like(n_dir) if n_norm < 1e-12 else n_dir / n_norm

        a_tan = self._alpha_tan(d)
        a_norm = self._alpha_norm(d)
        return self.k_tan * a_tan * u_tan + self.k_norm * a_norm * n_hat


# -------------------------- Adaptive streamline integration --------------------------

@dataclass
class AdaptiveStreamlineIntegrator:
    """
    距离自适应步长（近轨小、远轨大）+ RK2 + 步幅限幅
    """
    field: AdaptiveDirectionalField
    step_near: float = 0.01
    step_far: float = 0.06
    d_ref: float = 0.35
    max_step_norm: float = 0.08
    max_step_per_dim: Optional[np.ndarray] = None  # (D,)
    curvature_slowdown: bool = True

    @staticmethod
    def _smoothstep01(x: float) -> float:
        x = np.clip(x, 0.0, 1.0)
        return x * x * (3.0 - 2.0 * x)

    def _limit_step(self, dq: np.ndarray) -> np.ndarray:
        if self.max_step_per_dim is not None:
            lim = np.asarray(self.max_step_per_dim, dtype=float)
            dq = np.clip(dq, -lim, lim)
        n = np.linalg.norm(dq)
        if self.max_step_norm is not None and self.max_step_norm > 0 and n > self.max_step_norm:
            dq = dq * (self.max_step_norm / (n + 1e-12))
        return dq

    def _step_size_by_distance(self, q: np.ndarray) -> float:
        d = self.field.csdf.distance(q)
        s = self._smoothstep01(d / max(1e-6, self.d_ref))
        return self.step_near + (self.step_far - self.step_near) * s

    def step(self, q: np.ndarray) -> np.ndarray:
        h = self._step_size_by_distance(q)

        F1 = self.field.field(q)
        k1 = self._limit_step(h * F1)
        q_mid = q + 0.5 * k1

        F2 = self.field.field(q_mid)
        k2 = self._limit_step(h * F2)
        dq = k2

        if self.curvature_slowdown:
            n1 = np.linalg.norm(F1) + 1e-9
            n2 = np.linalg.norm(F2) + 1e-9
            cos_theta = np.dot(F1, F2) / (n1 * n2)
            cos_theta = np.clip(cos_theta, -1.0, 1.0)
            theta = np.arccos(cos_theta)
            scale = 1.0 / (1.0 + (theta / (np.pi / 4.0)) ** 2)
            dq *= scale
            dq = self._limit_step(dq)

        return q + dq

    def integrate(self,
                  q_start: np.ndarray,
                  T: int = 400,
                  wrap_mask: Optional[np.ndarray] = None) -> np.ndarray:
        q = np.asarray(q_start, dtype=float).reshape(-1)
        if wrap_mask is None:
            wrap_mask = np.zeros_like(q, dtype=bool)

        Q = [q.copy()]
        for _ in range(T - 1):
            q = self.step(q)
            if np.any(wrap_mask):
                q[wrap_mask] = wrap_to_pi(q[wrap_mask])
            Q.append(q.copy())
        return np.vstack(Q)


# -------------------------- Visualization --------------------------

def compute_field_slice(field: AdaptiveDirectionalField,
                        dims: Tuple[int, int] = (0, 1),
                        bounds: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
                        res: int = 140,
                        fixed_config: Optional[np.ndarray] = None):
    csdf = field.csdf
    D = csdf.D
    i, j = dims
    assert 0 <= i < D and 0 <= j < D and i != j

    if fixed_config is None:
        fixed = np.median(csdf.path_wrapped, axis=0)
    else:
        fixed = np.asarray(fixed_config, dtype=float).copy()
        assert fixed.shape == (D,)

    if bounds is None:
        b = []
        for d in dims:
            if csdf.revolute_mask[d]:
                b.append((-np.pi, np.pi))
            else:
                pd = csdf.path_wrapped[:, d]
                pad = 0.1 * (pd.max() - pd.min() + 1e-6)
                b.append((pd.min() - pad, pd.max() + pad))
        bounds = (b[0], b[1])

    (xi_min, xi_max), (xj_min, xj_max) = bounds
    x = np.linspace(xi_min, xi_max, res)
    y = np.linspace(xj_min, xj_max, res)
    X, Y = np.meshgrid(x, y, indexing='xy')

    Z = np.zeros_like(X)   # SDF distance（任务空间命名）
    U = np.zeros_like(X)   # field component i
    V = np.zeros_like(X)   # field component j

    q = fixed.copy()
    for r in range(res):
        for c in range(res):
            q[i] = X[r, c]
            q[j] = Y[r, c]
            Z[r, c] = csdf.distance(q)
            F = field.field(q)
            U[r, c] = F[i]
            V[r, c] = F[j]

    meta = dict(bounds=bounds, dims=dims, fixed=fixed)
    return X, Y, Z, U, V, meta


def plot_csdf_and_field_with_paths(csdf: CSDF,
                                   field: AdaptiveDirectionalField,
                                   paths: Optional[List[np.ndarray]] = None,
                                   dims: Tuple[int, int] = (0, 1),
                                   bounds: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
                                   res: int = 140,
                                   quiver_step: int = 16,     # 稀疏箭头
                                   cmap: str = 'viridis',
                                   figsize: Tuple[float, float] = (8, 7),
                                   title: Optional[str] = None):
    X, Y, Z, U, V, meta = compute_field_slice(field, dims=dims, bounds=bounds, res=res)
    i, j = dims

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    # SDF 等高线（任务空间命名）
    cf = ax.contourf(X, Y, Z, levels=28, cmap=cmap)
    fig.colorbar(cf, ax=ax, label='SDF distance')

    # 方向场箭头（单位化用于显示方向）
    ss = (slice(None, None, quiver_step), slice(None, None, quiver_step))
    UU = U[ss]; VV = V[ss]
    NN = np.sqrt(UU**2 + VV**2) + 1e-9
    ax.quiver(X[ss], Y[ss], UU/NN, VV/NN, color='white', alpha=0.75, scale=25, width=0.0025, headwidth=3)

    # 示教轨迹（任务空间）
    Pw = csdf.path_wrapped
    ax.plot(Pw[:, i], Pw[:, j], color='orange', lw=2.2, label='Demo path (task space)')

    # 多条生成轨迹
    if paths:
        colors = ['red', 'deepskyblue', 'limegreen', 'magenta']
        for k, Q in enumerate(paths):
            col = colors[k % len(colors)]
            ax.plot(Q[:, i], Q[:, j], color=col, lw=2.0, label=f'Path #{k+1}')
            ax.scatter(Q[0, i], Q[0, j], c=col, s=60, marker='o', edgecolor='k', zorder=5)

    # 坐标轴命名改为任务空间 x,y
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(title or "Task-space SDF + Adaptive directional field + Paths")
    ax.legend(loc='upper right')
    return fig, ax


# -------------------------- Example: four starts, start-only --------------------------

def _example():
    np.set_printoptions(precision=3, suppress=True)

    # 2-DoF 示例：示教路径的数值定义不变，但现在把两个维度视为任务空间 (x,y)
    D = 2
    revolute_mask = np.array([True, True], dtype=bool)  # 保持原参数，不改动

    # 示教路径（按方向排列；定义不变）
    t = np.linspace(-1.6, 1.6, 260)
    demo = np.zeros((t.size, D))
    demo[:, 0] = t
    demo[:, 1] = 0.9 * np.sin(2.2 * t)
    demo[:, 0] = wrap_to_pi(demo[:, 0])
    demo[:, 1] = wrap_to_pi(demo[:, 1])

    weight = np.array([1.0, 1.0])
    csdf = CSDF(demo_path=demo, revolute_mask=revolute_mask, weight=weight, window=8, use_kdtree=True)

    # 自适应方向场（算法与参数不变）
    field = AdaptiveDirectionalField(
        csdf=csdf,
        k_tan=1.0,
        k_norm=1.0,
        d_tan=0.35,
        d_norm=0.35,
        p_tan=2.0,
        p_norm=1.0
    )

    # 自适应积分器（参数不变）
    integrator = AdaptiveStreamlineIntegrator(
        field=field,
        step_near=0.01,
        step_far=0.06,
        d_ref=0.35,
        max_step_norm=0.08,
        max_step_per_dim=np.array([0.08, 0.08]),
        curvature_slowdown=True
    )

    # 四个起点（数值保持不变）
    starts = [
        np.array([-2.4, -1.4]),
        np.array([-2.0,  1.2]),
        np.array([ 2.4, -1.0]),
        np.array([ 2.2,  1.4]),
    ]
    for s in starts:
        s[revolute_mask] = wrap_to_pi(s[revolute_mask])

    # 生成路径（打印命名改为 SDF）
    paths = []
    for q0 in starts:
        Q = integrator.integrate(q_start=q0, T=600, wrap_mask=revolute_mask)
        paths.append(Q)
        mean_d = np.mean([csdf.distance(Q[i]) for i in range(Q.shape[0])])
        print(f"Start {np.round(q0,3)}: waypoints={Q.shape[0]}, mean SDF distance={mean_d:.3f}")

    # 任务空间范围：适当增加（相对示教轨迹外扩 30% 的边距）
    pd = csdf.path_wrapped
    span_x = pd[:, 0].max() - pd[:, 0].min()
    span_y = pd[:, 1].max() - pd[:, 1].min()
    pad_x = 0.30 * (span_x + 1e-6)
    pad_y = 0.30 * (span_y + 1e-6)
    bounds = ((float(pd[:, 0].min() - pad_x), float(pd[:, 0].max() + pad_x)),
              (float(pd[:, 1].min() - pad_y), float(pd[:, 1].max() + pad_y)))

    # 可视化（坐标命名为 x/y，其他参数不变）
    plot_csdf_and_field_with_paths(
        csdf=csdf,
        field=field,
        paths=paths,
        dims=(0, 1),
        bounds=bounds,
        res=160,
        quiver_step=16,
        cmap='viridis',
        figsize=(8, 7),
        title='Task-space SDF and adaptive field (x,y)'
    )
    plt.show()


if __name__ == "__main__":
    _example()