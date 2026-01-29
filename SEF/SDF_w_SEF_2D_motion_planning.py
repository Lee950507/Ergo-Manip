import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple

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
    if np.any(revolute_mask):
        delta = anchor_unwrapped - q
        n = np.zeros_like(q)
        n[revolute_mask] = np.round(delta[revolute_mask] / (2 * np.pi))
        q = q + n * (2 * np.pi)
    return q


def make_metric(weight: Optional[np.ndarray], D: int):
    if weight is None:
        weight = np.ones(D)
    weight = np.asarray(weight, dtype=float)
    assert weight.shape == (D,) and np.all(weight > 0)
    W = np.diag(weight)
    L = np.diag(np.sqrt(weight))
    return W, L


# -------------------------- CSDF: distance + projection onto polyline --------------------------

class CSDF:
    def __init__(self, demo_path, revolute_mask=None, weight=None, window=6, use_kdtree=True):
        self.demo_path = np.asarray(demo_path, dtype=float)
        assert self.demo_path.ndim == 2 and self.demo_path.shape[0] >= 2
        self.N, self.D = self.demo_path.shape

        if revolute_mask is None:
            self.revolute_mask = np.zeros(self.D, dtype=bool)
        else:
            self.revolute_mask = np.asarray(revolute_mask, dtype=bool)
            assert self.revolute_mask.shape == (self.D,)

        self.W, self.L = make_metric(weight, self.D)

        self.path_wrapped = self.demo_path.copy()
        if np.any(self.revolute_mask):
            self.path_wrapped[:, self.revolute_mask] = wrap_to_pi(self.path_wrapped[:, self.revolute_mask])
        self.path_unwrapped = unwrap_path(self.path_wrapped, self.revolute_mask)

        self.seg_p0 = self.path_unwrapped[:-1, :]
        self.seg_p1 = self.path_unwrapped[1:, :]
        self.seg_v = self.seg_p1 - self.seg_p0
        self.num_segments = self.seg_p0.shape[0]

        Pw = self.path_wrapped @ self.L.T
        self.kdtree = KDTree(Pw) if (use_kdtree and SCIPY_OK) else None
        if self.kdtree is None:
            self.Pw = Pw
        self.window = int(window)

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


# -------------------------- 2R arm: IK/SEF --------------------------

def angle_diff(a, b):
    return wrap_to_pi(a - b)

def forward_kinematics(q1, q2, l1=1.0, l2=0.8):
    x = l1 * np.cos(q1) + l2 * np.cos(q1 + q2)
    y = l1 * np.sin(q1) + l2 * np.sin(q1 + q2)
    return x, y

def inverse_kinematics_2R(x, y, l1=1.0, l2=0.8):
    r2 = x*x + y*y
    c2 = (r2 - l1*l1 - l2*l2) / (2.0*l1*l2)
    if c2 < -1.0 or c2 > 1.0:
        return []
    c2 = np.clip(c2, -1.0, 1.0)
    s2_abs = np.sqrt(max(0.0, 1.0 - c2*c2))
    q2_up = np.arctan2(+s2_abs, c2)
    q2_dn = np.arctan2(-s2_abs, c2)
    k1 = l1 + l2*c2
    alpha = np.arctan2(y, x)
    g_up = np.arctan2(l2*(+s2_abs), k1)
    g_dn = np.arctan2(l2*(-s2_abs), k1)
    q1_up = alpha - g_up
    q1_dn = alpha - g_dn
    return [np.array([wrap_to_pi(q1_up), wrap_to_pi(q2_up)]),
            np.array([wrap_to_pi(q1_dn), wrap_to_pi(q2_dn)])]

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


# -------------------------- Composite on intersection --------------------------

def compose_on_intersection(SDF, SEF,
                            w_sdf=1.0, w_sef=1.0,
                            normalize=True, q=0.9,
                            clamp_sef_negative=False):
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
    Composite[~mask] = np.nan
    return Composite, mask


# -------------------------- Sampling, gradient, demo tangent --------------------------

def bilinear_sample(grid_x, grid_y, Z, p):
    x, y = p[0], p[1]
    if x < grid_x[0] or x > grid_x[-1] or y < grid_y[0] or y > grid_y[-1]:
        return np.nan
    ix = np.searchsorted(grid_x, x) - 1
    iy = np.searchsorted(grid_y, y) - 1
    ix = np.clip(ix, 0, len(grid_x) - 2)
    iy = np.clip(iy, 0, len(grid_y) - 2)
    x0, x1 = grid_x[ix], grid_x[ix+1]
    y0, y1 = grid_y[iy], grid_y[iy+1]
    tx = 0.0 if x1 == x0 else (x - x0) / (x1 - x0)
    ty = 0.0 if y1 == y0 else (y - y0) / (y1 - y0)
    z00 = Z[iy, ix]; z10 = Z[iy, ix+1]
    z01 = Z[iy+1, ix]; z11 = Z[iy+1, ix+1]
    vals = np.array([z00, z10, z01, z11], dtype=float)
    weights = np.array([(1-tx)*(1-ty), tx*(1-ty), (1-tx)*ty, tx*ty], dtype=float)
    m = np.isfinite(vals)
    if not np.any(m):
        return np.nan
    return float(np.sum(weights[m] * vals[m]) / (np.sum(weights[m]) + 1e-12))


def numeric_gradient(grid_x, grid_y, Z, p, eps=1e-3):
    f0 = bilinear_sample(grid_x, grid_y, Z, p)
    if not np.isfinite(f0):
        return np.array([0.0, 0.0]), np.nan
    ex = np.array([eps, 0.0]); ey = np.array([0.0, eps])
    fpx = bilinear_sample(grid_x, grid_y, Z, p + ex)
    fmx = bilinear_sample(grid_x, grid_y, Z, p - ex)
    fpy = bilinear_sample(grid_x, grid_y, Z, p + ey)
    fmy = bilinear_sample(grid_x, grid_y, Z, p - ey)

    def diff(fp, fm, e):
        if np.isfinite(fp) and np.isfinite(fm):
            return (fp - fm) / (2*e)
        elif np.isfinite(fp) and np.isfinite(f0):
            return (fp - f0) / e
        elif np.isfinite(f0) and np.isfinite(fm):
            return (f0 - fm) / e
        else:
            return 0.0

    gx = diff(fpx, fmx, eps)
    gy = diff(fpy, fmy, eps)
    return np.array([gx, gy], dtype=float), f0


def nearest_point_and_tangent_on_demo(csdf: CSDF, p: np.ndarray):
    y_wrapped, d, grad, info = csdf.project(p)
    seg_idx = info["segment"]
    v = csdf.seg_v[seg_idx]
    v_norm = np.linalg.norm(v)
    u_tan = np.zeros_like(v) if v_norm < 1e-12 else v / v_norm
    p_proj = y_wrapped
    return p_proj, d, u_tan


# -------------------------- Planner (no animation) --------------------------

@dataclass
class PlannerConfig:
    xs: np.ndarray
    ys: np.ndarray
    SDF: np.ndarray
    csdf: CSDF

    # SEF/robot params
    l1: float = 1.0
    l2: float = 0.8
    q_opt: np.ndarray = np.array([np.pi/4, -np.pi/3])
    weights_sef: np.ndarray = np.array([1.0, 1.0])
    comfort_threshold: float = 0.5
    q_window_half: Tuple[float, float] = (np.pi/6, np.pi/6)  # 以当前 q 为中心的窗口半宽

    # composite weights (SDF vs SEF 的比重)
    w_sdf: float = 1.0
    w_sef: float = 1.0
    normalize_q: float = 0.9
    clamp_sef_negative: bool = False

    # tangent guidance (轨迹切向 vs 复合场负梯度 的比重)
    beta_tan_max: float = 1.0   # 切向最大权重（相对负梯度的加权）
    d_tan_ref: float = 0.4      # 距离参考，越小表示更靠近轨迹才强调切向
    p_tan: float = 2.0          # 距离-权重映射的幂指数

    # stepping / termination
    step_init: float = 0.06
    step_min: float = 0.004
    step_max: float = 0.12
    ls_shrink: float = 0.5
    max_iters: int = 200
    tol_cost: float = 1e-4
    tol_step: float = 2e-3
    grad_eps: float = 1e-3


def angle_L1(q, qref, w=np.array([1.0, 1.0])):
    dq = wrap_to_pi(q - qref)
    return np.sum(np.abs(w) * np.abs(dq))


def pick_ik_near(q_prev, x, y, cfg: PlannerConfig):
    sols = inverse_kinematics_2R(x, y, cfg.l1, cfg.l2)
    if not sols:
        return None
    q1_half, q2_half = cfg.q_window_half
    joint_bounds = ((float(q_prev[0]-q1_half), float(q_prev[0]+q1_half)),
                    (float(q_prev[1]-q2_half), float(q_prev[1]+q2_half)))
    (q1min, q1max), (q2min, q2max) = joint_bounds

    valid = []
    for s in sols:
        s1, s2 = wrap_to_pi(s[0]), wrap_to_pi(s[1])
        if _angle_in_bounds(s1, q1min, q1max) and _angle_in_bounds(s2, q2min, q2max):
            valid.append(np.array([s1, s2]))
    if not valid:
        return None
    dists = [angle_L1(v, q_prev) for v in valid]
    return valid[int(np.argmin(dists))]


def build_sef_and_composite_at_q(xs, ys, SDF, q_center, cfg: PlannerConfig):
    q1_half, q2_half = cfg.q_window_half
    joint_bounds = ((float(q_center[0]-q1_half), float(q_center[0]+q1_half)),
                    (float(q_center[1]-q2_half), float(q_center[1]+q2_half)))
    X, Y = np.meshgrid(xs, ys, indexing='xy')
    SEF = sef_on_grid_via_ik_window_L1(
        X, Y, cfg.l1, cfg.l2,
        cfg.q_opt, cfg.weights_sef, cfg.comfort_threshold,
        joint_bounds
    )
    Composite, mask = compose_on_intersection(
        SDF, SEF,
        w_sdf=cfg.w_sdf, w_sef=cfg.w_sef,
        normalize=True, q=cfg.normalize_q,
        clamp_sef_negative=cfg.clamp_sef_negative
    )
    return SEF, Composite, mask


def plan_once(q_start, cfg: PlannerConfig):
    # 初值
    q = wrap_to_pi(np.array(q_start, dtype=float))
    x0, y0 = forward_kinematics(q[0], q[1], cfg.l1, cfg.l2)
    p = np.array([x0, y0], dtype=float)

    Qs = [q.copy()]
    Ps = [p.copy()]
    costs = []

    # 初始化 SEF/Composite（SDF 已缓存）
    SEF, Composite, mask = build_sef_and_composite_at_q(cfg.xs, cfg.ys, cfg.SDF, q, cfg)
    _, c = numeric_gradient(cfg.xs, cfg.ys, Composite, p, eps=cfg.grad_eps)
    costs.append(c if np.isfinite(c) else np.nan)

    step = cfg.step_init

    for it in range(cfg.max_iters):
        g, c = numeric_gradient(cfg.xs, cfg.ys, Composite, p, eps=cfg.grad_eps)
        if not np.isfinite(c):
            print(f"[iter {it}] Composite undefined at p={p}. Stop.")
            break
        if np.linalg.norm(g) < 1e-9:
            print(f"[iter {it}] grad ~ 0. Stop.")
            break
        d_grad = -g / (np.linalg.norm(g) + 1e-12)

        # 切向引导
        _, d_demo, u_tan = nearest_point_and_tangent_on_demo(cfg.csdf, p)
        beta = cfg.beta_tan_max / (1.0 + (max(d_demo,0.0) / max(1e-6, cfg.d_tan_ref)) ** cfg.p_tan)
        d = d_grad + beta * u_tan
        d = d / (np.linalg.norm(d) + 1e-12)

        # 线搜索：SEF覆盖内且代价下降
        success = False
        step_try = float(np.clip(step, cfg.step_min, cfg.step_max))
        for _ in range(18):
            p_try = p + step_try * d
            p_try[0] = np.clip(p_try[0], cfg.xs[0], cfg.xs[-1])
            p_try[1] = np.clip(p_try[1], cfg.ys[0], cfg.ys[-1])
            mask_val = bilinear_sample(cfg.xs, cfg.ys, mask.astype(float), p_try)
            if not np.isfinite(mask_val) or mask_val < 0.5:
                step_try *= cfg.ls_shrink
                continue
            c_try = bilinear_sample(cfg.xs, cfg.ys, Composite, p_try)
            if not np.isfinite(c_try) or c_try > c - 1e-6:
                step_try *= cfg.ls_shrink
                continue
            success = True
            break
        if not success:
            print(f"[iter {it}] line search failed. Stop.")
            break

        # 受限 IK（在前一解附近）
        q_next = pick_ik_near(q, p_try[0], p_try[1], cfg)
        if q_next is None:
            back_ok = False
            for _ in range(8):
                step_try *= cfg.ls_shrink
                if step_try < cfg.step_min:
                    break
                p_try = p + step_try * d
                p_try[0] = np.clip(p_try[0], cfg.xs[0], cfg.xs[-1])
                p_try[1] = np.clip(p_try[1], cfg.ys[0], cfg.ys[-1])
                mask_val = bilinear_sample(cfg.xs, cfg.ys, mask.astype(float), p_try)
                if not (np.isfinite(mask_val) and mask_val >= 0.5):
                    continue
                q_next = pick_ik_near(q, p_try[0], p_try[1], cfg)
                if q_next is not None:
                    back_ok = True
                    break
            if not back_ok:
                print(f"[iter {it}] IK near previous failed. Stop.")
                break

        # 接受步
        q = q_next
        p = p_try
        Qs.append(q.copy())
        Ps.append(p.copy())
        costs.append(c_try)
        step = float(np.clip(step_try * 1.2, cfg.step_min, cfg.step_max))

        # 收敛
        if len(Ps) >= 2:
            dp = np.linalg.norm(Ps[-1] - Ps[-2])
            dc = abs(costs[-1] - costs[-2])
            if dp < cfg.tol_step or dc < cfg.tol_cost:
                print(f"[iter {it}] Converged: dp={dp:.3e}, dc={dc:.3e}")
                break

        # 动态更新 SEF/Composite（SDF 复用）
        SEF, Composite, mask = build_sef_and_composite_at_q(cfg.xs, cfg.ys, cfg.SDF, q, cfg)

    # 结束时刻的 SEF/Composite（使用终点 q）
    SEF_final, Composite_final, mask_final = build_sef_and_composite_at_q(cfg.xs, cfg.ys, cfg.SDF, q, cfg)
    return np.vstack(Qs), np.vstack(Ps), np.array(costs), SEF_final, Composite_final, mask_final


# -------------------------- Build fields and run --------------------------

def main():
    np.set_printoptions(precision=3, suppress=True)

    # Demo path（保持定义）
    D = 2
    revolute_mask = np.array([True, True], dtype=bool)
    t = np.linspace(-1.6, 1.6, 260)
    demo = np.zeros((t.size, D))
    demo[:, 0] = t
    demo[:, 1] = 0.9 * np.sin(2.2 * t)
    demo[:, 0] = wrap_to_pi(demo[:, 0])
    demo[:, 1] = wrap_to_pi(demo[:, 1])
    csdf = CSDF(demo_path=demo, revolute_mask=revolute_mask, weight=np.array([1.0, 1.0]), window=8, use_kdtree=True)

    # Grid
    bounds = ((-3.0, 3.0), (-3.0, 3.0))
    res = 241
    xs = np.linspace(bounds[0][0], bounds[0][1], res)
    ys = np.linspace(bounds[1][0], bounds[1][1], res)
    X, Y = np.meshgrid(xs, ys, indexing='xy')

    # SDF（仅计算一次，后续复用）
    print("Computing SDF grid once ...")
    SDF = compute_sdf_grid_from_csdf(csdf, X, Y)

    # Config（可调参数见注释）
    cfg = PlannerConfig(
        xs=xs, ys=ys, SDF=SDF, csdf=csdf,
        l1=1.0, l2=0.8,
        q_opt=np.array([np.pi/4, -np.pi/3]),
        weights_sef=np.array([1.0, 1.0]),
        comfort_threshold=0.5,
        q_window_half=(np.pi/6, np.pi/6),
        # 复合场权重（SDF vs SEF）
        w_sdf=1.0, w_sef=1.0,
        normalize_q=0.9, clamp_sef_negative=False,
        # 切向引导权重
        beta_tan_max=1.0, d_tan_ref=0.4, p_tan=2.0,
        # 步进与收敛
        step_init=0.06, step_min=0.004, step_max=0.12, ls_shrink=0.5,
        max_iters=160, tol_cost=1e-4, tol_step=2e-3, grad_eps=1e-3
    )

    # 起点（可改）
    q_start = np.array([np.pi/3, np.pi/2])

    # 规划
    Qs, Ps, costs, SEF_final, Composite_final, mask_final = plan_once(q_start, cfg)

    # 可视化：仅呈现最终时刻的 SEF 与 Composite，并叠加路径
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    # SEF final
    ax = axes[0]
    v = SEF_final[np.isfinite(SEF_final)]
    vmax = float(max(abs(np.nanmin(v)), abs(np.nanmax(v)))) if v.size > 0 else 1.0
    c1 = ax.contourf(X, Y, SEF_final, levels=32, cmap='RdBu_r', vmin=-vmax, vmax=+vmax)
    fig.colorbar(c1, ax=ax, label='SEF (final)')
    try:
        ax.contour(X, Y, SEF_final, levels=[0.0], colors='lime', linewidths=1.8)
    except Exception:
        pass
    ax.plot(Ps[:,0], Ps[:,1], 'k.-', lw=2, ms=4, label='planned path')
    ax.scatter(Ps[0,0], Ps[0,1], c='red', s=50, edgecolor='k', zorder=5, label='start')
    ax.plot(demo[:,0], demo[:,1], color='orange', lw=2.0, label='demo')
    ax.set_aspect('equal'); ax.grid(True, alpha=0.3); ax.set_title('SEF (final)')
    ax.legend(loc='best')

    # Composite final
    ax = axes[1]
    c2 = ax.contourf(X, Y, Composite_final, levels=32, cmap='magma')
    fig.colorbar(c2, ax=ax, label='Composite (final)')
    try:
        ax.contour(X, Y, mask_final.astype(float), levels=[0.5], colors='cyan', linewidths=1.5)
    except Exception:
        pass
    ax.plot(Ps[:,0], Ps[:,1], 'w.-', lw=2, ms=4, label='planned path')
    ax.scatter(Ps[0,0], Ps[0,1], c='red', s=50, edgecolor='k', zorder=5)
    ax.plot(demo[:,0], demo[:,1], color='orange', lw=2.0)
    ax.set_aspect('equal'); ax.grid(True, alpha=0.3); ax.set_title('Composite (final)')

    for ax in axes:
        ax.set_xlabel('x'); ax.set_ylabel('y')
        ax.set_xlim(xs[0], xs[-1]); ax.set_ylim(ys[0], ys[-1])

    plt.show()

    # 输出终点
    print("Final q:", np.round(Qs[-1], 4))
    print("Final p:", np.round(Ps[-1], 4))
    print("Final cost:", costs[-1])


if __name__ == "__main__":
    main()