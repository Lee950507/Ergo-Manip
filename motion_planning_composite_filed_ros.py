#!/usr/bin/env python3
import numpy as np
import math
import matplotlib.pyplot as plt
import utils
import transformation as tsf
import main_opt_static as mos

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.spatial.transform import Rotation as R
from itertools import product
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable, get_cmap
from scipy.interpolate import CubicSpline

import sys
import os
import signal
import time

import tkinter as tk
from tkinter import messagebox

from utils import plot_skeleton

# 获取 ROS 工作空间的路径
workspace_path = '/home/clover/catkin_ws'

# 添加编译后的库路径
sys.path.append(os.path.join(workspace_path, 'devel', 'lib'))

last_relative_pose_wrists = None
last_object_pose = None
global ind
ind = 1


def signal_handler(sig, frame):
    print('Python shutdown signal received...')
    print('Python shutdown.')
    sys.exit(0)


def transform_to_pose(pose_stamped):
    return np.array([
        pose_stamped.pose.position.x,
        pose_stamped.pose.position.y,
        pose_stamped.pose.position.z,
        pose_stamped.pose.orientation.x,
        pose_stamped.pose.orientation.y,
        pose_stamped.pose.orientation.z,
        pose_stamped.pose.orientation.w
    ])


def compress_bounds(joint_angle_bounds, q, compression_factor=0.5):
    new_bounds = []
    joint_center = q

    for i, (lower, upper) in enumerate(joint_angle_bounds):
        range_half = (upper - lower) * compression_factor / 2
        center = joint_center[i]
        new_lower = center - range_half
        new_upper = center + range_half

        new_bounds.append((new_lower, new_upper))

    return new_bounds


def trans_shoulder2global(joint_pos, shoulder_pos, arm='right'):
    if arm == 'left':
        joint_pos[[0, 1]] = -joint_pos[[1, 0]]
        joint_pos[1] = -joint_pos[1]
        joint_pos = joint_pos + shoulder_pos
    if arm == 'right':
        joint_pos[[0, 1]] = -joint_pos[[1, 0]]
        joint_pos = joint_pos + shoulder_pos
    return joint_pos


def trans_global2shoulder(shoulder, elbow, wrist, arm='left'):
    if arm == 'left':
        elbow_new = elbow - shoulder
        elbow_new = np.array([elbow_new[1], -elbow_new[0], elbow_new[2]])
        wrist_new = wrist - shoulder
        wrist_new = np.array([wrist_new[1], -wrist_new[0], wrist_new[2]])
    if arm == 'right':
        elbow_new = elbow - shoulder
        elbow_new = np.array([-elbow_new[1], -elbow_new[0], elbow_new[2]])
        wrist_new = wrist - shoulder
        wrist_new = np.array([-wrist_new[1], -wrist_new[0], wrist_new[2]])
    return elbow_new, wrist_new


def trans_global2shoulder_single(shoulder, point, arm='right'):
    """将全局坐标转换为肩部坐标系"""
    point_new = point - shoulder
    if arm == 'right':
        point_new = np.array([-point_new[1], -point_new[0], point_new[2]])
    elif arm == 'left':
        point_new = np.array([point_new[1], -point_new[0], point_new[2]])
    return point_new


# ==================== 辅助函数 ====================

def wrap_to_pi(x):
    """将角度归一化到 [-π, π]"""
    return (x + np.pi) % (2 * np.pi) - np.pi


def angle_diff(a, b):
    """计算两个角度的差值（考虑周期性）"""
    return wrap_to_pi(a - b)


# ==================== 雅可比矩阵计算 ====================

def compute_jacobian(q, d_uar, d_lar, eps=1e-6):
    """
    计算任务空间雅可比矩阵 J = ∂p/∂q

    Args:
        q: 关节角度 [4,]
        d_uar, d_lar: 手臂长度
        eps: 数值微分步长

    Returns:
        J: 雅可比矩阵 [3, 4]
    """
    J = np.zeros((3, 4))

    # 当前末端位置
    _, p_current = mos.forward_kinematics(q, d_uar, d_lar)

    # 对每个关节进行数值微分
    for i in range(4):
        q_plus = q.copy()
        q_plus[i] += eps
        _, p_plus = mos.forward_kinematics(q_plus, d_uar, d_lar)

        # 中心差分
        J[:, i] = (p_plus - p_current) / eps

    return J


def differential_ik(q_current, p_current, p_target, d_uar, d_lar,
                    damping=0.01, max_iter=10, tolerance=1e-4):
    """
    使用微分逆运动学求解关节角度

    Args:
        q_current: 当前关节角度 [4,]
        p_current: 当前末端位置 [3,]
        p_target: 目标末端位置 [3,]
        d_uar, d_lar: 手臂长度
        damping: 阻尼系数（用于阻尼最小二乘法）
        max_iter: 最大迭代次数
        tolerance: 收敛容差

    Returns:
        q_new: 新的关节角度 [4,]
    """
    q = q_current.copy()

    for iteration in range(max_iter):
        # 计算当前位置
        _, p = mos.forward_kinematics(q, d_uar, d_lar)

        # 计算位置误差
        dp = p_target - p
        error = np.linalg.norm(dp)

        # 如果误差足够小，返回
        if error < tolerance:
            return q

        # 计算雅可比矩阵
        J = compute_jacobian(q, d_uar, d_lar)

        # 阻尼最小二乘法（Damped Least Squares）
        # dq = J^T (J J^T + λ^2 I)^(-1) dp
        JJT = J @ J.T
        damping_matrix = damping ** 2 * np.eye(3)

        try:
            dq = J.T @ np.linalg.solve(JJT + damping_matrix, dp)
        except np.linalg.LinAlgError:
            # 如果矩阵奇异，使用伪逆
            dq = J.T @ np.linalg.pinv(JJT + damping_matrix) @ dp

        # 限制步长
        max_step = 0.1  # 最大关节角度变化
        step_norm = np.linalg.norm(dq)
        if step_norm > max_step:
            dq = dq * (max_step / step_norm)

        # 更新关节角度
        q = q + dq

        # 关节限位
        q = np.clip(q,
                    [b[0] for b in joint_angle_bounds],
                    [b[1] for b in joint_angle_bounds])

    return q


# ==================== CSDF: 基于参考轨迹的距离场 ====================

class CSDF_3D:
    """3D任务空间的CSDF，用于计算到参考轨迹的距离"""

    def __init__(self, demo_path, weight=None):
        """
        Args:
            demo_path: 参考轨迹路径点 [N, 3] (任务空间坐标)
            weight: 各维度权重 [3,]
        """
        self.demo_path = np.asarray(demo_path, dtype=float)
        assert self.demo_path.ndim == 2 and self.demo_path.shape[1] == 3
        self.N = self.demo_path.shape[0]

        # 权重矩阵
        if weight is None:
            weight = np.ones(3)
        self.W = np.diag(weight)

        # 构建线段
        self.seg_p0 = self.demo_path[:-1, :]
        self.seg_p1 = self.demo_path[1:, :]
        self.seg_v = self.seg_p1 - self.seg_p0
        self.num_segments = self.seg_p0.shape[0]

        # 构建KD树用于快速最近点查询
        from scipy.spatial import cKDTree as KDTree
        self.kdtree = KDTree(self.demo_path)

    def _nearest_waypoint_index(self, p):
        """找到最近的路径点索引"""
        _, idx = self.kdtree.query(p.reshape(1, -1), k=1)
        return int(idx)

    def _distance_to_segment(self, p, p0, p1):
        """计算点到线段的距离"""
        v = p1 - p0
        vWv = float(v @ self.W @ v)

        if vWv <= 1e-12:
            delta = p - p0
            d2 = float(delta @ self.W @ delta)
            return np.sqrt(max(0.0, d2)), p0.copy(), 0.0

        t = float((p - p0) @ self.W @ v) / vWv
        t_clamped = max(0.0, min(1.0, t))
        y = p0 + t_clamped * v
        delta = p - y
        d2 = float(delta @ self.W @ delta)
        return np.sqrt(max(0.0, d2)), y, t_clamped

    def project(self, p, window=6):
        """
        投影点到参考轨迹

        Returns:
            y: 投影点
            d: 距离
            grad: 梯度（指向远离轨迹的方向）
            u_tan: 切向单位向量
        """
        p = np.asarray(p, dtype=float).reshape(3)
        idx = self._nearest_waypoint_index(p)

        k0 = max(0, idx - window)
        k1 = min(self.num_segments - 1, idx + window)

        best_d = np.inf
        best_y = None
        best_seg = k0

        for s in range(k0, k1 + 1):
            d, y, t = self._distance_to_segment(p, self.seg_p0[s], self.seg_p1[s])
            if d < best_d:
                best_d = d
                best_y = y
                best_seg = s

        d = float(best_d)
        y = best_y

        # 计算梯度
        if d > 1e-12:
            delta = p - y
            grad = (self.W @ delta) / d
        else:
            grad = np.zeros(3)

        # 计算切向
        v = self.seg_v[best_seg]
        v_norm = np.linalg.norm(v)
        u_tan = np.zeros(3) if v_norm < 1e-12 else v / v_norm

        return y, d, grad, u_tan

    def distance(self, p):
        """计算点到参考轨迹的距离"""
        _, d, _, _ = self.project(p)
        return d


# ==================== TSEF: 任务空间人体工学场 ====================

def calculate_sef_value(q, q_opt, weights, comfort_threshold):
    """
    计算关节空间的SEF值

    Args:
        q: 当前关节角度 [4,]
        q_opt: 最优关节角度 [4,]
        weights: 关节权重 [4,]
        comfort_threshold: 舒适度阈值

    Returns:
        phi: SEF值（负值表示舒适区域）
    """
    dq = np.array([angle_diff(q[i], q_opt[i]) for i in range(len(q))])
    phi = np.sum(weights * np.abs(dq)) - comfort_threshold
    return phi


def calculate_tsef_gradient(p, q, cfg, d_uar, d_lar, shoulder, eps=1e-4):
    """
    计算TSEF在任务空间的梯度（修正版：使用连续梯度）

    Args:
        p: 任务空间位置 [3,]
        q: 当前关节配置 [4,]
        cfg: 配置对象
        d_uar, d_lar: 手臂长度
        shoulder: 肩部位置
        eps: 数值微分步长

    Returns:
        grad: TSEF梯度 [3,] (任务空间)
        phi: 当前SEF值
    """
    # 当前SEF值
    phi = calculate_sef_value(q, cfg['q_opt'], cfg['weights_sef'], cfg['comfort_threshold'])

    # 计算雅可比矩阵 J = ∂p/∂q, shape: [3, 4]
    J = compute_jacobian(q, d_uar, d_lar)

    # 计算SEF在关节空间的梯度, shape: [4,]
    # 使用平滑的梯度，而不是符号函数
    dq = np.array([angle_diff(q[i], cfg['q_opt'][i]) for i in range(len(q))])

    # 平滑的梯度：∂φ/∂q = w * 2 * (q - q_opt) / comfort_threshold
    # 这样在舒适区内梯度较小，远离舒适区时梯度增大
    grad_q = cfg['weights_sef'] * dq / (cfg['comfort_threshold'] + 1e-6)

    # 映射到任务空间
    try:
        J_pinv = np.linalg.pinv(J, rcond=1e-6)  # shape: [4, 3]
        grad = J_pinv.T @ grad_q  # shape: [3, 4] @ [4,] = [3,]
    except np.linalg.LinAlgError:
        print("  Warning: Pseudoinverse failed, using zero gradient")
        grad = np.zeros(3)

    return grad, phi


def composite_field_planning(
        p_start, p_goal,
        q_start, q_opt,
        demo_path,
        shoulder, d_uar, d_lar,
        weights_sef, comfort_threshold,
        w_sdf=1.0, w_sef=0.8, w_goal=1.2,
        step_size=0.008,  # 减小步长
        max_iters=300,
        tol_goal=0.04
):
    """
    使用复合场进行运动规划（修正版）

    关键改进：
    1. 使用关节空间速度控制代替IK
    2. 平滑的梯度计算
    3. 更好的场组合策略
    4. 增加轨迹平滑机制
    """
    # 构建CSDF
    csdf = CSDF_3D(demo_path)

    # 配置对象
    cfg = {
        'q_opt': q_opt,
        'weights_sef': weights_sef,
        'comfort_threshold': comfort_threshold
    }

    # 初始化
    p = p_start.copy()
    q = q_start.copy()

    trajectory = [p.copy()]
    joint_trajectory = [q.copy()]
    sef_values = []
    sdf_values = []

    # 速度平滑（指数移动平均）
    v_prev = np.zeros(3)
    dq_prev = np.zeros(4)
    alpha_smooth = 0.7  # 平滑系数

    # 自适应步长
    adaptive_step = step_size

    print(f"\n开始复合场规划:")
    print(f"  起点: {p_start}")
    print(f"  终点: {p_goal}")
    print(f"  初始关节: {q_start}")
    print(f"  最优关节: {q_opt}")

    for it in range(max_iters):
        # 检查是否到达目标
        dist_to_goal = np.linalg.norm(p_goal - p)
        if dist_to_goal < tol_goal:
            print(f"✓ 在第 {it} 次迭代到达目标")
            break

        # ========== 1. 计算SDF场 ==========
        y_proj, d_sdf, grad_sdf, u_tan = csdf.project(p)
        sdf_values.append(d_sdf)

        # SDF势场：使用排斥力模型
        # F_sdf = -k * ∇d (指向参考轨迹)
        if d_sdf > 1e-6:
            F_sdf = -grad_sdf  # 负梯度指向轨迹
        else:
            F_sdf = np.zeros(3)

        # ========== 2. 计算TSEF场 ==========
        grad_sef, phi_sef = calculate_tsef_gradient(p, q, cfg, d_uar, d_lar, shoulder)
        sef_values.append(phi_sef)

        # TSEF势场：负梯度指向更舒适区域
        F_sef = -grad_sef

        # ========== 3. 目标吸引力 ==========
        d_goal_vec = p_goal - p
        dist_goal = np.linalg.norm(d_goal_vec)

        if dist_goal > 1e-6:
            # 使用吸引势场模型：F = k * d * d_hat
            F_goal = d_goal_vec  # 线性吸引
        else:
            F_goal = np.zeros(3)

        # ========== 4. 自适应权重 ==========
        # 距离参考轨迹的权重：远离时增大
        w_sdf_adaptive = w_sdf * (1.0 + np.tanh(d_sdf / 0.1))

        # 目标权重：接近时增大
        if dist_goal < 0.3:
            w_goal_adaptive = w_goal * (1.0 + 3.0 * (0.3 - dist_goal) / 0.3)
        else:
            w_goal_adaptive = w_goal

        # SEF权重：在不舒适区增大
        if phi_sef > 0:
            w_sef_adaptive = w_sef * (1.0 + phi_sef / cfg['comfort_threshold'])
        else:
            w_sef_adaptive = w_sef * 0.5  # 舒适区内降低权重

        # ========== 5. 组合势场力 ==========
        # 归一化各个力
        F_sdf_norm = F_sdf / (np.linalg.norm(F_sdf) + 1e-9)
        F_sef_norm = F_sef / (np.linalg.norm(F_sef) + 1e-9)
        F_goal_norm = F_goal / (np.linalg.norm(F_goal) + 1e-9)

        # 加权组合
        F_total = (w_goal_adaptive * F_goal_norm +
                   w_sdf_adaptive * F_sdf_norm +
                   w_sef_adaptive * F_sef_norm)

        # 归一化总力
        F_norm = np.linalg.norm(F_total)
        if F_norm > 1e-9:
            v_desired = F_total / F_norm
        else:
            v_desired = F_goal_norm  # 默认朝向目标

        # ========== 6. 速度平滑 ==========
        v = alpha_smooth * v_desired + (1.0 - alpha_smooth) * v_prev
        v = v / (np.linalg.norm(v) + 1e-9)
        v_prev = v.copy()

        # ========== 7. 自适应步长 ==========
        # 根据场的变化调整步长
        if it > 0:
            if dist_to_goal > 0.2:
                adaptive_step = step_size * 1.2  # 远离目标时加速
            elif dist_to_goal < 0.1:
                adaptive_step = step_size * 0.5  # 接近目标时减速
            else:
                adaptive_step = step_size

        # ========== 8. 更新位置（任务空间）==========
        p_new = p + adaptive_step * v

        # ========== 9. 关节空间更新（使用雅可比）==========
        # 计算任务空间速度
        dp = p_new - p

        # 通过雅可比计算关节速度
        J = compute_jacobian(q, d_uar, d_lar)

        try:
            # 使用阻尼最小二乘法求解关节速度
            # dq = J^T (J J^T + λI)^(-1) dp
            lambda_damping = 0.01
            JJT = J @ J.T
            dq = J.T @ np.linalg.solve(JJT + lambda_damping * np.eye(3), dp)
        except:
            # 备选方案：使用伪逆
            J_pinv = np.linalg.pinv(J)
            dq = J_pinv @ dp

        # 限制关节速度
        max_joint_vel = 0.05  # rad per step
        dq_norm = np.linalg.norm(dq)
        if dq_norm > max_joint_vel:
            dq = dq * (max_joint_vel / dq_norm)

        # 关节速度平滑
        dq = alpha_smooth * dq + (1.0 - alpha_smooth) * dq_prev
        dq_prev = dq.copy()

        # 更新关节角度
        q_new = q + dq

        # 关节限位
        q_new = np.clip(q_new,
                        [b[0] for b in joint_angle_bounds],
                        [b[1] for b in joint_angle_bounds])

        # ========== 10. 验证并接受新状态 ==========
        # 通过正运动学验证位置
        _, p_fk = mos.forward_kinematics(q_new, d_uar, d_lar)
        p_fk_global = trans_shoulder2global(p_fk, shoulder, arm='right')

        # 检查位置误差
        pos_error = np.linalg.norm(p_fk_global - p_new)

        if pos_error < 0.05:  # 误差可接受
            p = p_fk_global  # 使用FK的结果确保一致性
            q = q_new
        else:
            # 如果误差太大，使用IK修正
            p_new_shoulder = trans_global2shoulder_single(shoulder, p_new, arm='right')
            try:
                q_corrected = differential_ik(
                    q_current=q,
                    p_current=mos.forward_kinematics(q, d_uar, d_lar)[1],
                    p_target=p_new_shoulder,
                    d_uar=d_uar,
                    d_lar=d_lar,
                    damping=0.02,
                    max_iter=5,
                    tolerance=1e-4
                )
                q = q_corrected
                _, p_corrected = mos.forward_kinematics(q, d_uar, d_lar)
                p = trans_shoulder2global(p_corrected, shoulder, arm='right')
            except:
                # IK失败，保持原状态但仍向前移动
                p = p_new

        # 记录轨迹
        trajectory.append(p.copy())
        joint_trajectory.append(q.copy())

        # ========== 11. 打印进度 ==========
        if it % 30 == 0 or it < 10:
            print(f"  [{it:3d}] dist_goal={dist_to_goal:.4f}, d_sdf={d_sdf:.4f}, "
                  f"phi_sef={phi_sef:.4f}, step={adaptive_step:.4f}")
            print(f"       weights: goal={w_goal_adaptive:.2f}, sdf={w_sdf_adaptive:.2f}, "
                  f"sef={w_sef_adaptive:.2f}")

    print(f"\n规划完成: {len(trajectory)} 个轨迹点")
    print(f"  最终距离目标: {np.linalg.norm(trajectory[-1] - p_goal):.4f}m")
    print(f"  平均SDF: {np.mean(sdf_values):.4f}m")
    print(f"  SEF改善: {sef_values[0]:.4f} -> {sef_values[-1]:.4f}")

    return np.array(trajectory), np.array(joint_trajectory), np.array(sef_values), np.array(sdf_values)


# ==================== 集成到现有代码 ====================

def run_composite_field_planning(num_iterations=150):
    """
    替换现有的 run_iterations 函数（修正版：确保初始状态一致）
    """
    global current_q, global_positions, trajectory_hand, trajectory_elbow, score_history, joint_history, d_uar, d_lar

    # ========== 1. 正确初始化关节配置和位置 ==========
    # 起始关节配置
    q_start = current_q.copy()

    # 通过正运动学计算真实的起始位置（确保一致性）
    shoulder = global_positions[3].copy()
    _, p_start_shoulder = mos.forward_kinematics(q_start, d_uar, d_lar)
    p_start = trans_shoulder2global(p_start_shoulder, shoulder, arm='right')

    # 更新 global_positions[5] 以确保一致性
    global_positions[5] = p_start.copy()

    # 同样更新肘部位置
    elbow_shoulder, _ = mos.forward_kinematics(q_start, d_uar, d_lar)
    global_positions[4] = trans_shoulder2global(elbow_shoulder, shoulder, arm='right')

    print(f"\n初始化检查:")
    print(f"  q_start: {q_start}")
    print(f"  p_start (从FK计算): {p_start}")
    print(f"  global_positions[5]: {global_positions[5]}")
    print(f"  位置差异: {np.linalg.norm(p_start - global_positions[5]):.6f}m")

    # ========== 2. 设置目标 ==========
    q_opt = optimal_q.copy()
    _, p_goal_shoulder = mos.forward_kinematics(q_opt, d_uar, d_lar)
    p_goal = trans_shoulder2global(p_goal_shoulder, shoulder, arm='right')

    print(f"  q_opt: {q_opt}")
    print(f"  p_goal (从FK计算): {p_goal}")
    print(f"  optimal_position: {optimal_position}")
    print(f"  目标位置差异: {np.linalg.norm(p_goal - optimal_position):.6f}m")

    # ========== 3. 生成参考路径 ==========
    if len(trajectory_hand) > 1:
        demo_path = np.array(trajectory_hand)
        print(f"  使用已有轨迹作为参考，长度: {len(demo_path)}")
    else:
        print("  生成弧形参考路径...")
        # 生成中间点
        p_mid = (p_start + p_goal) / 2.0
        p_mid[2] += 0.15

        # 使用三次样条插值
        t = np.linspace(0, 1, 20)
        demo_path = np.array([
            np.interp(t, [0, 0.5, 1], [p_start[i], p_mid[i], p_goal[i]])
            for i in range(3)
        ]).T
        print(f"  生成参考路径长度: {len(demo_path)}")

    # ========== 4. 配置参数 ==========
    weights_sef = np.array([1.0, 1.0, 1.0, 1.0])
    comfort_threshold = 0.05

    print(f"\n开始复合场规划...")
    print(f"  起始位置: {p_start}")
    print(f"  目标位置: {p_goal}")
    print(f"  起始关节: {q_start}")
    print(f"  最优关节: {q_opt}")
    print(f"  距离: {np.linalg.norm(p_goal - p_start):.4f}m")

    # ========== 5. 执行规划 ==========
    trajectory, joint_traj, sef_vals, sdf_vals = composite_field_planning(
        p_start=p_start,
        p_goal=p_goal,
        q_start=q_start,
        q_opt=q_opt,
        demo_path=demo_path,
        shoulder=shoulder,
        d_uar=d_uar,
        d_lar=d_lar,
        weights_sef=weights_sef,
        comfort_threshold=comfort_threshold,
        w_sdf=1.0,
        w_sef=0.8,
        w_goal=1.2,
        step_size=0.008,
        max_iters=300,
        tol_goal=0.04
    )

    # ========== 6. 验证第一步 ==========
    print(f"\n第一步验证:")
    print(f"  q[0]: {joint_traj[0]}")
    print(f"  q[1]: {joint_traj[1]}")
    print(f"  关节变化: {np.linalg.norm(joint_traj[1] - joint_traj[0]):.6f} rad")
    print(f"  p[0]: {trajectory[0]}")
    print(f"  p[1]: {trajectory[1]}")
    print(f"  位置变化: {np.linalg.norm(trajectory[1] - trajectory[0]):.6f}m")

    # ========== 7. 更新全局变量 ==========
    trajectory_hand = trajectory.tolist()
    joint_history = joint_traj.tolist()
    score_history = sef_vals.tolist()

    current_q = joint_traj[-1]
    global_positions[5] = trajectory[-1]

    # 更新肘部轨迹
    trajectory_elbow_new = []
    for i in range(len(trajectory)):
        elbow, _ = mos.forward_kinematics(joint_traj[i], d_uar, d_lar)
        elbow = trans_shoulder2global(elbow, shoulder, arm='right')
        trajectory_elbow_new.append(elbow)

    trajectory_elbow = trajectory_elbow_new

    print(f"\n复合场规划完成，生成 {len(trajectory)} 个轨迹点")

    # ========== 8. 可视化 ==========
    visualize_planning_results(trajectory, joint_traj, sef_vals, sdf_vals, demo_path,
                               p_start, q_start)
    visualize_trajectory_details()

    return trajectory_hand, trajectory_elbow, joint_history, score_history


def visualize_planning_results(trajectory, joint_traj, sef_vals, sdf_vals, demo_path,
                               p_start_verified, q_start_verified):
    """可视化规划结果（修正版：确保初始状态显示正确）"""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(18, 12))

    # ============ 3D骨架和轨迹可视化 ============
    ax_3d = fig.add_subplot(2, 3, 1, projection='3d')

    # 绘制skeleton模型
    utils.plot_skeleton(ax_3d, global_positions, skeleton_parent_indices, color='gray')

    # 绘制参考路径
    demo_path_array = np.array(demo_path)
    ax_3d.plot(demo_path_array[:, 0], demo_path_array[:, 1], demo_path_array[:, 2],
               'g--', linewidth=2, label='Reference Path', alpha=0.6, zorder=2)

    shoulder_pos = global_positions[3]

    # ========== 正确计算初始手臂姿态 ==========
    initial_elbow_shoulder, initial_wrist_shoulder = mos.forward_kinematics(q_start_verified, d_uar, d_lar)
    initial_elbow = trans_shoulder2global(initial_elbow_shoulder, shoulder_pos, arm='right')
    initial_wrist = trans_shoulder2global(initial_wrist_shoulder, shoulder_pos, arm='right')

    # 验证初始wrist和trajectory起点是否重合
    wrist_diff = np.linalg.norm(initial_wrist - trajectory[0])
    print(f"\n可视化验证:")
    print(f"  初始wrist (从q_start FK): {initial_wrist}")
    print(f"  trajectory[0]: {trajectory[0]}")
    print(f"  差异: {wrist_diff:.6f}m")

    if wrist_diff > 0.001:
        print(f"  警告: 初始wrist和轨迹起点不重合！")

    # 绘制肩部
    ax_3d.scatter(shoulder_pos[0], shoulder_pos[1], shoulder_pos[2],
                  c='black', s=100, marker='o', label='Shoulder', zorder=5)

    # 绘制初始手臂姿态（使用FK计算的正确位置）
    ax_3d.scatter(initial_elbow[0], initial_elbow[1], initial_elbow[2],
                  c='orange', s=80, marker='o', label='Initial Elbow', alpha=0.7, zorder=5)
    ax_3d.scatter(initial_wrist[0], initial_wrist[1], initial_wrist[2],
                  c='red', s=80, marker='o', label='Initial Wrist', alpha=0.7, zorder=5)

    ax_3d.plot([shoulder_pos[0], initial_elbow[0], initial_wrist[0]],
               [shoulder_pos[1], initial_elbow[1], initial_wrist[1]],
               [shoulder_pos[2], initial_elbow[2], initial_wrist[2]],
               c='red', linewidth=2, alpha=0.5, label='Initial Arm', zorder=3)

    # 绘制规划的手腕轨迹
    ax_3d.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2],
               'b-', linewidth=3, label='Planned Trajectory', zorder=4)

    # 标记起点和终点（应该和initial_wrist重合）
    ax_3d.scatter(trajectory[0, 0], trajectory[0, 1], trajectory[0, 2],
                  c='green', s=150, marker='*', label='Start', zorder=6,
                  edgecolors='black', linewidths=2)
    ax_3d.scatter(trajectory[-1, 0], trajectory[-1, 1], trajectory[-1, 2],
                  c='red', s=150, marker='*', label='Goal', zorder=6,
                  edgecolors='black', linewidths=2)

    # 绘制最终手臂姿态
    final_elbow, final_wrist = mos.forward_kinematics(joint_traj[-1], d_uar, d_lar)
    final_elbow = trans_shoulder2global(final_elbow, shoulder_pos, arm='right')
    final_wrist = trans_shoulder2global(final_wrist, shoulder_pos, arm='right')

    ax_3d.scatter(final_elbow[0], final_elbow[1], final_elbow[2],
                  c='blue', s=80, marker='o', label='Final Elbow', zorder=5)
    ax_3d.scatter(final_wrist[0], final_wrist[1], final_wrist[2],
                  c='blue', s=80, marker='o', label='Final Wrist', zorder=5)

    ax_3d.plot([shoulder_pos[0], final_elbow[0], final_wrist[0]],
               [shoulder_pos[1], final_elbow[1], final_wrist[1]],
               [shoulder_pos[2], final_elbow[2], final_wrist[2]],
               c='blue', linewidth=3, label='Final Arm', zorder=4)

    # 绘制最优位置
    ax_3d.scatter(optimal_position[0], optimal_position[1], optimal_position[2],
                  c='magenta', s=100, marker='D', label='Optimal Position',
                  zorder=5, edgecolors='black', linewidths=2)

    ax_3d.set_xlabel('X (m)', fontsize=10)
    ax_3d.set_ylabel('Y (m)', fontsize=10)
    ax_3d.set_zlabel('Z (m)', fontsize=10)
    ax_3d.set_title('3D Skeleton with Trajectories', fontsize=12, fontweight='bold')
    ax_3d.legend(loc='upper left', fontsize=7, framealpha=0.9, ncol=2)
    ax_3d.grid(True, alpha=0.3)

    # ============ 其余图表保持不变 ============
    # ... [俯视图、侧视图等保持不变]

    # ============ 俯视图 (XY) ============
    ax_xy = fig.add_subplot(2, 3, 2)
    ax_xy.plot(demo_path_array[:, 0], demo_path_array[:, 1], 'g--', linewidth=2, label='Reference', alpha=0.6)
    ax_xy.plot(trajectory[:, 0], trajectory[:, 1], 'b-', linewidth=2, label='Planned')
    ax_xy.scatter(trajectory[0, 0], trajectory[0, 1], c='green', s=80, marker='*', zorder=5, label='Start')
    ax_xy.scatter(trajectory[-1, 0], trajectory[-1, 1], c='red', s=80, marker='*', zorder=5, label='Goal')
    ax_xy.scatter(shoulder_pos[0], shoulder_pos[1], c='black', s=60, marker='o', label='Shoulder')
    ax_xy.set_xlabel('X (m)')
    ax_xy.set_ylabel('Y (m)')
    ax_xy.set_title('Top View (XY)')
    ax_xy.grid(True, alpha=0.3)
    ax_xy.legend()
    ax_xy.axis('equal')

    # ============ 侧视图 (XZ) ============
    ax_xz = fig.add_subplot(2, 3, 3)
    ax_xz.plot(demo_path_array[:, 0], demo_path_array[:, 2], 'g--', linewidth=2, label='Reference', alpha=0.6)
    ax_xz.plot(trajectory[:, 0], trajectory[:, 2], 'b-', linewidth=2, label='Planned')
    ax_xz.scatter(trajectory[0, 0], trajectory[0, 2], c='green', s=80, marker='*', zorder=5, label='Start')
    ax_xz.scatter(trajectory[-1, 0], trajectory[-1, 2], c='red', s=80, marker='*', zorder=5, label='Goal')
    ax_xz.scatter(shoulder_pos[0], shoulder_pos[2], c='black', s=60, marker='o', label='Shoulder')
    ax_xz.set_xlabel('X (m)')
    ax_xz.set_ylabel('Z (m)')
    ax_xz.set_title('Side View (XZ)')
    ax_xz.grid(True, alpha=0.3)
    ax_xz.legend()
    ax_xz.axis('equal')

    # ============ SEF值历史 ============
    ax_sef = fig.add_subplot(2, 3, 4)
    ax_sef.plot(sef_vals, 'b-', linewidth=2, label='TSEF Value')
    ax_sef.axhline(y=0, color='r', linestyle='--', linewidth=2, label='Comfort Threshold')
    ax_sef.fill_between(range(len(sef_vals)), sef_vals, 0,
                        where=(np.array(sef_vals) < 0), alpha=0.3, color='green', label='Comfort Zone')
    ax_sef.fill_between(range(len(sef_vals)), sef_vals, 0,
                        where=(np.array(sef_vals) >= 0), alpha=0.3, color='red', label='Discomfort Zone')
    ax_sef.set_xlabel('Iteration')
    ax_sef.set_ylabel('TSEF Value')
    ax_sef.set_title('Ergonomic Score (TSEF)')
    ax_sef.grid(True, alpha=0.3)
    ax_sef.legend(fontsize=8)

    textstr = f'Initial: {sef_vals[0]:.4f}\nFinal: {sef_vals[-1]:.4f}\nMin: {np.min(sef_vals):.4f}'
    ax_sef.text(0.98, 0.98, textstr, transform=ax_sef.transAxes, fontsize=9,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # ============ SDF值历史 ============
    ax_sdf = fig.add_subplot(2, 3, 5)
    ax_sdf.plot(sdf_vals, 'r-', linewidth=2, label='Distance to Reference')
    ax_sdf.set_xlabel('Iteration')
    ax_sdf.set_ylabel('SDF Value (m)')
    ax_sdf.set_title('Distance to Reference Path (CSDF)')
    ax_sdf.grid(True, alpha=0.3)
    ax_sdf.legend(fontsize=8)

    textstr = f'Mean: {np.mean(sdf_vals):.4f}m\nMax: {np.max(sdf_vals):.4f}m\nFinal: {sdf_vals[-1]:.4f}m'
    ax_sdf.text(0.98, 0.98, textstr, transform=ax_sdf.transAxes, fontsize=9,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    # ============ 关节角度历史 ============
    ax_joint = fig.add_subplot(2, 3, 6)
    joint_names = ['Joint 1', 'Joint 2', 'Joint 3', 'Joint 4']
    colors = ['red', 'green', 'blue', 'orange']

    for i in range(joint_traj.shape[1]):
        ax_joint.plot(joint_traj[:, i], label=joint_names[i],
                      linewidth=2, color=colors[i])
        ax_joint.axhline(y=optimal_q[i], color=colors[i],
                         linestyle=':', alpha=0.5, linewidth=1.5, label=f'Optimal {joint_names[i]}')

    ax_joint.set_xlabel('Iteration')
    ax_joint.set_ylabel('Joint Angle (rad)')
    ax_joint.set_title('Joint Angle Evolution')
    ax_joint.grid(True, alpha=0.3)
    ax_joint.legend(fontsize=7, loc='best', ncol=2)

    plt.tight_layout()

    output_dir = '/home/clover/Chenzui/Ergo-Manip/data'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_path = os.path.join(output_dir, 'composite_field_planning_corrected.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n图形已保存到: {output_path}")
    plt.show()


def visualize_trajectory_details():
    """额外的详细轨迹可视化"""
    global trajectory_hand, trajectory_elbow, global_positions, shoulder

    fig = plt.figure(figsize=(18, 6))

    traj = np.array(trajectory_hand)

    # ============ XY平面投影 ============
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.plot(traj[:, 0], traj[:, 1], 'b-', linewidth=2, label='Wrist Path')
    ax1.scatter(traj[0, 0], traj[0, 1], c='green', s=100, marker='*',
                label='Start', zorder=5, edgecolors='black', linewidths=2)
    ax1.scatter(traj[-1, 0], traj[-1, 1], c='red', s=100, marker='*',
                label='Goal', zorder=5, edgecolors='black', linewidths=2)
    ax1.scatter(optimal_position[0], optimal_position[1], c='magenta',
                s=80, marker='D', label='Optimal', zorder=5)
    ax1.scatter(shoulder[0], shoulder[1], c='black', s=80, marker='o',
                label='Shoulder', zorder=5)
    ax1.set_xlabel('X (m)', fontsize=10)
    ax1.set_ylabel('Y (m)', fontsize=10)
    ax1.set_title('Top View (XY Plane)', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8)
    ax1.axis('equal')

    # ============ XZ平面投影 ============
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.plot(traj[:, 0], traj[:, 2], 'b-', linewidth=2, label='Wrist Path')
    ax2.scatter(traj[0, 0], traj[0, 2], c='green', s=100, marker='*',
                label='Start', zorder=5, edgecolors='black', linewidths=2)
    ax2.scatter(traj[-1, 0], traj[-1, 2], c='red', s=100, marker='*',
                label='Goal', zorder=5, edgecolors='black', linewidths=2)
    ax2.scatter(optimal_position[0], optimal_position[2], c='magenta',
                s=80, marker='D', label='Optimal', zorder=5)
    ax2.scatter(shoulder[0], shoulder[2], c='black', s=80, marker='o',
                label='Shoulder', zorder=5)
    ax2.set_xlabel('X (m)', fontsize=10)
    ax2.set_ylabel('Z (m)', fontsize=10)
    ax2.set_title('Side View (XZ Plane)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8)
    ax2.axis('equal')

    # ============ YZ平面投影 ============
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.plot(traj[:, 1], traj[:, 2], 'b-', linewidth=2, label='Wrist Path')
    ax3.scatter(traj[0, 1], traj[0, 2], c='green', s=100, marker='*',
                label='Start', zorder=5, edgecolors='black', linewidths=2)
    ax3.scatter(traj[-1, 1], traj[-1, 2], c='red', s=100, marker='*',
                label='Goal', zorder=5, edgecolors='black', linewidths=2)
    ax3.scatter(optimal_position[1], optimal_position[2], c='magenta',
                s=80, marker='D', label='Optimal', zorder=5)
    ax3.scatter(shoulder[1], shoulder[2], c='black', s=80, marker='o',
                label='Shoulder', zorder=5)
    ax3.set_xlabel('Y (m)', fontsize=10)
    ax3.set_ylabel('Z (m)', fontsize=10)
    ax3.set_title('Front View (YZ Plane)', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=8)
    ax3.axis('equal')

    plt.tight_layout()

    output_dir = '/home/clover/Chenzui/Ergo-Manip/data'
    output_path = os.path.join(output_dir, 'trajectory_projections.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"轨迹投影图已保存到: {output_path}")
    plt.show()


if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)

    print("使用直接赋值的数据进行测试...")

    # 直接赋值机器人和人体位置数据
    sub_robot = np.array([-0.2195, 1.11462, 0, 0, 0, 0, 1])
    sub_shouR = np.array([2, 1.5, -0.25, 0, 0, 0, 1])
    sub_elbowR = np.array([1.9, 1.3, -0.3, 0, 0, 0, 1])
    sub_wristR = np.array([1.8, 1.4, -0.3, 0, 0, 0, 1])

    T_optitrack2robotbase = np.eye(4)

    shouR_position_init = sub_shouR[:3]
    elbowR_position_init = sub_elbowR[:3]
    wristR_position_init = sub_wristR[:3]

    # 关节角度范围和最优配置
    joint_angle_bounds = [
        (-math.pi / 18, 17 * math.pi / 18),
        (-math.pi / 18, 17 * math.pi / 18),
        (-np.pi / 3, np.pi / 2),
        (-np.pi / 2, np.pi / 3)
    ]
    optimal_q = np.array([0, 0, 0, -math.pi / 4])

    # 读取骨架数据
    skeleton_joint_name, skeleton_joints, skeleton_parent_indices, skeleton_joint_local_translation = \
        utils.read_skeleton_motion('/home/clover/Chenzui/Ergo-Manip/data/demo_2_test_chenzui_only_optitrack2hotu.npy')
    skeleton_joint = skeleton_joints[500, :]
    global_positions, global_rotations = utils.forward_kinematics(skeleton_joint_local_translation,
                                                                  skeleton_joint, skeleton_parent_indices)
    global_positions[:, 2] = global_positions[:, 2] * 1.2

    # 更新手臂位置
    global_positions[4] = global_positions[3] + (elbowR_position_init - shouR_position_init)
    global_positions[5] = global_positions[3] + (wristR_position_init - shouR_position_init)

    shou_center = shouR_position_init
    global_positions = global_positions + np.array([shou_center[0], shou_center[1], 0])

    initial_position = global_positions[5]

    # 计算手臂尺寸
    d_ual, d_uar, d_lal, d_lar = mos.calculate_arm_dimensions(
        shouR_position_init, elbowR_position_init, wristR_position_init,
        shouR_position_init, elbowR_position_init, wristR_position_init
    )

    # 计算最优位置
    _, optimal_position = mos.forward_kinematics(optimal_q, d_uar, d_lar)
    optimal_position = trans_shoulder2global(optimal_position, global_positions[3], arm='right')

    # 初始化关节角度
    p_elbowR_init, p_wristR_init = trans_global2shoulder(
        global_positions[3], global_positions[4], global_positions[5], arm='right'
    )

    current_q = mos.inverse_kinematics(p_elbowR_init, p_wristR_init, d_uar, d_lar)
    current_score = utils.calculate_upper_limb_score_with_joint_angles(current_q)

    hand_current = global_positions[5]
    elbow_current = global_positions[4]
    shoulder = global_positions[3].copy()

    # 初始化轨迹记录
    trajectory_hand = [hand_current.copy()]
    trajectory_elbow = [elbow_current.copy()]
    score_history = []
    joint_history = []

    print("\n=== 开始复合场规划（改进版）===")
    print(f"初始位置: {hand_current}")
    print(f"目标位置: {optimal_position}")
    print(f"初始关节角度: {current_q}")
    print(f"最优关节角度: {optimal_q}")

    try:
        # 运行复合场规划
        trajectory_hand, trajectory_elbow, joint_history, score_history = run_composite_field_planning(
            num_iterations=150)

        print(f"\n=== 规划完成 ===")
        print(f"生成轨迹点数: {len(trajectory_hand)}")
        print(f"初始SEF值: {score_history[0]:.4f}")
        print(f"最终SEF值: {score_history[-1]:.4f}")
        print(f"SEF改善: {score_history[0] - score_history[-1]:.4f}")

    except Exception as e:
        print(f"\n规划过程中出现错误: {e}")
        import traceback

        traceback.print_exc()

    print("\n程序运行完成!")