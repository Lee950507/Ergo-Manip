#!/usr/bin/env python3
import os
import sys
import time
import signal
import subprocess
import numpy as np
import rospy

# ROS workspace binary module path
workspace_path = "/home/clover/catkin_ws"
sys.path.append(os.path.join(workspace_path, "devel", "lib"))

from libpython_curi_dual_arm_ic import Python_CURI_Control

roslaunch_process = None
vrpn_roslaunch_process = None


def launch_roslaunch():
    launch_file = "~/catkin_ws/src/curi_whole_body_interface/launch/python_curi_dual_arm_ic_qbhand.launch"
    return subprocess.Popen(f"roslaunch {launch_file}", shell=True)


def vrpn_launch_roslaunch():
    launch_file = "~/catkin_ws/src/vrpn_client_ros/launch/sample.launch"
    return subprocess.Popen(f"roslaunch {launch_file} server:=192.168.10.7", shell=True)


def signal_handler(sig, frame):
    del sig, frame
    print("Python shutdown signal received...")
    rospy.signal_shutdown("shutdown by manual")
    if roslaunch_process is not None:
        roslaunch_process.terminate()
    if vrpn_roslaunch_process is not None:
        vrpn_roslaunch_process.terminate()
    sys.exit(0)


def load_reference_trajectory():
    """
    从文件加载参考轨迹，支持:
    - .npy: Nx3 array
    - .npz: 包含 key 'reference_trajectory' 或 'trajectory' 的 Nx3 array
    """
    traj_path = os.environ.get(
        "REFERENCE_TRAJ_PATH",
        "/home/clover/Chenzui/Ergo-Manip/data/composite_field/reference_trajectory.npy",
    )
    if not os.path.exists(traj_path):
        raise FileNotFoundError(f"Reference trajectory file not found: {traj_path}")

    data = np.load(traj_path, allow_pickle=True)
    if isinstance(data, np.ndarray):
        traj = data
    else:
        if "reference_trajectory" in data:
            traj = data["reference_trajectory"]
        elif "trajectory" in data:
            traj = data["trajectory"]
        else:
            raise KeyError(f"No 'reference_trajectory' or 'trajectory' in {traj_path}")

    traj = np.asarray(traj, dtype=float)
    if traj.ndim != 2 or traj.shape[1] != 3 or len(traj) < 2:
        raise ValueError("Reference trajectory must be Nx3 with N>=2.")
    return traj


def generate_vic_carrying_reference_trajectory(fs_hz=1000.0, left_y_sign=1.0):
    """
    在机器人基座/躯干坐标系下生成分段直线参考轨迹（位置，单位 m），采样率 fs_hz。
    约定：+Z 向上；沿 Y 「左/右」由 left_y_sign 决定（默认 +Y 为「左」）。

    段 1：沿 +Z 上升 0.15 m，历时 3 s
    段 2：沿 Y 向左 0.10 m，历时 2 s
    段 3：沿 Y 向右 0.20 m，历时 4 s
    段 4：沿 Y 向左 0.10 m，历时 2 s

    返回 shape (N, 3)，首点为 [0,0,0]，总时长 11 s，N = 11000 @ 1000 Hz。
    """
    segments = [
        (6.0, np.array([0.0, 0.0, 0.40])),   # dz=+15cm
        (3.0, np.array([0.0, 0.20 * left_y_sign, 0.0])),
        (6.0, np.array([0.0, -0.40 * left_y_sign, 0.0])),
        (3.0, np.array([0.0, 0.10 * left_y_sign, 0.0])),
    ]
    points = []
    p = np.zeros(3, dtype=float)
    for duration, delta in segments:
        n = max(1, int(round(duration * fs_hz)))
        for k in range(n):
            alpha = (k + 1) / n
            points.append(p + alpha * delta)
        p = p + delta
    traj = np.asarray(points, dtype=float)
    return traj


if __name__ == "__main__":
    rospy.init_node("dual_arm_pose_tracking_ref")
    signal.signal(signal.SIGINT, signal_handler)

    roslaunch_process = launch_roslaunch()
    time.sleep(1.0)

    curi = Python_CURI_Control(0, [])
    curi.start()

    time.sleep(1.0)
    vrpn_roslaunch_process = vrpn_launch_roslaunch()

    # curi.set_impedance(0, 2, 40, 800)
    # curi.set_impedance(1, 2, 40, 800)

    robot_left_rotation_matrix_init = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    robot_right_rotation_matrix_init = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
    base2torso_matrix = np.array([[1, 0, 0, -0.29], [0, 1, 0, 0], [0, 0, 1, -0.985], [0, 0, 0, 1]])

    robot_left_position_init = np.array([0.9, 0.190, 0.8])
    robot_right_position_init = np.array([0.9, -0.190, 0.8])

    robot_left_pose_matrix_init = np.r_[
        np.c_[robot_left_rotation_matrix_init, robot_left_position_init.T], np.array([[0, 0, 0, 1]])
    ]
    robot_right_pose_matrix_init = np.r_[
        np.c_[robot_right_rotation_matrix_init, robot_right_position_init.T], np.array([[0, 0, 0, 1]])
    ]
    curi.set_tcp_moveL(base2torso_matrix @ robot_left_pose_matrix_init, base2torso_matrix @ robot_right_pose_matrix_init)

    while curi.get_curi_mode(0) != 2 and curi.get_curi_mode(1) != 2:
        print("waiting robot external control")
        time.sleep(1.0)

    fs_hz = float(os.environ.get("REFERENCE_FS_HZ", "1000"))
    # 若你的基座定义「左」为 -Y，可设环境变量 REF_LEFT_Y_SIGN=-1
    left_y_sign = float(os.environ.get("REF_LEFT_Y_SIGN", "1"))
    use_file = os.environ.get("USE_REFERENCE_FILE", "").lower() in ("1", "true", "yes")
    if use_file:
        reference_trajectory = load_reference_trajectory()
        print(f"Loaded reference trajectory from file: {reference_trajectory.shape}")
    else:
        reference_trajectory = generate_vic_carrying_reference_trajectory(fs_hz=fs_hz, left_y_sign=left_y_sign)
        print(f"Generated reference trajectory @ {fs_hz} Hz: {reference_trajectory.shape}")

    reference_displacement = reference_trajectory - reference_trajectory[0]
    max_plan_steps = len(reference_displacement)
    control_dt = 1.0 / fs_hz

    robot_positions_left = []
    robot_positions_right = []
    recorded_timestamps = []

    folder = os.environ.get(
        "SAVE_FOLDER", "/home/clover/Chenzui/Ergo-Manip/data/vic_carrying_w_load_changing"
    )
    os.makedirs(folder, exist_ok=True)
    start_time = time.time()

    print("Start dual-arm pose tracking with reference trajectory...")
    for step in range(max_plan_steps):
        displacement_cmd = reference_displacement[step]

        robot_left_position = robot_left_position_init + displacement_cmd
        robot_right_position = robot_right_position_init + displacement_cmd

        robot_left_pose_matrix = np.r_[
            np.c_[robot_left_rotation_matrix_init, robot_left_position.T], np.array([[0, 0, 0, 1]])
        ]
        robot_right_pose_matrix = np.r_[
            np.c_[robot_right_rotation_matrix_init, robot_right_position.T], np.array([[0, 0, 0, 1]])
        ]
        curi.set_tcp_servo(base2torso_matrix @ robot_left_pose_matrix, base2torso_matrix @ robot_right_pose_matrix)


        recorded_timestamps.append(time.time() - start_time)
        robot_positions_left.append(robot_left_position.copy())
        robot_positions_right.append(robot_right_position.copy())

        if step % 5000 == 0:
            print(f"Step {step}: cmd displacement = {displacement_cmd.round(5)}")
        time.sleep(control_dt)

    np.save(os.path.join(folder, "robot_positions_left.npy"), np.array(robot_positions_left))
    np.save(os.path.join(folder, "robot_positions_right.npy"), np.array(robot_positions_right))
    np.save(os.path.join(folder, "reference_trajectory_generated.npy"), reference_trajectory)
    np.save(os.path.join(folder, "recorded_wall_time.npy"), np.array(recorded_timestamps))

    print(f"Feedforward tracking finished. Saved {len(robot_positions_left)} samples to: {folder}")

    while True:
        time.sleep(1.0)