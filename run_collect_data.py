#!/usr/bin/env python3
"""
采集 OptiTrack 的 wristR 平滑位置数据并保存到本地。
默认运行 60 秒。通过环境变量可配置时长与保存目录。
"""
import numpy as np
import os
import sys
import time
import signal
import subprocess

import rospy
from geometry_msgs.msg import PoseStamped

import transformation as tsf

workspace_path = '/home/clover/catkin_ws'
sys.path.append(os.path.join(workspace_path, 'devel', 'lib'))

latest_robot_msg = None
latest_wristR_msg = None


def vrpn_launch_roslaunch():
    launch_file = "~/catkin_ws/src/vrpn_client_ros/launch/sample.launch"
    command = f"roslaunch {launch_file} server:=192.168.10.7"
    return subprocess.Popen(command, shell=True)


def signal_handler(sig, frame):
    print('Python shutdown signal received...')
    rospy.signal_shutdown("shutdown by manual")
    if 'vrpn_roslaunch_process' in globals():
        try:
            vrpn_roslaunch_process.terminate()
            vrpn_roslaunch_process.wait()
        except Exception:
            pass
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


def robot_callback(msg):
    global latest_robot_msg
    latest_robot_msg = msg


def wristR_callback(msg):
    global latest_wristR_msg
    latest_wristR_msg = msg


def setup_subscribers():
    rospy.Subscriber('/vrpn_client_node/robot/pose', PoseStamped, robot_callback)
    rospy.Subscriber('/vrpn_client_node/wristR/pose', PoseStamped, wristR_callback)


def collect_wrist_only(duration_sec=60, save_dir=None, control_dt=0.01, optitrack_smooth_alpha=0.8):
    """
    采集 OptiTrack 的 wristR 平滑位置数据并保存到本地。
    duration_sec: 采集时长（秒），默认 60
    save_dir: 保存目录，默认为 data/optitrack_collect
    control_dt: 采样周期（秒），默认 0.01（约 100Hz）
    optitrack_smooth_alpha: EMA 平滑系数
    """
    global latest_robot_msg, latest_wristR_msg
    setup_subscribers()
    print("Waiting for OptiTrack robot and wristR poses...")
    start_wait = time.time()
    while (latest_robot_msg is None or latest_wristR_msg is None) and (time.time() - start_wait) < 10.0:
        time.sleep(0.05)
    if latest_robot_msg is None or latest_wristR_msg is None:
        rospy.logerr("Timeout: robot or wristR pose not received. Exit.")
        return
    wristR_position_smooth = None
    recorded_timestamps = []
    recorded_wrist_smooth = []
    t_start = time.time()
    print("Collecting wristR_position_smooth for {:.0f} s (sampling every {:.3f} s)...".format(duration_sec, control_dt))
    while (time.time() - t_start) < duration_sec:
        if latest_robot_msg is None or latest_wristR_msg is None:
            time.sleep(0.01)
            continue
        sub_robot = transform_to_pose(latest_robot_msg)
        sub_wristR = transform_to_pose(latest_wristR_msg)
        T_optitrack2robotbase = np.linalg.inv(
            tsf.transform_optitrack_origin_to_optitrack_robot(
                sub_robot) @ tsf.transform_optitrack_robot_to_robot_base())
        wristR_position = T_optitrack2robotbase[:3, :3] @ sub_wristR[:3] + T_optitrack2robotbase[:3, 3]
        if wristR_position_smooth is None:
            wristR_position_smooth = wristR_position.copy()
        else:
            wristR_position_smooth = optitrack_smooth_alpha * wristR_position_smooth + (1 - optitrack_smooth_alpha) * wristR_position
        recorded_timestamps.append(time.time() - t_start)
        recorded_wrist_smooth.append(wristR_position_smooth.copy())
        time.sleep(control_dt)
    if save_dir is None:
        save_dir = '/home/clover/Chenzui/Ergo-Manip/data/composite_field/demonstration'
    os.makedirs(save_dir, exist_ok=True)
    timestamps_str = time.strftime('%Y%m%d_%H%M%S', time.localtime(t_start))
    out_path = os.path.join(save_dir, 'wristR_position_smooth_{}.npy'.format(timestamps_str))
    data = {
        'timestamps': np.array(recorded_timestamps),
        'wristR_position_smooth': np.array(recorded_wrist_smooth),
    }
    np.save(out_path, data)
    n = len(recorded_timestamps)
    duration_actual = recorded_timestamps[-1] - recorded_timestamps[0] if n > 1 else 0.0
    rate = n / duration_actual if duration_actual > 0 else 0.0
    print("Saved {} samples to {}".format(n, out_path))
    print("Duration: {:.2f} s, rate: {:.1f} Hz".format(duration_actual, rate))


if __name__ == '__main__':
    rospy.init_node('cf_hrc')
    signal.signal(signal.SIGINT, signal_handler)

    time.sleep(1)
    vrpn_roslaunch_process = vrpn_launch_roslaunch()

    print("Start Collecting...")
    time.sleep(3)

    collect_duration = int(os.environ.get('COLLECT_WRIST_DURATION', '20'))
    save_folder = os.environ.get('COLLECT_WRIST_SAVE_DIR', '/home/clover/Chenzui/Ergo-Manip/data/composite_field/demonstration')
    collect_wrist_only(duration_sec=collect_duration, save_dir=save_folder)
    print("Collect done. Exiting.")
