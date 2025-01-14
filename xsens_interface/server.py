#!/usr/bin/python3.8
from __future__ import print_function
import numpy as np
import rospy

from geometry_msgs.msg import PoseStamped, PoseArray
from datetime import datetime
from interface import XsensInterface

from utility import common
from utility import transform


class XsensServer:
    def __init__(self, kwargs):
        self.interface = XsensInterface(**kwargs)

        self.human_shoulder_to_torso = kwargs["human_shoulder_to_torso"]

        ref_frame_lowercase = kwargs["ref_frame"].lower()
        if ref_frame_lowercase in self.interface._body_frames:
            self.ref_frame = ref_frame_lowercase
            self.ref_frame_id = self.interface._body_frames[ref_frame_lowercase]
        elif ref_frame_lowercase == "" or ref_frame_lowercase == "world":
            rospy.logwarn("XSensServer: Reference frame is the world frame")
            self.ref_frame = "world"
            self.ref_frame_id = None
        else:
            rospy.logerr(
                f"XSensServer: reference frame {ref_frame_lowercase} is not supported"
            )

        # Cartesian pose publishers
        self.all_poses_publisher = rospy.Publisher(
            "/xsens/all_poses", PoseArray, queue_size=1
        )
        self.main_body_poses_publisher = rospy.Publisher(
            "/xsens/main_body_poses", PoseArray, queue_size=1
        )

        self.pub_prop = kwargs["prop"]
        # Prop pose publisher
        self.prop_publisher = rospy.Publisher(
            "/xsens/prop", PoseArray, queue_size=1
        )

        self.left_finger_poses_publisher = rospy.Publisher(
            "/xsens/left_finger_poses", PoseArray, queue_size=1
        )
        self.right_finger_poses_publisher = rospy.Publisher(
            "/xsens/right_finger_poses", PoseArray, queue_size=1
        )

        self.torso_pose_publisher = rospy.Publisher(
            "/xsens/torso", PoseStamped, queue_size=1
        )
        self.left_tcp_publisher = rospy.Publisher(
            "/xsens/left_tcp", PoseStamped, queue_size=1
        )
        self.right_tcp_publisher = rospy.Publisher(
            "/xsens/right_tcp", PoseStamped, queue_size=1
        )
        self.left_tcp2armbase_publisher = rospy.Publisher(
            "/xsens/left_tcp2armbase", PoseStamped, queue_size=1
        )
        self.right_tcp2armbase_publisher = rospy.Publisher(
            "/xsens/right_tcp2armbase", PoseStamped, queue_size=1
        )
        self.head_publisher = rospy.Publisher("/xsens/head", PoseStamped, queue_size=1)

        self.pose_publisher_list = [
            self.all_poses_publisher,
            self.main_body_poses_publisher,
            self.prop_publisher,
            self.left_finger_poses_publisher,
            self.right_finger_poses_publisher,

            self.torso_pose_publisher,
            self.head_publisher,
            self.left_tcp_publisher,
            self.right_tcp_publisher,
            self.left_tcp2armbase_publisher,
            self.right_tcp2armbase_publisher
        ]

        rate = kwargs["rate"]

        print("[Xsens] No data received")  # Overwrite the previous value
        self.all_poses_msg_timer = rospy.Timer(
            rospy.Duration.from_sec(1.0 / rate), self.callback
        )

    def callback(self, event):
        datagram = self.interface.get_datagram()
        if datagram is None:
            print("[Xsens] No data received", end='\r')  # Overwrite the previous value
            return

        print("[Xsens] Data recived on ", datetime.now().strftime("%Y-%m-%d %H:%M:%S"), end='\r')

        pose_msg_dict = self.parsing(datagram)

        for pub, msg in zip(self.pose_publisher_list, pose_msg_dict.values()):
            pub.publish(msg)

    def parsing(self, datagram):
        """Decode the bytes in the streaming data to pose array message.

        1. The body poses excepts hand segment poses
        2. Prop segment poses if existed
        2. Finger segment poses if existed
        3. Interested key poses.

        :param ref_frame: str Reference frame name of the generated pose array message.
        :param ref_frame_id: None/int If not None, all poses will be shifted subject to
                             the frame with this ID. This frame should belong to the human.
        :param scaling_factor: float Scale the position of the pose if src_frame_id is not None.
                               Its value equals to the robot/human body dimension ratio
        :return: PoseArray PoseStamped ...

        """

        ## all pose is based on world
        all_pose_list = []

        for i in range(datagram.item_num):
            item = datagram.payload[
                   i * datagram.item_size: (i + 1) * datagram.item_size
                   ]
            pose = self.interface.decode_to_pose(item)
            if pose is None:
                rospy.logerr("Some data is missing. Abort.")
            all_pose_list.append(pose)

        # all_poses should at least contain body segment poses
        assert len(all_pose_list) == len(datagram.main_body_segment_index) + len(datagram.prop_segment_index) + len(
            datagram.finger_segment_index), \
            rospy.logerr(
                "XSensInterface: all_segments_poses should at least contain body segment poses"
            )

        _stamp = rospy.Time.now()

        # all msg
        all_pose_msg = PoseArray()
        all_pose_msg.header.stamp = _stamp
        all_pose_msg.header.frame_id = self.ref_frame

        # main body
        main_body_msg = PoseArray()
        main_body_msg.header = all_pose_msg.header

        # prop
        prop_msg = PoseArray()
        prop_msg.header = main_body_msg.header

        # left finger
        left_finger_msg = PoseArray()
        left_finger_msg.header.stamp = _stamp
        left_finger_msg.header.frame_id = "left_carpus"

        # right finger
        right_finger_msg = PoseArray()
        right_finger_msg.header.stamp = _stamp
        right_finger_msg.header.frame_id = "right_carpus"

        ## interested key poses
        torso_pose_msg = PoseStamped()
        head_msg = PoseStamped()
        left_tcp_msg = PoseStamped()
        right_tcp_msg = PoseStamped()
        left_tcp2armbase_msg = PoseStamped()
        right_tcp2armbase_msg = PoseStamped()

        # Initialize message headers
        torso_pose_msg.header = main_body_msg.header
        head_msg.header = main_body_msg.header
        left_tcp_msg.header = main_body_msg.header
        right_tcp_msg.header = main_body_msg.header

        left_tcp2armbase_msg.header = main_body_msg.header
        left_tcp2armbase_msg.header.frame_id = "left_shoulder"

        right_tcp2armbase_msg.header = main_body_msg.header
        right_tcp2armbase_msg.header.frame_id = "right_shoulder"

        if self.ref_frame_id is not None and self.ref_frame_id < len(all_pose_list):
            body_reference_pose = all_pose_list[self.ref_frame_id]
        else:
            body_reference_pose = np.zeros(7)
            body_reference_pose[-1] = 1

        right_shoulder_pos = all_pose_list[self.interface._body_frames["right_shoulder"]][:3]
        left_shoulder_pos = all_pose_list[self.interface._body_frames["left_shoulder"]][:3]
        t8_orn = all_pose_list[self.interface._body_frames["t8"]][3:]
        two_shoulder_center_pose = np.zeros(7)
        two_shoulder_center_pose[:3] = (right_shoulder_pos + left_shoulder_pos) / 2
        two_shoulder_center_pose[3:] = t8_orn
        right_shoullder2center_pose = np.zeros(7)
        left_shoullder2center_pose = np.zeros(7)
        right_shoullder2center_pose[1] = -self.human_shoulder_to_torso
        left_shoullder2center_pose[1] = self.human_shoulder_to_torso
        right_shoullder2center_pose[3:] = np.array([0, 0, 0, 1])
        left_shoullder2center_pose[3:] = np.array([0, 0, 0, 1])

        # base world
        right_shoulder_reference_pose = np.zeros(7)
        right_reference_T = common.get_transform_commutative(mid2base=two_shoulder_center_pose,
                                                             target2mid=right_shoullder2center_pose)
        right_shoulder_reference_pose[:3] = transform.translation_from_matrix(right_reference_T)
        right_shoulder_reference_pose[3:] = transform.quaternion_from_matrix(right_reference_T)
        # print( right_shoulder_reference_pose[3:])

        left_shoulder_reference_pose = np.zeros(7)
        left_reference_T = common.get_transform_commutative(mid2base=two_shoulder_center_pose,
                                                            target2mid=left_shoullder2center_pose)
        left_shoulder_reference_pose[:3] = transform.translation_from_matrix(left_reference_T)
        left_shoulder_reference_pose[3:] = transform.quaternion_from_matrix(left_reference_T)

        # print(1, right_shoulder_reference_pose)
        # print(2, left_shoulder_reference_pose)
        # print(3, all_pose_list[self.interface._body_frames["pelvis"]])
        # print(4, all_pose_list[self.interface._body_frames["right_hand"]])
        # print(5, all_pose_list[self.interface._body_frames["left_hand"]])
        # print("-----------------------------------------")
        for ind in datagram.main_body_segment_index:
            pose_msg = self.interface.get_transform_to_ros_pose(base_pose=body_reference_pose,
                                                                target_pose=all_pose_list[ind])
            main_body_msg.poses.append(pose_msg)
            all_pose_msg.poses.append(pose_msg)

            if ind == 4:  # T8
                torso_pose_msg.pose = pose_msg
            if ind == 6:  # Head
                head_msg.pose = pose_msg
            if ind == 10:  # right hand
                right_tcp_msg.pose = pose_msg
                right_tcp2armbase_msg.pose = self.interface.get_transform_to_ros_pose(
                    base_pose=right_shoulder_reference_pose, target_pose=all_pose_list[ind])
                # print(right_tcp2armbase_msg.pose)
            if ind == 14:  # left hand
                left_tcp_msg.pose = pose_msg
                left_tcp2armbase_msg.pose = self.interface.get_transform_to_ros_pose(
                    base_pose=left_shoulder_reference_pose, target_pose=all_pose_list[ind])

        for ind in datagram.prop_segment_index:
            pose_msg = self.interface.get_transform_to_ros_pose(base_pose=body_reference_pose,
                                                                target_pose=all_pose_list[ind])
            prop_msg.poses.append(pose_msg)
            all_pose_msg.poses.append(pose_msg)

        if len(datagram.left_finger_segment_index):
            left_finger_reference_pose = all_pose_list[datagram.left_finger_segment_index[0]]
            for ind in datagram.left_finger_segment_index:
                pose_msg = self.interface.get_transform_to_ros_pose(base_pose=left_finger_reference_pose,
                                                                    target_pose=all_pose_list[ind])

                left_finger_msg.poses.append(pose_msg)
                all_pose_msg.poses.append(pose_msg)

        if len(datagram.right_finger_segment_index):
            right_finger_reference_pose = all_pose_list[datagram.right_finger_segment_index[0]]
            for ind in datagram.right_finger_segment_index:
                pose_msg = self.interface.get_transform_to_ros_pose(base_pose=right_finger_reference_pose,
                                                                    target_pose=all_pose_list[ind])
                right_finger_msg.poses.append(pose_msg)
                all_pose_msg.poses.append(pose_msg)

        pose_msg_dict = {
            # PoseArray
            'all_pose': all_pose_msg,
            'main_body': main_body_msg,
            'prop': prop_msg,
            'left_finger': left_finger_msg,
            'right_finger': right_finger_msg,
            # PoseStamped
            'key_segment_torso': torso_pose_msg,
            'key_segment_left': left_tcp_msg,
            'key_segment_right': right_tcp_msg,
            'key_segment_head': head_msg,
            'key_segment_left2armbase': left_tcp2armbase_msg,
            'key_segment_right2armbase': right_tcp2armbase_msg,
        }

        return pose_msg_dict


if __name__ == '__main__':
    try:
        rospy.init_node("xsens_server")
        configs = {
            "human_shoulder_to_torso": common.get_param("human_shoulder_to_torso", 0),
            "udp_ip": common.get_param("ip", ""),
            "udp_port": common.get_param("port", 9763),
            "ref_frame": common.get_param("xsens_ref_frame", "world"),
            "scaling": common.get_param("scaling", 1.0),
            "rate": common.get_param("rate", 60.0),
            "prop": common.get_param("prop", False),
        }

        assert configs["human_shoulder_to_torso"] != 0, "human shoulder to torso length shoulder be set first"

        # If udp_ip is not given, get local IP as UDP IP
        if configs["udp_ip"] == "":
            import socket

            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            configs["udp_ip"] = s.getsockname()[0]
            s.close()

        if not common.is_ip_valid(configs["udp_ip"]) or not common.is_port_valid(configs["udp_port"]):
            exit(-1)

        server = XsensServer(configs)
        rospy.loginfo("[Xsens] Xsens server ready.")
        rospy.spin()
    except rospy.ROSInterruptException as e:
        print(e)
