from __future__ import print_function

import socket
import sys
import numpy as np

try:
    import rospy
    import tf2_ros
    import tf2_geometry_msgs  # import this is mandatory to use PoseStamped msg

    import moveit_commander

    import geometry_msgs.msg as geo_msg
    import sensor_msgs.msg as sensor_msg
except ImportError:
    rospy = None
    geo_msg = None
    sensor_msg = None

# import rospkg
# sys.path.append(rospkg.RosPack().get_path('utility'))
from utility import common


class Header:
    def __init__(self, header):
        assert isinstance(header, list) and len(header) == 10
        self.ID_str = header[0]
        self.sample_counter = header[1]
        self.datagram_counter = header[2]
        self.item_counter = header[3]  # Amount of items (point/segments) body+prop+finger
        self.time_code = header[4]  # Time since start measurement
        self.character_ID = header[5]  # Amount of people recorded at the same time
        self.body_segments_num = header[6]  # number of body segments measured
        self.props_num = header[7]  # Amount of property sensors
        self.finger_segments_num = header[8]  # Number of finger data segments
        self.payload_size = header[9]  # Size of the measurement excluding the header

    def __repr__(self):
        s = (
            "Header {}: \nsample_counter {}, datagram_counter {},\n"
            "item #{}, body segment #{}, prop #{}, finger segment #{}\n".format(
                self.ID_str,
                self.sample_counter,
                self.datagram_counter,
                self.item_counter,
                self.body_segments_num,
                self.props_num,
                self.finger_segments_num,
            )
        )
        return s

    @property
    def is_valid(self):
        if self.ID_str != "MXTP02":
            rospy.logwarn(
                "XSensInterface: Currently only support MXTP02, but got {}".format(
                    self.ID_str
                )
            )
            return False
        if (
                self.item_counter
                != self.body_segments_num + self.props_num + self.finger_segments_num
        ):
            rospy.logwarn(
                "XSensInterface: Segments number in total does not match item counter"
            )
            return False
        if self.payload_size % self.item_counter != 0:
            rospy.logwarn(
                "XSensInterface: Payload size {} is not dividable by item number {}".format(
                    self.payload_size, self.item_num
                )
            )
            return False
        return True


class Datagram(object):
    def __init__(self, header, payload):
        self.header = header
        self.payload = payload

    @property
    def main_body_segment_index(self):
        return list(range(self.header.body_segments_num))

    @property
    def prop_segment_index(self):
        return list(range(self.header.body_segments_num, self.header.body_segments_num + self.header.props_num))

    @property
    def finger_segment_index(self):
        if self.header.finger_segments_num:
            return list(range(self.header.body_segments_num + self.header.props_num,
                              self.header.body_segments_num + self.header.props_num + self.header.finger_segments_num))
        else:
            return []

    @property
    def left_finger_segment_index(self):
        if self.header.finger_segments_num:
            return list(range(self.header.body_segments_num + self.header.props_num,
                              self.header.body_segments_num + self.header.props_num + 20))
        else:
            return []

    @property
    def right_finger_segment_index(self):
        if self.header.finger_segments_num:
            return list(range(self.header.body_segments_num + self.header.props_num + 20,
                              self.header.body_segments_num + self.header.props_num + self.header.finger_segments_num))
        else:
            return []

    @property
    def item_num(self):
        return self.header.item_counter

    @property
    def item_size(self):
        """Define how many bytes in a item"""
        return self.header.payload_size // self.item_num


class XsensInterface(object):
    def __init__(
            self,
            udp_ip,
            udp_port,
            scaling=1.0,
            buffer_size=4096,
            **kwargs  # DO NOT REMOVE
    ):
        super(XsensInterface, self).__init__()

        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # UDP
        self._sock.bind((udp_ip, udp_port))
        self._buffer_size = buffer_size

        self._body_frames = {
            "pelvis": 0,
            "l5": 1,
            "l3": 2,
            "t12": 3,
            "t8": 4,
            "neck": 5,
            "head": 6,
            "right_shoulder": 7,
            "right_upper_arm": 8,
            "right_forearm": 9,
            "right_hand": 10,
            "left_shoulder": 11,
            "left_upper_arm": 12,
            "left_forearm": 13,
            "left_hand": 14,
            "right_upper_leg": 15,
            "right_lower_leg": 16,
            "right_foot": 17,
            "right_toe": 18,
            "left_upper_leg": 19,
            "left_lower_leg": 20,
            "left_foot": 21,
            "left_toe": 22,
        }

        self.scaling_factor = scaling
        self.header = None
        self.object_poses = None
        self.all_segments_poses = None

    def get_datagram(self):
        """[Main entrance function] Get poses from the datagram."""
        data, _ = self._sock.recvfrom(self._buffer_size)
        self.datagram = self._get_datagram(data)
        return self.datagram

    def get_transform_to_ros_pose(self, base_pose, target_pose):

        relative_pose = common.get_transform_same_base(base_pose, target_pose)
        msg = common.to_ros_pose(relative_pose)
        return msg

    @staticmethod
    def _get_header(data):
        """Get the header data from the received MVN Awinda datagram.

        :param data: Tuple From self._sock.recvfrom(self._buffer_size)
        :return: Header
        """
        if len(data) < 24:
            rospy.logwarn(
                "XSensInterface: Data length {} is less than 24".format(len(data))
            )
            return None
        id_str = common.byte_to_str(data[0:6], 6)
        sample_counter = common.byte_to_uint32(data[6:10])
        datagram_counter = data[10]
        item_number = common.byte_to_uint8(data[11])
        time_code = common.byte_to_uint32(data[12:16])
        character_id = common.byte_to_uint8(data[16])
        body_segments_num = common.byte_to_uint8(data[17])
        props_num = common.byte_to_uint8(data[18])
        finger_segments_num = common.byte_to_uint8(data[19])
        # 20 21 are reserved for future use
        payload_size = common.byte_to_uint16(data[22:24])
        header = Header(
            [
                id_str,
                sample_counter,
                datagram_counter,
                item_number,
                time_code,
                character_id,
                body_segments_num,
                props_num,
                finger_segments_num,
                payload_size,
            ]
        )
        rospy.logdebug(header.__repr__())
        return header

    def _get_datagram(self, data):
        header = self._get_header(data)
        if header is not None and header.is_valid:
            self.header = header
            return Datagram(header, data[24:])
        else:
            return None

    @staticmethod
    def decode_to_pose(item):
        """Decode a type 02 stream tself.header.iteo ROS pose message.

        :param item: str String of bytes
        """
        if len(item) != 32:
            rospy.logerr(
                "XSensInterface: Payload pose data size is not 32: {}".format(len(item))
            )
            return None
        # segment_id = common.byte_to_uint32(item[:4])
        x = common.byte_to_float(item[4:8])
        y = common.byte_to_float(item[8:12])
        z = common.byte_to_float(item[12:16])
        qw = common.byte_to_float(item[16:20])
        qx = common.byte_to_float(item[20:24])
        qy = common.byte_to_float(item[24:28])
        qz = common.byte_to_float(item[28:32])
        pose = np.array([x, y, z, qx, qy, qz, qw])
        # We do not need to convert the pose from MVN frame (x forward, y up, z right) to ROS frame,
        # since the type 02 data is Z-up, see:
        # https://www.xsens.com/hubfs/Downloads/Manuals/MVN_real-time_network_streaming_protocol_specification.pdf
        return pose
