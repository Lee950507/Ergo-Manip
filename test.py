import numpy as np
from scipy.spatial.transform import Rotation as R

def compute_updated_object_pose(initial_pose, initial_points, updated_points):
    # Extract rotation and translation from the initial pose
    rotation_initial = initial_pose[:3, :3]
    translation_initial = initial_pose[:3, 3]

    # Compute the vector for the initial and updated second points
    vector_initial = initial_points[1] - initial_points[0]
    vector_updated = updated_points[1] - updated_points[0]

    # Normalize the vectors
    vector_initial_normalized = vector_initial / np.linalg.norm(vector_initial)
    vector_updated_normalized = vector_updated / np.linalg.norm(vector_updated)

    # Calculate the rotation required to align the initial vector with the updated vector
    rotation_vector = np.cross(vector_initial_normalized, vector_updated_normalized)
    angle_of_rotation = np.arccos(np.clip(np.dot(vector_initial_normalized, vector_updated_normalized), -1.0, 1.0))

    # Create a rotation matrix using the rotation vector and angle
    if np.linalg.norm(rotation_vector) > 1e-6:  # Avoid division by zero
        rotation = R.from_rotvec(rotation_vector * angle_of_rotation)
    else:
        rotation = R.identity()

    # Update the rotation of the initial pose
    updated_rotation = rotation * R.from_matrix(rotation_initial)

    # Calculate the vectors from the initial pose to the initial points
    vector_initial_1 = translation_initial - initial_points[0]

    # Apply the rotation to the initial vectors
    rotated_vector_1 = rotation.apply(vector_initial_1)
    translation_updated = updated_points[0] + rotated_vector_1

    # Construct the updated pose
    updated_pose = np.eye(4)
    updated_pose[:3, :3] = updated_rotation.as_matrix()
    updated_pose[:3, 3] = translation_updated  # Set the updated translation

    return updated_pose, rotation


def update_additional_points(initial_pose, updated_pose, robot_point, rotation):
    initial_position = initial_pose[:3, 3]
    updated_position = updated_pose[:3, 3]
    vector_initial = robot_point - initial_position
    rotated_vector_1 = rotation.apply(vector_initial)
    updated_point = updated_position + rotated_vector_1
    return updated_point


# Example usage:
# Initial pose (4x4 transformation matrix)
initial_pose = np.array([[ 1.,   0.,   0.,   1.3],
 [ 0.,   0.,  -1.,  0. ],
 [ 0.,   1.,   0.,   1.4],
 [ 0.,   0.,   0.,   1. ]])

# Initial positions of two points on the object (2x3)
initial_points = np.array([[ 1.84565942,  0.20162924,  1.44610119],
 [ 1.82866393, -0.16297343,  1.28196371]])

# Updated positions of the two points in Cartesian space (2x3)
updated_points = np.array([[ 1.9158142,   0.28917484,  1.2548238 ],
 [ 1.89903483, -0.24794107,  1.28982309]])  # New positions of the points

# Compute the updated pose
updated_pose, rotation = compute_updated_object_pose(initial_pose, initial_points, updated_points)
print("Updated Pose:\n", updated_pose)

# Initial positions of two additional points on the object (2x3)
additional_points = np.array([-1, 1, 1])

# Update the positions of the additional points
updated_additional_points = update_additional_points(initial_pose, updated_pose, additional_points, rotation)

print("Updated Additional Points:\n", updated_additional_points)

