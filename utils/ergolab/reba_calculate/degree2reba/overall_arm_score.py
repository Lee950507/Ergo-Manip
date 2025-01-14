import numpy as np

# Create a 3D NumPy array based on the provided table (upper arm, lower arm, wrist)
# The dimensions are: (upper_arm, lower_arm, wrist)
overall_score = np.array([
    # Upper Arm Score 1
    [
        [1, 2, 2],  # Lower Arm 1 (Wrist 1, 2, 3)
        [1, 2, 3]  # Lower Arm 2 (Wrist 1, 2, 3)
    ],
    # Upper Arm Score 2
    [
        [1, 2, 3],  # Lower Arm 1
        [2, 3, 4]  # Lower Arm 2
    ],
    # Upper Arm Score 3
    [
        [3, 4, 5],  # Lower Arm 1
        [4, 5, 5]  # Lower Arm 2
    ],
    # Upper Arm Score 4
    [
        [4, 5, 5],  # Lower Arm 1
        [5, 6, 7]  # Lower Arm 2
    ],
    # Upper Arm Score 5
    [
        [6, 7, 8],  # Lower Arm 1
        [7, 8, 8]  # Lower Arm 2
    ],
    # Upper Arm Score 6
    [
        [7, 8, 8],  # Lower Arm 1
        [8, 9, 9]  # Lower Arm 2
    ]
])

def calculate_overall_score(upper_arm, lower_arm, wrist):
    """
    Function to calculate the overall score based on the given upper arm, lower arm, and wrist scores.

    :param upper_arm: Upper arm score (1 to 6)
    :param lower_arm: Lower arm score (1 to 2)
    :param wrist: Wrist score (1 to 3)
    :return: Overall score from the table
    """

    try:
        # Adjust the input scores to be zero-indexed for NumPy array indexing
        return overall_score[upper_arm - 1, lower_arm - 1, wrist - 1]
    except IndexError:
        return "Invalid input scores. Ensure upper arm is 1-6, lower arm is 1-2, and wrist is 1-3."

def calculate_overall_score_continous(upper_arm, lower_arm, wrist):

    return upper_arm + lower_arm + wrist

def calculate_upper_limb_score_with_joint_angles(q):
    shoulder_abduction, shoulder_flextion, shoulder_rotation, elbow_flextion = q
    shoulder_score = 0
    shoulder_score += 1 + 2 * abs(shoulder_abduction) / np.pi
    shoulder_score += 4.5 * abs(shoulder_flextion) / np.pi
    shoulder_score += 2 * abs(shoulder_rotation) / np.pi
    elbow_score = 0
    elbow_score += 7 * abs(elbow_flextion + np.pi / 6) / np.pi

    return shoulder_score + elbow_score
