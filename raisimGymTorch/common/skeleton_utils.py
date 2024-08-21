import numpy as np


def keypoint_hflip(kp, img_width):
    # Flip a keypoint horizontally around the y-axis
    # kp N,2
    if len(kp.shape) == 2:
        kp[:, 0] = (img_width - 1.0) - kp[:, 0]
    elif len(kp.shape) == 3:
        kp[:, :, 0] = (img_width - 1.0) - kp[:, :, 0]
    return kp


def convert_kps(joints2d, src, dst):
    src_names = eval(f"get_{src}_joint_names")()
    dst_names = eval(f"get_{dst}_joint_names")()

    out_joints2d = np.zeros((joints2d.shape[0], len(dst_names), joints2d.shape[-1]))

    for idx, jn in enumerate(dst_names):
        if jn in src_names:
            out_joints2d[:, idx] = joints2d[:, src_names.index(jn)]

    return out_joints2d


def get_perm_idxs(src, dst):
    src_names = eval(f"get_{src}_joint_names")()
    dst_names = eval(f"get_{dst}_joint_names")()
    idxs = [src_names.index(h) for h in dst_names if h in src_names]
    return idxs


def get_smpl_joint_names():
    return [
        "hips",  # 0
        "leftUpLeg",  # 1
        "rightUpLeg",  # 2
        "spine",  # 3
        "leftLeg",  # 4
        "rightLeg",  # 5
        "spine1",  # 6
        "leftFoot",  # 7
        "rightFoot",  # 8
        "spine2",  # 9
        "leftToeBase",  # 10
        "rightToeBase",  # 11
        "neck",  # 12
        "leftShoulder",  # 13
        "rightShoulder",  # 14
        "head",  # 15
        "leftArm",  # 16
        "rightArm",  # 17
        "leftForeArm",  # 18
        "rightForeArm",  # 19
        "leftHand",  # 20
        "rightHand",  # 21
        "leftHandIndex1",  # 22
        "rightHandIndex1",  # 23
    ]


def get_smpl_skeleton():
    return np.array(
        [
            [0, 1],
            [0, 2],
            [0, 3],
            [1, 4],
            [2, 5],
            [3, 6],
            [4, 7],
            [5, 8],
            [6, 9],
            [7, 10],
            [8, 11],
            [9, 12],
            [9, 13],
            [9, 14],
            [12, 15],
            [13, 16],
            [14, 17],
            [16, 18],
            [17, 19],
            [18, 20],
            [19, 21],
            [20, 22],
            [21, 23],
        ]
    )


def get_mano_joint_names():
    return [
        "wrist",  # 0
        "index1",  # 1
        "index2",  # 2
        "index3",  # 3
        "middle1",  # 4
        "middle2",  # 5
        "middle3",  # 6
        "pinky1",  # 7
        "pinky2",  # 8
        "pinky3",  # 9
        "ring1",  # 10
        "ring2",  # 11
        "ring3",  # 12
        "thumb1",  # 13
        "thumb2",  # 14
        "thumb3",  # 15
    ]


def get_mano_skeleton():
    return np.array(
        [
            [0, 1],
            [1, 2],
            [2, 3],
            [0, 4],
            [4, 5],
            [5, 6],
            [0, 7],
            [7, 8],
            [8, 9],
            [0, 10],
            [10, 11],
            [11, 12],
            [0, 13],
            [13, 14],
            [14, 15],
        ]
    )


def get_freihand_joint_names():
    return [
        "wrist",
        "thumb1",
        "thumb2",
        "thumb3",
        "thumb_tip",
        "index1",
        "index2",
        "index3",
        "index_tip",
        "middle1",
        "middle2",
        "middle3",
        "middle_tip",
        "ring1",
        "ring2",
        "ring3",
        "ring_tip",
        "pinky1",
        "pinky2",
        "pinky3",
        "pinky_tip",
    ]


def get_freihand_skeleton():
    return np.array(
        [
            # index
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 4],
            # thumb
            [0, 5],
            [5, 6],
            [6, 7],
            [7, 8],
            # middle
            [0, 9],
            [9, 10],
            [10, 11],
            [11, 12],
            # ring
            [0, 13],
            [13, 14],
            [14, 15],
            [15, 16],
            # pinky
            [0, 17],
            [17, 18],
            [18, 19],
            [19, 20],
        ]
    )


def get_mano21_joint_names():
    return [
        "wrist",  # 0
        "index1",  # 1
        "index2",  # 2
        "index3",  # 3
        "middle1",  # 4
        "middle2",  # 5
        "middle3",  # 6
        "pinky1",  # 7
        "pinky2",  # 8
        "pinky3",  # 9
        "ring1",  # 10
        "ring2",  # 11
        "ring3",  # 12
        "thumb1",  # 13
        "thumb2",  # 14
        "thumb3",  # 15
        "thumb_tip",  # 16
        "index_tip",  # 17
        "middle_tip",  # 18
        "ring_tip",  # 19
        "pinky_tip",  # 20
    ]


def get_mano21_skeleton():
    return np.array(
        [
            # index
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 17],
            # middle
            [0, 4],
            [4, 5],
            [5, 6],
            [6, 18],
            # pinky
            [0, 7],
            [7, 8],
            [8, 9],
            [9, 20],
            # ring
            [0, 10],
            [10, 11],
            [11, 12],
            [12, 19],
            # thumb
            [0, 13],
            [13, 14],
            [14, 15],
            [15, 16],
        ]
    )
