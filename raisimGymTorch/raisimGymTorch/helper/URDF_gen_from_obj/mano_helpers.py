import collections

import smplx
from smplx.joint_names import JOINT_NAMES, SMPLH_JOINT_NAMES
import numpy as np

'''
JOINT_LIMIT_FINGER_R = {'O': [[[-0.25000, 0.25000], 
                               [-0.8    , 0.8    ],
                               [-1.57    , 1.57   ]],
                              [[-0.25000, 0.25000],
                               [-0.25000, 0.25000],
                               [-1.57  , 1.57   ]],
                              [[-0.25000, 0.25000],
                               [-0.25000, 0.25000],
                               [-1.57  , 1.57   ]]],
                        'F': [[[-0.25000, 0.25000], 
                               [-0.8    , 0.8    ],
                               [-1.57    , 1.57   ]],
                              [[-0.25000, 0.25000],
                               [-0.25000, 0.25000],
                               [-1.57 , 1.57   ]],
                              [[-0.25000, 0.25000],
                               [-0.25000, 0.25000],
                               [-1.57  , 1.57   ]]],
                        'T':  [[[-1.57       , 1.57   ], 
                               [-0.8    , 0.8    ],
                               [-0.25   , 0.25   ]],
                              [[-1      , 1      ],
                               [-0.5    , 0.5    ],
                               [-0.5    , 0.5    ]],
                              [[-1.57  , 1.57   ],
                               [-1.000  , 1.000  ],
                               [-1.000  , 1.000  ]]],
                        'W':  [[-0.8    , 0.8    ], 
                               [-6.28   , 6.28   ]],
                       }

JOINT_LIMIT_FINGER_R = {'O': [[[-0.25000, 0.25000], 
                               [-0.8    , 0.8    ],
                               [-0.0    , 1.57   ]],
                              [[-0.25000, 0.25000],
                               [-0.25000, 0.25000],
                               [-0.000  , 1.57   ]],
                              [[-0.25000, 0.25000],
                               [-0.25000, 0.25000],
                               [-0.000  , 1.57   ]]],
                        'F': [[[-0.25000, 0.25000], 
                               [-0.8    , 0.8    ],
                               [-0.0    , 1.57   ]],
                              [[-0.25000, 0.25000],
                               [-0.25000, 0.25000],
                               [-0.5000 , 1.57   ]],
                              [[-0.25000, 0.25000],
                               [-0.25000, 0.25000],
                               [-0.000  , 1.57   ]]],
                        'T':  [[[0       , 1.57   ], 
                               [-0.8    , 0.8    ],
                               [-0.25   , 0.25   ]],
                              [[-1      , 1      ],
                               [-0.5    , 0.5    ],
                               [-0.5    , 0.5    ]],
                              [[-0.000  , 1.57   ],
                               [-1.000  , 1.000  ],
                               [-1.000  , 1.000  ]]],
                        'W':  [[-0.8    , 0.8    ], 
                               [-6.28   , 6.28   ]],
                       }                

         
JOINT_LIMIT_FINGER_L = {'O': [[[-0.25000, 0.25000], 
                               [-0.8    , 0.8    ],
                               [-1.57   , 0.00   ]],
                              [[-0.25000, 0.25000],
                               [-0.25000, 0.25000],
                               [-1.57   , 0.00   ]],
                              [[-0.25000, 0.25000],
                               [-0.25000, 0.25000],
                               [-1.57   , 0.00   ]]],
                        'F': [[[-0.25000, 0.25000], 
                               [-0.8    , 0.8    ],
                               [-1.57   , 0.00   ]],
                              [[-0.25000, 0.25000],
                               [-0.25000, 0.25000],
                               [-1.57   , 0.50   ]],
                              [[-0.25000, 0.25000],
                               [-0.25000, 0.25000],
                               [-1.57   , 0.00   ]]],
                        'T':  [[[0       , 1.57  ], 
                               [-0.8    , 0.8    ],
                               [-0.25   , 0.25   ]],
                              [[-1      , 1      ],
                               [-0.5    , 0.5    ],
                               [-0.5    , 0.5    ]],
                              [[-0.000  , 1.57   ],
                               [-1.000  , 1.000  ],
                               [-1.000  , 1.000  ]]],
                        'W':  [[-0.8    , 0.8    ], 
                               [-6.28   , 6.28   ]],
                       }

JOINT_LIMIT_FINGER_L = {'O': [[[-0.25000, 0.25000], 
                               [-0.8    , 0.8    ],
                               [-1.57   , 0.00   ]],
                              [[-0.25000, 0.25000],
                               [-0.25000, 0.25000],
                               [-1.57   , 0.00   ]],
                              [[-0.25000, 0.25000],
                               [-0.25000, 0.25000],
                               [-1.57   , 0.00   ]]],
                        'F': [[[-0.25000, 0.25000], 
                               [-0.8    , 0.8    ],
                               [-1.57   , 0.00   ]],
                              [[-0.25000, 0.25000],
                               [-0.25000, 0.25000],
                               [-1.57   , 0.50   ]],
                              [[-0.25000, 0.25000],
                               [-0.25000, 0.25000],
                               [-1.57   , 0.00   ]]],
                        'T':  [[[0       , 1.57  ], 
                               [-0.8    , 0.8    ],
                               [-0.25   , 0.25   ]],
                              [[-1      , 1      ],
                               [-1      , 0.5    ],
                               [-1      , 0.5    ]],
                              [[-0.600  , 1.57   ],
                               [-1.000  , 1.000  ],
                               [-1.000  , 1.000  ]]],
                        'W':  [[-0.8    , 0.8    ], 
                               [-6.28   , 6.28   ]],
                       }

JOINT_LIMIT_FINGER_L_HIGH = [0.7,  1.,   0.9,  0.7,  0.2,  0.6,  0.7,  0.4,  0.6,
                            0.7,  0.7,  0.6,  0.3,  0.6,  1.1,  0.2,  0.1,  0.5,
                            1.4,  1.3,  0.8,  2.,   1.1,  1.3,  0.6,  0.3,  0.9,
                            0.9,  0.8,  0.5,  0.1,  0.3,  1.3,  0.2,  0.2,  0.7,
                            1.8,  0.9,  0.9,  0.8,  0.8,  0.7,  1.3,  1.,   1.2]
JOINT_LIMIT_FINGER_L_LOW = [-0.4, -0.7, -1.7, -0.6, -0.5, -1.8, -0.4, -0.3, -1.,  
                            -0.5, -0.6, -1.9, -0.6, -0.2, -1.6, -0.5, -0.4, -0.9, 
                            -2.5, -0.9, -2.1, -1.4, -0.4, -1.4, -1.1, -0.8, -0.9, 
                            -0.9, -0.5, -2.,  -0.8, -0.4, -1.2, -0.7, -0.4, -1.1, 
                            -0.3, -0.6, -0.7, -1.6, -1.4, -1.5, -1.2, -1.3, -1.3]
'''
def get_limit(is_rhand=True):
    JOINT_LIMIT_FINGER_L_HIGH = np.loadtxt("limit_high_l.txt")
    JOINT_LIMIT_FINGER_L_LOW = np.loadtxt("limit_low_l.txt")
    JOINT_LIMIT_FINGER_R_HIGH = np.loadtxt("limit_high_r.txt")
    JOINT_LIMIT_FINGER_R_LOW = np.loadtxt("limit_low_r.txt")
    if is_rhand:
        return [JOINT_LIMIT_FINGER_R_LOW, JOINT_LIMIT_FINGER_R_HIGH]
    else:
        return [JOINT_LIMIT_FINGER_L_LOW, JOINT_LIMIT_FINGER_L_HIGH]


def is_finger_joint(name):
    return any(x in name for x in ['index', 'middle', 'pinky', 'ring', 'thumb'])


def is_finger_part_index(name):
    return 'index' in name


def is_finger_part_other(name):
    return (is_finger_joint(name) and 
           (not is_finger_part_index(name)) and 
           (not is_finger_part_thumb(name)))


def is_finger_part_thumb(name):
    return 'thumb' in name


def is_right_body_joint(name):
    return 'right' in name


def is_wrist(name):
    return 'wrist' in name


def is_hand(name):
    return is_finger_joint(name) or is_wrist(name)


def is_tip_of_finger(name):
    nr = name[-1]
    return not nr.isdigit()


def get_child_finger_joint_name(name):
    assert (is_finger_joint(name))

    nr = name[-1]
    if not nr.isdigit():
        return None  # tip of finger has no child

    nr = int(nr) + 1
    if nr >= 4:
        return name[:-1]  # tip of finger
    return name[:-1] + str(nr)


def get_mano_joint_names(is_rhand=True):
    # names changed from https://meshcapade.wiki/SMPL#skeleton-layout to official names
    joint_names_mano_r = [SMPLH_JOINT_NAMES [21]] + SMPLH_JOINT_NAMES [37:52]
    joint_names_mano_l = [SMPLH_JOINT_NAMES [20]] + SMPLH_JOINT_NAMES [22:37]

    # check if joint names follow the official naming and order
    if is_rhand:
        return joint_names_mano_r
    else:
        return joint_names_mano_l


def get_kinematic_child_joint(name):
    parents = get_kinematic_order()

    # invert parent dict
    children = dict((v, k) for k, v in parents.items())

    return children.get(name)


def get_kinematic_order_mano(is_rhand=True):
    # order according to smplx blender plugin
    parents_r = {
        'right_wrist': None,  # root

        'right_index1': 'right_wrist', 'right_index2': 'right_index1', 'right_index3': 'right_index2',
        'right_middle1': 'right_wrist', 'right_middle2': 'right_middle1', 'right_middle3': 'right_middle2',
        'right_pinky1': 'right_wrist', 'right_pinky2': 'right_pinky1', 'right_pinky3': 'right_pinky2',
        'right_ring1': 'right_wrist', 'right_ring2': 'right_ring1', 'right_ring3': 'right_ring2',
        'right_thumb1': 'right_wrist', 'right_thumb2': 'right_thumb1', 'right_thumb3': 'right_thumb2'
    }

    parents_l = {
        'left_wrist': None,  # root

        'left_index1': 'left_wrist', 'left_index2': 'left_index1', 'left_index3': 'left_index2',
        'left_middle1': 'left_wrist', 'left_middle2': 'left_middle1', 'left_middle3': 'left_middle2',
        'left_pinky1': 'left_wrist', 'left_pinky2': 'left_pinky1', 'left_pinky3': 'left_pinky2',
        'left_ring1': 'left_wrist', 'left_ring2': 'left_ring1', 'left_ring3': 'left_ring2',
        'left_thumb1': 'left_wrist', 'left_thumb2': 'left_thumb1', 'left_thumb3': 'left_thumb2',
    }

    if is_rhand:
        return parents_r
    else:
        return parents_l

def get_mano_part_number(is_rhand=True):
    # order according to smplx blender plugin
    parents_r = {
        'right_wrist': -1,  # root

        'right_index1': 0, 'right_index2': 1, 'right_index3': 2,
        'right_middle1': 3, 'right_middle2': 4, 'right_middle3': 5,
        'right_pinky1': 6, 'right_pinky2': 7, 'right_pinky3': 8,
        'right_ring1': 9, 'right_ring2': 10, 'right_ring3': 11,
        'right_thumb1': 12, 'right_thumb2': 13, 'right_thumb3': 14
    }

    parents_l = {
        'left_wrist': -1,  # root

        'left_index1': 0, 'left_index2': 1, 'left_index3': 2,
        'left_middle1': 3, 'left_middle2': 4, 'left_middle3': 5,
        'left_pinky1': 6, 'left_pinky2': 7, 'left_pinky3': 8,
        'left_ring1': 9, 'left_ring2': 10, 'left_ring3': 11,
        'left_thumb1': 12, 'left_thumb2': 13, 'left_thumb3': 14,
    }

    if is_rhand:
        return parents_r
    else:
        return parents_l

def get_mano_data(model_path, is_rhand=True, ncomps=24, v_template=None):
    sbj_m = smplx.create(model_path=model_path,
                         model_type='mano',
                         is_rhand=is_rhand,
                         num_pca_comps=ncomps,
                         v_template=v_template,
                         flat_hand_mean=True,
                         batch_size=1)

    sbj_out = sbj_m(**{'return_full_pose': True})

    lbs_weight_matrix = sbj_m.lbs_weights.detach().cpu().numpy()
    joints = sbj_out['joints'].detach().cpu().numpy().squeeze() # (21,3)
    verts = sbj_out['vertices'].detach().cpu().numpy().squeeze()
    if is_rhand:
        joints_dict = dict(zip([SMPLH_JOINT_NAMES [21]] + SMPLH_JOINT_NAMES [37:52] + SMPLH_JOINT_NAMES [68:73], joints))
    else:
        joints_dict = dict(zip([SMPLH_JOINT_NAMES [20]] + SMPLH_JOINT_NAMES [22:37] + SMPLH_JOINT_NAMES [63:68], joints))

    return lbs_weight_matrix, verts, joints_dict


def to_joint_dict(joints, is_rhand=True):
    return dict(zip(get_mano_joint_name(is_rhand), joints))
