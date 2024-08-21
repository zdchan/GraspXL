import numpy as np
from raisimGymTorch.helper import rotations
import argparse
# import open3d as o3d
from scipy.spatial.transform import Rotation as R
import torch
import random
import os

def concat_dict(dict):
    ret = {}
    for key in dict.keys():
        for k in dict[key].keys():
            if k in ret.keys():
                ret[k] = np.concatenate((ret[k], dict[key][k]), axis=0)
            else:
                ret[k] = dict[key][k]
    return ret

IDX_TO_OBJ = {
    1: ['002_master_chef_can',0.414, 0, [0.051,0.139,0.0]],
    2: ['003_cracker_box', 0.453, 1, [0.06, 0.158, 0.21]],
    3: ['004_sugar_box', 0.514, 1, [0.038, 0.089, 0.175]],
    4: ['005_tomato_soup_can', 0.349, 0, [0.033, 0.101,0.0]],
    5: ['006_mustard_bottle', 0.431,2, [0.0,0.0,0.0]],
    6: ['007_tuna_fish_can', 0.171, 0, [0.0425, 0.033,0.0]],
    7: ['008_pudding_box', 0.187, 3, [0.21, 0.089, 0.035]],
    8: ['009_gelatin_box', 0.097, 3, [0.028, 0.085, 0.073]],
    9: ['010_potted_meat_can', 0.37, 3, [0.05, 0.097, 0.089]],
    10: ['011_banana', 0.066,2, [0.028, 0.085, 0.073]],
    11: ['019_pitcher_base', 0.178,2, [0.0,0.0,0.0]],
    12: ['021_bleach_cleanser', 0.302,2, [0.0,0.0,0.0]], # not sure about weight here
    13: ['024_bowl', 0.147,2, [0.0,0.0,0.0]],
    14: ['025_mug', 0.118,2, [0.0,0.0,0.0]],
    15: ['035_power_drill', 0.895,2, [0.0,0.0,0.0]],
    16: ['036_wood_block', 0.729, 3, [0.085, 0.085, 0.2]],
    17: ['037_scissors', 0.082,2, [0.0,0.0,0.0]],
    18: ['040_large_marker', 0.01, 3, [0.009,0.121,0.0]],
    19: ['051_large_clamp', 0.125,2, [0.0,0.0,0.0]],
    20: ['052_extra_large_clamp', 0.102,2, [0.0,0.0,0.0]],
    21: ['061_foam_brick', 0.028, 1, [0.05, 0.075, 0.05]],
}

def setup_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_obj_pcd(path,num_p=100):
    mesh = o3d.io.read_triangle_mesh(path, enable_post_processing=True)
    mesh.remove_duplicated_vertices()
    obj_vertices = mesh.sample_points_uniformly(number_of_points=num_p)
    obj_vertices = np.asarray(obj_vertices.points)

    return obj_vertices

def first_nonzero(arr, axis, invalid_val=-1):
    arr = torch.Tensor(arr)
    mask = arr!=0
    mask = mask.to(torch.uint8)
    return torch.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val)

def dgrasp_to_mano(param, is_right=True):
    bs = param.shape[0]
    eulers = param[:,6:].reshape(bs,-1, 3).copy()

    # # exchange ring finger and little finger's sequence
    # temp = eulers[:,6:9].copy()
    # eulers[:,6:9] = eulers[:,9:12]
    # eulers[:,9:12] = temp

    eulers = eulers.reshape(-1,3)
    # change euler angle to axis angle
    rotvec = R.from_euler('XYZ', eulers, degrees=False)
    rotvec = rotvec.as_rotvec().reshape(bs,-1)
    global_orient = R.from_euler('XYZ', param[:,3:6], degrees=False)
    global_orient = global_orient.as_rotvec()

    # translation minus an offset
    if is_right:
        offset = np.array([[0.09566994, 0.00638343, 0.0061863]])
    else:
        offset = np.array([[-0.09566994, 0.00638343, 0.0061863]])
    mano_param = np.concatenate([global_orient, rotvec, param[:,:3] - offset],axis=1)

    return mano_param

def show_pointcloud_objhand(hand, obj):
    '''
    Draw hand and obj xyz at the same time
    :param hand: [778, 3]
    :param obj: [3000, 3]
    '''


    hand_dim = hand.shape[0]
    obj_dim = obj.shape[0]
    handObj = np.vstack((hand, obj))
    c_hand, c_obj = np.array([[1, 0, 0]]), np.array([[0, 0, 1]]) # RGB
    c_hand = np.repeat(c_hand, repeats=hand_dim, axis=0) # [778,3]
    c_obj = np.repeat(c_obj, repeats=obj_dim, axis=0) # [3000,3]
    c_hanObj = np.vstack((c_hand, c_obj)) # [778+3000, 3]

    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(handObj)
    pc.colors = o3d.utility.Vector3dVector(c_hanObj)
    o3d.visualization.draw_geometries([pc])

def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cfg', help='config file', type=str, default='cfg.yaml')
    parser.add_argument('-m', '--mode', help='set mode either train or test', type=str, default='train')
    parser.add_argument('-d', '--logdir', help='set dir for storing data', type=str, default=None)
    parser.add_argument('-e', '--exp_name', help='exp_name', type=str, default="grasping")
    parser.add_argument('-w', '--weight', type=str, default='full_400.pt')
    parser.add_argument('-sd', '--storedir', type=str, default='data_all')
    parser.add_argument('-pr', '--prior', action="store_true")
    parser.add_argument('-o', '--obj_id', type=int, default=7)
    parser.add_argument('-t', '--test', action="store_true")
    parser.add_argument('-mc', '--mesh_collision', action="store_true")
    parser.add_argument('-ao', '--all_objects', action="store_true")
    parser.add_argument('-to', '--test_object_set', type=int, default=-1)
    parser.add_argument('-ac', '--all_contact', action="store_true")
    parser.add_argument('-seed', '--seed', type=int, default=1)
    parser.add_argument('-itr', '--num_iterations', type=int, default=3001)
    parser.add_argument('-nr', '--num_repeats', type=int, default=10)
    parser.add_argument('-ev', '--vis_evaluate', action="store_true")
    parser.add_argument('-sv', '--store_video', action="store_true")

    args = parser.parse_args()

    return args

def repeat_label(label_dict, num_repeats):
    ret = {}
    for k,v in label_dict.items():
        if 'type' in k or 'idx' in k or 'name' in k:
            ret[k] = np.repeat(v, num_repeats, 0)
        else :
            ret[k] = np.repeat(v,num_repeats,0).astype('float32')

    return ret

def euler_noise_to_quat(quats, palm_pose, noise):
    eulers_palm_mats = np.array([rotations.euler2mat(pose) for pose in palm_pose]).copy()
    eulers_mats =  np.array([rotations.quat2mat(quat) for quat in quats])

    rotmats_list = np.array([rotations.euler2mat(noise) for noise in noise])

    eulers_new = np.matmul(rotmats_list,eulers_mats)
    eulers_rotmated = np.array([rotations.mat2euler(mat) for mat in eulers_new])

    eulers_palm_new = np.matmul(rotmats_list,eulers_palm_mats)
    eulers_palm_rotmated = np.array([rotations.mat2euler(mat) for mat in eulers_palm_new])

    quat_list = [rotations.euler2quat(noise) for noise in eulers_rotmated]

    return np.array(quat_list), eulers_new, eulers_palm_rotmated