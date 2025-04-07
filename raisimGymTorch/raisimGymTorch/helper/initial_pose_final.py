# import numpy as np
# import json
# from collections import OrderedDict
import numpy.random
import torch
import numpy as np
import trimesh
# import os
# import yaml
# import h5py
# from sklearn.cluster import KMeans
# import open3d as o3d
# try:
#     from pytorch3d.ops.knn import knn_points
#     from pytorch3d.structures import Meshes
#     from pytorch3d.transforms.rotation_conversions import axis_angle_to_matrix
#     from utils.visualize import show_pointcloud_objhand
#     from mayavi import mlab
# except:
#     pass

# from networks.cmapnet_objhand import pointnet_reg
from scipy.spatial.transform import Rotation as R
from scipy.spatial import KDTree

import argparse
from sklearn.decomposition import PCA

grasp_pca = [-0.76798457, -0.7777186, -0.7385778, -0.7582664, -0.19614932, -0.75268096, 0.6672541, 0.75400156, \
             0.7918432, 0.7916097, -0.854025, 0.83326906, 0.74003863, -0.7825493, -0.83677787]


class GraspGenMesh():
    def __init__(self,verts,faces,sampled_point):
        # if verts is not a tensor

        self.verts = verts if torch.is_tensor(verts) else torch.tensor(verts,dtype=torch.float32).unsqueeze(0)

        self.faces = faces if torch.is_tensor(faces) else torch.tensor(faces,dtype=torch.int32).unsqueeze(0)

        self.sampled_point = sampled_point if torch.is_tensor(sampled_point) else torch.tensor(sampled_point,dtype=torch.float32).unsqueeze(0)

        # self.sampled_normal = sampled_normal if torch.is_tensor(sampled_normal) else torch.tensor(sampled_normal,dtype=torch.float32).unsqueeze(0)

    def to(self,device):
        self.verts = self.verts.to(device)
        self.faces = self.faces.to(device)
        self.sampled_point = self.sampled_point.to(device)
        # self.sampled_normal = self.sampled_normal.to(device)
        return self

    def repeat(self,n):
        self.verts = torch.repeat_interleave(self.verts,n,dim=0)
        self.faces = torch.repeat_interleave(self.faces,n,dim=0)
        self.sampled_point = torch.repeat_interleave(self.sampled_point,n,dim=0)
        # self.sampled_normal = torch.repeat_interleave(self.sampled_normal,n,dim=0)
        return self

    def __len__(self):
        return self.verts.shape[0]

    def get_trimesh(self,idx):
        return trimesh.Trimesh(vertices=self.verts[idx].detach().cpu().numpy(),faces=self.faces[idx].detach().cpu().numpy())

    def remove_by_mask(self,mask):
        self.verts = self.verts[mask]
        self.faces = self.faces[mask]
        self.sampled_point = self.sampled_point[mask]
        # self.sampled_normal = self.sampled_normal[mask]
        return self
def project_to_plane(points, normal_vector):
    return points - np.dot(points, normal_vector)[:, np.newaxis] * normal_vector
def find_smallest_boundary_axis(points, normal_vector):
    # Project the points onto the plane
    projected_points = project_to_plane(points, normal_vector)
    # Perform PCA to calculate the principal components
    pca = PCA(n_components=2)
    pca.fit(projected_points)
    principal_components = pca.components_

    # Find the axis with the smallest boundary (lowest variance)
    smallest_boundary_axis = np.argmin(pca.explained_variance_ratio_)

    #axis_lat, range_lat = principal_components[smallest_boundary_axis],pca.singular_values_[smallest_boundary_axis]
    axis_lat = principal_components[smallest_boundary_axis]

    projected_points_on_axis = np.dot(projected_points, axis_lat)

    # Find the minimum and maximum values
    min_value = np.min(projected_points_on_axis)
    max_value = np.max(projected_points_on_axis)

    range_lat = max_value - min_value

    projected_points = project_to_plane(points, axis_lat)
    # along axis_lat, find the projected_points with longest range
    new_axis = np.cross(axis_lat, normal_vector)
    projected_points = project_to_plane(projected_points,new_axis)
    length = np.linalg.norm(projected_points,axis=-1).max()
    return axis_lat, range_lat, length

def get_hand_rot(vec,  vec_in_hand=[1.2,1,0]):

    vec_in_hand = vec_in_hand/np.linalg.norm(vec_in_hand)
    vec_in_hand = vec_in_hand[np.newaxis]

    vec = vec/np.linalg.norm(vec,axis=-1,keepdims=True)
    axis = np.cross(vec_in_hand,vec)
    axis = axis/np.linalg.norm(axis,axis=-1,keepdims=True)
    angle = np.arccos((vec_in_hand*vec).sum(-1))
    return axis*angle[:,np.newaxis]


# generate initial pose for mano
def get_initial_pose(obj_mesh, non_aff_mesh, easy=False):
    # sample 3000 points from the pytorch3d mesh
    points = obj_mesh.vertices if torch.is_tensor(obj_mesh.vertices) else torch.tensor(obj_mesh.vertices,dtype=torch.float32).unsqueeze(0)
    obj_pcd = points.detach().cpu().numpy()
    aff_center = obj_mesh.centroid

    if non_aff_mesh is not None:
        aff_points, aff_face_id = trimesh.sample.sample_surface(obj_mesh, 200)
        aff_point = torch.from_numpy(aff_points).to('cuda').unsqueeze(0)
        non_aff_points, non_aff_face_id = trimesh.sample.sample_surface(non_aff_mesh, 200)
        non_aff_point = torch.from_numpy(non_aff_points).to('cuda').unsqueeze(0)
        non_aff_center = non_aff_mesh.centroid


    x_axis = [1.7, 1, 0]
    y_axis = [-np.sqrt(1/3.25), np.sqrt(2.25/3.25),0]
    z_axis = [0,0,1]


    got_sample = False
    while not got_sample:
        current_direction = np.random.uniform(-1, 1, (1, 3))
        dir = current_direction / np.linalg.norm(current_direction, axis=1, keepdims=True)

        supp_line = np.cross(z_axis,dir)
        if np.linalg.norm(supp_line) == 0:
            supp_line[0] = 1

        axis_list = []
        axis, lat_length, long_length = find_smallest_boundary_axis(obj_pcd[0], dir[0])
        axis_list.append(axis)
        axis_lat = axis.copy()

        if lat_length > 0.12:
            continue

        axis = np.stack(axis_list, axis=0)

        # rotate the x axis of the hand (grasping direction) to the target direction
        rot = get_hand_rot(dir, vec_in_hand=x_axis)
        rot_R = R.from_rotvec(rot)

        y_axis = rot_R.apply(y_axis)
        z_axis = rot_R.apply(z_axis)

        # calculate the angle to rotate the y axis of the hand to the direction of the narrowest boundary
        angle = np.arccos((y_axis*axis).sum(axis=-1))[:,np.newaxis]
        # angle[dir_mask] *= -1

        offset = 2*np.pi / 20
        if easy:
            rand_offset = 0
        else:
            rand_offset = np.random.uniform(-np.pi, np.pi)
        rot2_temp = dir * (angle+offset)
        z_axis = R.from_rotvec(rot2_temp).apply(z_axis)

        rot2 = dir * (angle + offset + rand_offset)
        rot12 = comp_axis_angle(rot2, rot)

        if easy:
            sample_point = aff_center
            ray_origins = sample_point + 0.5 * dir
            locations, index_ray, index_tri = obj_mesh.ray.intersects_location(ray_origins=ray_origins,
                                                                               ray_directions=-dir,
                                                                               multiple_hits=False)
            if len(locations) == 0:
                continue
        else:
            # projected_points = project_to_plane(obj_pcd[0] - aff_center, dir[0])
            projected_points = project_to_plane(obj_pcd[0], dir[0])
            rot_rand = dir * (rand_offset)

            axis_lat_rot = R.from_rotvec(rot_rand).apply(axis_lat)

            projected_points_on_axis = np.dot(projected_points, axis_lat_rot[0])

            # Find the minimum and maximum values
            min_value = np.min(projected_points_on_axis)
            max_value = np.max(projected_points_on_axis)

            rotated_lat_length = max_value - min_value

            if rotated_lat_length > 0.12:
                continue

            sample_idx = numpy.random.randint(0, obj_pcd[0].shape[0])
            sample_point = obj_pcd[0][sample_idx]
            ray_origins = sample_point + 0.5 * dir
            locations, index_ray, index_tri = obj_mesh.ray.intersects_location(ray_origins=ray_origins,
                                                                               ray_directions=-dir,
                                                                               multiple_hits=False)
            if len(locations) == 0:
                continue

            line_direction = (locations - aff_center)/np.linalg.norm(locations - aff_center)
            if (line_direction*dir).sum() < 0.:
                # print('got stuck because of line direction')
                continue

        if non_aff_mesh is not None:
            locations_torch = torch.from_numpy(locations.reshape(1,3)).to('cuda')
            non_aff_points, non_aff_face_id = trimesh.sample.sample_surface(non_aff_mesh, 200)
            non_aff_point = torch.from_numpy(non_aff_points).to('cuda').unsqueeze(0)
            non_af_dists = torch.cdist(locations_torch, non_aff_point)
            min_dis_non_af, min_idx_non_af = torch.min(non_af_dists, dim=2)
            if np.linalg.norm(locations - aff_center) > min_dis_non_af:
                # print(obj_name)
                # print('got stuck because of sample point')
                continue

        target = locations + 0.01 * (aff_center - locations)/np.linalg.norm(aff_center - locations)
        # target = locations
        pos = target + 0.3 * dir

        # bias = target - aff_center
        bias = target

        if non_aff_mesh is not None:
            pos_torch = torch.from_numpy(pos.reshape(1,3)).to('cuda')
            non_af_dists = torch.cdist(pos_torch, non_aff_point)
            min_dis_non_af, min_idx_non_af = torch.min(non_af_dists, dim=2)
            af_dists = torch.cdist(pos_torch, aff_point)
            min_dis_af, min_idx_af = torch.min(af_dists, dim=2)
            if min_dis_non_af < min_dis_af:
                # print("got stuck because of distance to non-affordance")
                continue

            hand_center_pos = target + 0.1 * dir
            if np.linalg.norm(hand_center_pos - aff_center) > np.linalg.norm(hand_center_pos - non_aff_center):
                # print(obj_name)
                # print('got stuck because of center distance')
                # print(np.linalg.norm(hand_center_pos - aff_center))
                # print(np.linalg.norm(hand_center_pos - non_aff_center))
                continue

            non_aff_locations, _, _ = non_aff_mesh.ray.intersects_location(ray_origins=ray_origins,
                                                                           ray_directions=-dir,
                                                                           multiple_hits=False)
            if len(non_aff_locations) > 0 and np.linalg.norm(ray_origins - non_aff_locations) < np.linalg.norm(ray_origins - locations):
                # print(obj_name)
                print('got stuck because of ray intersection')
                continue

        got_sample = True
        # print('center: ', target)
        # print('pos', target + (0.12 + long_length) * dir)
        # print('axis angle', rot12)
        # bias = np.array([[-0.05565647, -0.0425261, -0.0108046]])
        # rot12 = np.array([[-0.37194046, -1.48301687, 2.52831426]])
        # pos = np.array([[-0.04757783, -0.03444745, -0.00272595]])
        # opt_pos = target + (0.12 + long_length) * dir
    return rot12, pos, bias
    # return rot12, pos, bias, opt_pos

# get the initial pose for mano on the 500k+ dataset
def get_initial_pose_universal(obj_mesh, non_aff_mesh):
    # sample 3000 points from the pytorch3d mesh
    points = obj_mesh.vertices if torch.is_tensor(obj_mesh.vertices) else torch.tensor(obj_mesh.vertices,dtype=torch.float32).unsqueeze(0)
    obj_pcd = points.detach().cpu().numpy()
    aff_center = obj_mesh.centroid

    if non_aff_mesh is not None:
        aff_points, aff_face_id = trimesh.sample.sample_surface(obj_mesh, 200)
        aff_point = torch.from_numpy(aff_points).to('cuda').unsqueeze(0)
        non_aff_points, non_aff_face_id = trimesh.sample.sample_surface(non_aff_mesh, 200)
        non_aff_point = torch.from_numpy(non_aff_points).to('cuda').unsqueeze(0)
        non_aff_center = non_aff_mesh.centroid


    x_axis = [1.7, 1, 0]
    y_axis = [-np.sqrt(1/3.25), np.sqrt(2.25/3.25),0]
    z_axis = [0,0,1]


    sample_time = 0
    got_sample = False
    while not got_sample:
        sample_time += 1
        current_direction = np.random.uniform(-1, 1, (1, 3))
        dir = current_direction / np.linalg.norm(current_direction, axis=1, keepdims=True)

        supp_line = np.cross(z_axis,dir)
        if np.linalg.norm(supp_line) == 0:
            supp_line[0] = 1

        axis_list = []
        axis, lat_length, long_length = find_smallest_boundary_axis(obj_pcd[0], dir[0])
        axis_list.append(axis)
        axis_lat = axis.copy()

        if sample_time < 200:

            if lat_length > 0.14:
                continue

            axis = np.stack(axis_list, axis=0)
            # long_length_list = np.stack(long_length_list, axis=0)

            # rotate the x axis of the hand (grasping direction) to the target direction
            rot = get_hand_rot(dir, vec_in_hand=x_axis)
            rot_R = R.from_rotvec(rot)

            y_axis = rot_R.apply(y_axis)
            z_axis = rot_R.apply(z_axis)

            # calculate the angle to rotate the y axis of the hand to the direction of the narrowest boundary
            angle = np.arccos((y_axis*axis).sum(axis=-1))[:,np.newaxis]

            offset = 2*np.pi / 20
            rand_offset = np.random.uniform(-np.pi, np.pi)
            rot2_temp = dir * (angle+offset)
            z_axis = R.from_rotvec(rot2_temp).apply(z_axis)

            rot2 = dir * (angle + offset + rand_offset)
            rot12 = comp_axis_angle(rot2, rot)

            # projected_points = project_to_plane(obj_pcd[0] - aff_center, dir[0])
            projected_points = project_to_plane(obj_pcd[0], dir[0])
            rot_rand = dir * (rand_offset)

            axis_lat_rot = R.from_rotvec(rot_rand).apply(axis_lat)

            projected_points_on_axis = np.dot(projected_points, axis_lat_rot[0])

            # Find the minimum and maximum values
            min_value = np.min(projected_points_on_axis)
            max_value = np.max(projected_points_on_axis)

            rotated_lat_length = max_value - min_value

            if rotated_lat_length > 0.14:
                continue

            sample_idx = numpy.random.randint(0, obj_pcd[0].shape[0])
            sample_point = obj_pcd[0][sample_idx]
            ray_origins = sample_point + 0.5 * dir
            locations, index_ray, index_tri = obj_mesh.ray.intersects_location(ray_origins=ray_origins,
                                                                               ray_directions=-dir,
                                                                               multiple_hits=False)
            if len(locations) == 0:
                continue

            line_direction = (locations - aff_center)/np.linalg.norm(locations - aff_center)
            if (line_direction*dir).sum() < 0.:
                continue

            target = locations + 0.01 * (aff_center - locations)/np.linalg.norm(aff_center - locations)
            # target = locations
            pos = target + 0.3 * dir

            bias = target

            got_sample = True
        else:
            axis = np.stack(axis_list, axis=0)
            # long_length_list = np.stack(long_length_list, axis=0)

            # rotate the x axis of the hand (grasping direction) to the target direction
            rot = get_hand_rot(dir, vec_in_hand=x_axis)
            rot_R = R.from_rotvec(rot)

            y_axis = rot_R.apply(y_axis)
            z_axis = rot_R.apply(z_axis)

            # calculate the angle to rotate the y axis of the hand to the direction of the narrowest boundary
            angle = np.arccos((y_axis*axis).sum(axis=-1))[:,np.newaxis]

            offset = 2*np.pi / 20
            rand_offset = np.random.uniform(-np.pi, np.pi)
            rot2_temp = dir * (angle+offset)
            z_axis = R.from_rotvec(rot2_temp).apply(z_axis)

            rot2 = dir * (angle + offset + rand_offset)
            rot12 = comp_axis_angle(rot2, rot)

            sample_idx = numpy.random.randint(0, obj_pcd[0].shape[0])
            sample_point = obj_pcd[0][sample_idx]
            ray_origins = sample_point + 0.5 * dir
            locations, index_ray, index_tri = obj_mesh.ray.intersects_location(ray_origins=ray_origins,
                                                                               ray_directions=-dir,
                                                                               multiple_hits=False)
            if len(locations) == 0:
                sample_point = aff_center
                ray_origins = sample_point + 0.5 * dir
                locations, index_ray, index_tri = obj_mesh.ray.intersects_location(ray_origins=ray_origins,
                                                                                     ray_directions=-dir,
                                                                                     multiple_hits=False)

            target = locations + 0.01 * (aff_center - locations)/np.linalg.norm(aff_center - locations)
            pos = target + 0.3 * dir

            bias = target

            got_sample = True


    return rot12, pos, bias
    # return rot12, pos, bias, opt_pos

# get the initial pose for mano on the 500k+ dataset (for demostration)
def get_initial_pose_universal_demo(obj_mesh, non_aff_mesh):
    # sample 3000 points from the pytorch3d mesh
    points = obj_mesh.vertices if torch.is_tensor(obj_mesh.vertices) else torch.tensor(obj_mesh.vertices,dtype=torch.float32).unsqueeze(0)
    obj_pcd = points.detach().cpu().numpy()
    aff_center = obj_mesh.centroid

    if non_aff_mesh is not None:
        aff_points, aff_face_id = trimesh.sample.sample_surface(obj_mesh, 200)
        aff_point = torch.from_numpy(aff_points).to('cuda').unsqueeze(0)
        non_aff_points, non_aff_face_id = trimesh.sample.sample_surface(non_aff_mesh, 200)
        non_aff_point = torch.from_numpy(non_aff_points).to('cuda').unsqueeze(0)
        non_aff_center = non_aff_mesh.centroid


    x_axis = [1.7, 1, 0]
    y_axis = [-np.sqrt(1/3.25), np.sqrt(2.25/3.25),0]
    z_axis = [0,0,1]


    sample_time = 0
    got_sample = False
    while not got_sample:
        sample_time += 1
        current_direction = np.random.uniform(-1, 1, (1, 3))
        dir = current_direction / np.linalg.norm(current_direction, axis=1, keepdims=True)

        supp_line = np.cross(z_axis,dir)
        if np.linalg.norm(supp_line) == 0:
            supp_line[0] = 1

        axis_list = []
        axis, lat_length, long_length = find_smallest_boundary_axis(obj_pcd[0], dir[0])
        axis_list.append(axis)
        axis_lat = axis.copy()

        if sample_time < 200:

            if lat_length > 0.14:
                continue

            axis = np.stack(axis_list, axis=0)
            # long_length_list = np.stack(long_length_list, axis=0)

            # rotate the x axis of the hand (grasping direction) to the target direction
            rot = get_hand_rot(dir, vec_in_hand=x_axis)
            rot_R = R.from_rotvec(rot)

            y_axis = rot_R.apply(y_axis)
            z_axis = rot_R.apply(z_axis)

            # calculate the angle to rotate the y axis of the hand to the direction of the narrowest boundary
            angle = np.arccos((y_axis*axis).sum(axis=-1))[:,np.newaxis]

            offset = 2*np.pi / 20
            # rand_offset = np.random.uniform(-np.pi, np.pi)
            rand_offset = 0
            rot2_temp = dir * (angle+offset)
            z_axis = R.from_rotvec(rot2_temp).apply(z_axis)

            rot2 = dir * (angle + offset + rand_offset)
            rot12 = comp_axis_angle(rot2, rot)

            # projected_points = project_to_plane(obj_pcd[0] - aff_center, dir[0])
            projected_points = project_to_plane(obj_pcd[0], dir[0])
            rot_rand = dir * (rand_offset)

            axis_lat_rot = R.from_rotvec(rot_rand).apply(axis_lat)

            projected_points_on_axis = np.dot(projected_points, axis_lat_rot[0])

            # Find the minimum and maximum values
            min_value = np.min(projected_points_on_axis)
            max_value = np.max(projected_points_on_axis)

            rotated_lat_length = max_value - min_value

            if rotated_lat_length > 0.14:
                continue

            # sample_idx = numpy.random.randint(0, obj_pcd[0].shape[0])
            # sample_point = obj_pcd[0][sample_idx]
            sample_point = aff_center
            ray_origins = sample_point + 0.5 * dir
            locations, index_ray, index_tri = obj_mesh.ray.intersects_location(ray_origins=ray_origins,
                                                                               ray_directions=-dir,
                                                                               multiple_hits=False)
            if len(locations) == 0:
                continue

            line_direction = (locations - aff_center)/np.linalg.norm(locations - aff_center)
            if (line_direction*dir).sum() < 0.:
                continue

            target = locations + 0.01 * (aff_center - locations)/np.linalg.norm(aff_center - locations)
            # target = locations
            pos = target + 0.3 * dir

            bias = target

            got_sample = True
        else:
            axis = np.stack(axis_list, axis=0)
            # long_length_list = np.stack(long_length_list, axis=0)

            # rotate the x axis of the hand (grasping direction) to the target direction
            rot = get_hand_rot(dir, vec_in_hand=x_axis)
            rot_R = R.from_rotvec(rot)

            y_axis = rot_R.apply(y_axis)
            z_axis = rot_R.apply(z_axis)

            # calculate the angle to rotate the y axis of the hand to the direction of the narrowest boundary
            angle = np.arccos((y_axis*axis).sum(axis=-1))[:,np.newaxis]

            offset = 2*np.pi / 20
            # rand_offset = np.random.uniform(-np.pi, np.pi)
            rand_offset = 0
            rot2_temp = dir * (angle+offset)
            z_axis = R.from_rotvec(rot2_temp).apply(z_axis)

            rot2 = dir * (angle + offset + rand_offset)
            rot12 = comp_axis_angle(rot2, rot)

            # sample_idx = numpy.random.randint(0, obj_pcd[0].shape[0])
            # sample_point = obj_pcd[0][sample_idx]
            sample_point = aff_center
            ray_origins = sample_point + 0.5 * dir
            locations, index_ray, index_tri = obj_mesh.ray.intersects_location(ray_origins=ray_origins,
                                                                               ray_directions=-dir,
                                                                               multiple_hits=False)
            # if len(locations) == 0:
            #     sample_point = aff_center
            #     ray_origins = sample_point + 0.5 * dir
            #     locations, index_ray, index_tri = obj_mesh.ray.intersects_location(ray_origins=ray_origins,
            #                                                                        ray_directions=-dir,
            #                                                                        multiple_hits=False)

            target = locations + 0.01 * (aff_center - locations)/np.linalg.norm(aff_center - locations)
            pos = target + 0.3 * dir

            bias = target

            got_sample = True


    return rot12, pos, bias
    # return rot12, pos, bias, opt_pos

# get the initial pose for 4 kinds of hands with given objectives
def get_initial_pose_set(obj_mesh, non_aff_mesh, direction, rotation, point, hand=None):
    # sample 3000 points from the pytorch3d mesh
    points = obj_mesh.vertices if torch.is_tensor(obj_mesh.vertices) else torch.tensor(obj_mesh.vertices,dtype=torch.float32).unsqueeze(0)
    obj_pcd = points.detach().cpu().numpy()
    aff_center = obj_mesh.centroid

    if non_aff_mesh is not None:
        aff_points, aff_face_id = trimesh.sample.sample_surface(obj_mesh, 200)
        aff_point = torch.from_numpy(aff_points).to('cuda').unsqueeze(0)
        non_aff_points, non_aff_face_id = trimesh.sample.sample_surface(non_aff_mesh, 200)
        non_aff_point = torch.from_numpy(non_aff_points).to('cuda').unsqueeze(0)
        non_aff_center = non_aff_mesh.centroid


    x_axis = [1.7, 1, 0]
    y_axis = [-np.sqrt(1/3.25), np.sqrt(2.25/3.25),0]
    z_axis = [0,0,1]

    if hand == 'faive':
        rot_mat= [[0,0,-1],[-1,0,0],[0,1,0]]
        rot_mat = np.array(rot_mat).reshape(3,3)
    elif hand == 'shadow':
        rot_mat = [[0,0,1],[0,1,0],[-1,0,0]]
        rot_mat = np.array(rot_mat).reshape(3,3)
    elif hand == 'allegro':
        rot_mat = [[0,-1,0],[0,0,1],[-1,0,0]]
        rot_mat = np.array(rot_mat).reshape(3,3)
    else:
        rot_mat = np.eye(3)

    x_axis = np.matmul(rot_mat, x_axis)
    y_axis = np.matmul(rot_mat, y_axis)
    z_axis = np.matmul(rot_mat, z_axis)

    x_axis = x_axis.reshape(3,)
    y_axis = y_axis.reshape(3,)
    z_axis = z_axis.reshape(3,)

    dir = direction / np.linalg.norm(direction, axis=1, keepdims=True)

    supp_line = np.cross(z_axis,dir)
    if np.linalg.norm(supp_line) == 0:
        supp_line[0] = 1

    axis_list = []
    axis, lat_length, long_length = find_smallest_boundary_axis(obj_pcd[0], dir[0])
    axis_list.append(axis)
    axis_lat = axis.copy()

    axis = np.stack(axis_list, axis=0)
    # long_length_list = np.stack(long_length_list, axis=0)

    # rotate the x axis of the hand (grasping direction) to the target direction
    rot = get_hand_rot(dir, vec_in_hand=x_axis)
    rot_R = R.from_rotvec(rot)

    y_axis = rot_R.apply(y_axis)
    z_axis = rot_R.apply(z_axis)

    # calculate the angle to rotate the y axis of the hand to the direction of the narrowest boundary
    angle = np.arccos((y_axis*axis).sum(axis=-1))[:,np.newaxis]
    # angle[dir_mask] *= -1

    offset = 2*np.pi / 20
    rand_offset = rotation
    rot2_temp = dir * (angle+offset)
    z_axis = R.from_rotvec(rot2_temp).apply(z_axis)

    rot2 = dir * (angle + offset + rand_offset)
    rot12 = comp_axis_angle(rot2, rot)

    sample_point = point
    ray_origins = sample_point + 0.5 * dir
    locations, index_ray, index_tri = obj_mesh.ray.intersects_location(ray_origins=ray_origins,
                                                                       ray_directions=-dir,
                                                                       multiple_hits=False)

    target = locations + 0.01 * (aff_center - locations)/np.linalg.norm(aff_center - locations)
    # target = sample_point + 0.01 * (aff_center - sample_point)/np.linalg.norm(aff_center - sample_point)

    pos = target + 0.3 * dir

    bias = target

    return rot12, pos, bias

# get the initial pose for robot hands (trianing)
def get_initial_pose_faive(obj_mesh, non_aff_mesh, hand_type='faive'):
    # sample 3000 points from the pytorch3d mesh
    points = obj_mesh.vertices if torch.is_tensor(obj_mesh.vertices) else torch.tensor(obj_mesh.vertices,
                                                                                       dtype=torch.float32).unsqueeze(0)
    obj_pcd = points.detach().cpu().numpy()
    aff_center = obj_mesh.centroid

    if non_aff_mesh is not None:
        aff_points, aff_face_id = trimesh.sample.sample_surface(obj_mesh, 200)
        aff_point = torch.from_numpy(aff_points).to('cuda').unsqueeze(0)
        non_aff_points, non_aff_face_id = trimesh.sample.sample_surface(non_aff_mesh, 200)
        non_aff_point = torch.from_numpy(non_aff_points).to('cuda').unsqueeze(0)
        non_aff_center = non_aff_mesh.centroid

    x_axis = np.array([1.7, 1, 0])
    y_axis = np.array([-np.sqrt(1 / 3.25), np.sqrt(2.25 / 3.25), 0])
    z_axis = np.array([0, 0, 1])

    if hand_type == 'faive':
        rot_mat = [[0, 0, -1], [-1, 0, 0], [0, 1, 0]]
        rot_mat = np.array(rot_mat).reshape(3, 3)
    elif hand_type == 'shadow':
        rot_mat = [[0, 0, 1], [0, 1, 0], [-1, 0, 0]]
        rot_mat = np.array(rot_mat).reshape(3, 3)
    elif hand_type == 'allegro':
        rot_mat = [[0, -1, 0], [0, 0, 1], [-1, 0, 0]]
        rot_mat = np.array(rot_mat).reshape(3, 3)

    x_axis = np.matmul(rot_mat, x_axis)
    y_axis = np.matmul(rot_mat, y_axis)
    z_axis = np.matmul(rot_mat, z_axis)

    x_axis = x_axis.reshape(3, )
    y_axis = y_axis.reshape(3, )
    z_axis = z_axis.reshape(3, )

    got_sample = False
    while not got_sample:
        current_direction = np.random.uniform(-1, 1, (1, 3))
        dir = current_direction / np.linalg.norm(current_direction, axis=1, keepdims=True)

        supp_line = np.cross(z_axis, dir)
        if np.linalg.norm(supp_line) == 0:
            supp_line[0] = 1

        axis_list = []

        axis, lat_length, long_length = find_smallest_boundary_axis(obj_pcd[0], dir[0])
        axis_list.append(axis)
        axis_lat = axis.copy()

        if lat_length > 0.12:
            continue

        axis = np.stack(axis_list, axis=0)
        # long_length_list = np.stack(long_length_list, axis=0)

        # rotate the x axis of the hand (grasping direction) to the target direction
        rot = get_hand_rot(dir, vec_in_hand=x_axis)
        rot_R = R.from_rotvec(rot)

        y_axis = rot_R.apply(y_axis)
        z_axis = rot_R.apply(z_axis)

        # calculate the angle to rotate the y axis of the hand to the direction of the narrowest boundary
        angle = np.arccos((y_axis * axis).sum(axis=-1))[:, np.newaxis]
        # angle[dir_mask] *= -1

        offset = 2 * np.pi / 20
        rand_offset = np.random.uniform(-np.pi, np.pi)
        rot2_temp = dir * (angle + offset)
        z_axis = R.from_rotvec(rot2_temp).apply(z_axis)

        rot2 = dir * (angle + offset + rand_offset)
        rot12 = comp_axis_angle(rot2, rot)

        # projected_points = project_to_plane(obj_pcd[0] - aff_center, dir[0])
        projected_points = project_to_plane(obj_pcd[0], dir[0])
        rot_rand = dir * (rand_offset)

        axis_lat_rot = R.from_rotvec(rot_rand).apply(axis_lat)

        projected_points_on_axis = np.dot(projected_points, axis_lat_rot[0])

        # Find the minimum and maximum values
        min_value = np.min(projected_points_on_axis)
        max_value = np.max(projected_points_on_axis)

        rotated_lat_length = max_value - min_value

        if rotated_lat_length > 0.12:
            continue

        sample_idx = numpy.random.randint(0, obj_pcd[0].shape[0])
        sample_point = obj_pcd[0][sample_idx]
        ray_origins = sample_point + 0.5 * dir
        locations, index_ray, index_tri = obj_mesh.ray.intersects_location(ray_origins=ray_origins,
                                                                           ray_directions=-dir,
                                                                           multiple_hits=False)
        if len(locations) == 0:
            continue

        line_direction = (locations - aff_center) / np.linalg.norm(locations - aff_center)
        if (line_direction * dir).sum() < 0.:
            # print('got stuck because of line direction')
            continue

        if non_aff_mesh is not None:
            locations_torch = torch.from_numpy(locations.reshape(1, 3)).to('cuda')
            non_aff_points, non_aff_face_id = trimesh.sample.sample_surface(non_aff_mesh, 200)
            non_aff_point = torch.from_numpy(non_aff_points).to('cuda').unsqueeze(0)
            non_af_dists = torch.cdist(locations_torch, non_aff_point)
            min_dis_non_af, min_idx_non_af = torch.min(non_af_dists, dim=2)
            if np.linalg.norm(locations - aff_center) > min_dis_non_af:
                # print(obj_name)
                # print('got stuck because of sample point')
                continue

        target = locations + 0.01 * (aff_center - locations) / np.linalg.norm(aff_center - locations)
        # target = locations
        pos = target + 0.3 * dir

        # bias = target - aff_center
        bias = target

        if non_aff_mesh is not None:
            pos_torch = torch.from_numpy(pos.reshape(1, 3)).to('cuda')
            non_af_dists = torch.cdist(pos_torch, non_aff_point)
            min_dis_non_af, min_idx_non_af = torch.min(non_af_dists, dim=2)
            af_dists = torch.cdist(pos_torch, aff_point)
            min_dis_af, min_idx_af = torch.min(af_dists, dim=2)
            if min_dis_non_af < min_dis_af:
                # print("got stuck because of distance to non-affordance")
                continue

            hand_center_pos = target + 0.1 * dir
            if np.linalg.norm(hand_center_pos - aff_center) > np.linalg.norm(hand_center_pos - non_aff_center):
                continue

            non_aff_locations, _, _ = non_aff_mesh.ray.intersects_location(ray_origins=ray_origins,
                                                                           ray_directions=-dir,
                                                                           multiple_hits=False)
            if len(non_aff_locations) > 0 and np.linalg.norm(ray_origins - non_aff_locations) < np.linalg.norm(
                    ray_origins - locations):
                # print(obj_name)
                print('got stuck because of ray intersection')
                continue

        got_sample = True

    return rot12, pos, bias
    # return rot12, pos, bias, opt_pos

# get the label for robot hands (evaluation)
def get_initial_pose_faive_label(obj_mesh, non_aff_mesh, hand_type='faive', load_dir=None, load_offset=None,
                                 load_center=None):
    # sample 3000 points from the pytorch3d mesh
    points = obj_mesh.vertices if torch.is_tensor(obj_mesh.vertices) else torch.tensor(obj_mesh.vertices,
                                                                                       dtype=torch.float32).unsqueeze(0)
    obj_pcd = points.detach().cpu().numpy()
    aff_center = obj_mesh.centroid

    if non_aff_mesh is not None:
        aff_points, aff_face_id = trimesh.sample.sample_surface(obj_mesh, 200)
        aff_point = torch.from_numpy(aff_points).to('cuda').unsqueeze(0)
        non_aff_points, non_aff_face_id = trimesh.sample.sample_surface(non_aff_mesh, 200)
        non_aff_point = torch.from_numpy(non_aff_points).to('cuda').unsqueeze(0)
        non_aff_center = non_aff_mesh.centroid

    x_axis = np.array([1.7, 1, 0])
    y_axis = np.array([-np.sqrt(1 / 3.25), np.sqrt(2.25 / 3.25), 0])
    z_axis = np.array([0, 0, 1])

    if hand_type == 'faive':
        rot_mat = [[0, 0, -1], [-1, 0, 0], [0, 1, 0]]
        rot_mat = np.array(rot_mat).reshape(3, 3)
    elif hand_type == 'shadow':
        rot_mat = [[0, 0, 1], [0, 1, 0], [-1, 0, 0]]
        rot_mat = np.array(rot_mat).reshape(3, 3)
    elif hand_type == 'allegro':
        rot_mat = [[0, -1, 0], [0, 0, 1], [-1, 0, 0]]
        rot_mat = np.array(rot_mat).reshape(3, 3)

    x_axis = np.matmul(rot_mat, x_axis)
    y_axis = np.matmul(rot_mat, y_axis)
    z_axis = np.matmul(rot_mat, z_axis)

    x_axis = x_axis.reshape(3, )
    y_axis = y_axis.reshape(3, )
    z_axis = z_axis.reshape(3, )

    dir = load_dir

    supp_line = np.cross(z_axis, dir)
    if np.linalg.norm(supp_line) == 0:
        supp_line[0] = 1

    axis_list = []

    axis, lat_length, long_length = find_smallest_boundary_axis(obj_pcd[0], dir[0])
    axis_list.append(axis)
    axis_lat = axis.copy()

    # if lat_length > 0.12:
    #     continue

    axis = np.stack(axis_list, axis=0)
    # long_length_list = np.stack(long_length_list, axis=0)

    # rotate the x axis of the hand (grasping direction) to the target direction
    rot = get_hand_rot(dir, vec_in_hand=x_axis)
    rot_R = R.from_rotvec(rot)

    y_axis = rot_R.apply(y_axis)
    z_axis = rot_R.apply(z_axis)

    # calculate the angle to rotate the y axis of the hand to the direction of the narrowest boundary
    angle = np.arccos((y_axis * axis).sum(axis=-1))[:, np.newaxis]
    # angle[dir_mask] *= -1

    offset = 2 * np.pi / 20
    rand_offset = load_offset
    rot2_temp = dir * (angle + offset)
    z_axis = R.from_rotvec(rot2_temp).apply(z_axis)

    rot2 = dir * (angle + offset + rand_offset)
    rot12 = comp_axis_angle(rot2, rot)

    target = load_center
    pos = target + 0.3 * dir

    bias = target

    return rot12, pos, bias
def comp_axis_angle(axisang1, axisang2):
    # transform two angle into scipy rotation
    R1 = R.from_rotvec(axisang1)
    R2 = R.from_rotvec(axisang2)
    # compose two rotation
    R12 = R1 * R2
    # transform back to axis angle
    axisang12 = R12.as_rotvec()
    return axisang12
