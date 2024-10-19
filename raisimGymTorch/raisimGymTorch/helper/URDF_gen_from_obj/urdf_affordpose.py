import json
import math
import os.path
import xml.etree.ElementTree as ET
from xml.dom import minidom

import numpy as np

from rotation_helper import from_to_as_rot
from mano_helpers import get_kinematic_order_mano, is_finger_joint, get_child_finger_joint_name, is_wrist, is_tip_of_finger, \
    is_finger_part_thumb, is_right_body_joint, get_kinematic_child_joint, \
    is_hand, is_finger_part_index, is_finger_part_other, get_limit, get_mano_part_number


def get_finger_part_radius(name):
    # name eg: 'right_index1'
    assert (is_finger_joint(name))
    # radius based on d-grasp mean_mano.urdf
    radius_dict = {
        'index1': 0.009,
        'index2': 0.0075,
        'index3': 0.0075,
        'middle1': 0.009,
        'middle2': 0.0075,
        'middle3': 0.0075,
        'ring1': 0.009,
        'ring2': 0.0075,
        'ring3': 0.0075,
        'pinky1': 0.009,
        'pinky2': 0.0075,
        'pinky3': 0.0065,
        'thumb1': 0.01,
        'thumb2': 0.01,
        'thumb3': 0.01
    }
    finger_part = name.split("_")[1]

    return radius_dict[finger_part]


def val_to_str(val, precision='.16g'):
    return f'{val:{precision}}'


def list_to_str(num_list, precision='.16g'):
    return ' '.join([f"{val:{precision}}" for val in num_list])


def add_origin(parent_tag, rpy=None, xyz=[0, 0, 0]):
    xyz_str = list_to_str(xyz)
    if rpy is not None:
        rpy_str = list_to_str(rpy)
        origin_tag = ET.SubElement(parent_tag, 'origin', {'rpy': rpy_str, f'xyz': xyz_str})
    else:
        origin_tag = ET.SubElement(parent_tag, 'origin', {f'xyz': xyz_str})

    return origin_tag


def add_inertial(parent_tag, rpy=None, xyz=None, mass=0.0, inertia=[0, 0, 0, 0, 0, 0]):
    inertial_tag = ET.SubElement(parent_tag, 'inertial')
    if xyz is not None:
        add_origin(inertial_tag, rpy, xyz)

    mass_str = val_to_str(mass)
    mass_tag = ET.SubElement(inertial_tag, 'mass', {'value': mass_str})

    inertia_tag = ET.SubElement(inertial_tag, 'inertia',
                                {'ixx': val_to_str(inertia[0]),
                                 'ixy': val_to_str(inertia[1]),
                                 'ixz': val_to_str(inertia[2]),
                                 'iyy': val_to_str(inertia[3]),
                                 'iyz': val_to_str(inertia[4]),
                                 'izz': val_to_str(inertia[5])})


def add_geometry(parent_tag, mesh_fname, mesh_scale):
    geometry_tag = ET.SubElement(parent_tag, 'geometry')
    scale_str = list_to_str(mesh_scale)
    mesh_tag = ET.SubElement(geometry_tag, 'mesh', {'filename': mesh_fname, 'scale': scale_str})


def add_geometry_box(parent_tag, size=[0, 0, 0]):
    geometry_tag = ET.SubElement(parent_tag, 'geometry')
    capsule_tag = ET.SubElement(geometry_tag, 'box', {'size': list_to_str(size)})


def add_geometry_capsule(parent_tag, radius, length):
    geometry_tag = ET.SubElement(parent_tag, 'geometry')
    capsule_tag = ET.SubElement(geometry_tag, 'capsule', {'length': val_to_str(length), 'radius': val_to_str(radius)})


def add_geometry_sphere(parent_tag, radius):
    geometry_tag = ET.SubElement(parent_tag, 'geometry')
    sphere_tag = ET.SubElement(geometry_tag, 'sphere', {'radius': val_to_str(radius)})


def add_geometry_cylinder(parent_tag, radius, length):
    geometry_tag = ET.SubElement(parent_tag, 'geometry')
    cylinder_tag = ET.SubElement(geometry_tag, 'cylinder', {'length': val_to_str(length), 'radius': val_to_str(radius)})


def add_material_contact(parent_tag, name):
    material_tag = ET.SubElement(parent_tag, 'material', {'name': ''})
    contact_tag = ET.SubElement(material_tag, 'contact', {'name': name})


def add_visual(parent_tag, mesh_fname, material_name, mesh_scale=[1, 1, 1], rpy=[0, 0, 0], xyz=[0, 0, 0]):
    visual_tag = ET.SubElement(parent_tag, 'visual')

    add_origin(visual_tag, rpy, xyz)

    add_geometry(visual_tag, mesh_fname, mesh_scale)

    material_tag = ET.SubElement(visual_tag, 'material', {'name': material_name})




def add_collision(parent_tag, xyz, rpy, material_contact_name):
    collision_tag = ET.SubElement(parent_tag, 'collision')

    add_origin(collision_tag, rpy, xyz)

    if material_contact_name != '':
        add_material_contact(collision_tag, material_contact_name)

    return collision_tag


def add_collision_mesh(parent_tag, mesh_fname, mesh_scale=[1, 1, 1], rpy=[0, 0, 0], xyz=[0, 0, 0],
                       material_contact_name=''):
    collision_tag = add_collision(parent_tag, xyz, rpy, material_contact_name)

    add_geometry(collision_tag, mesh_fname, mesh_scale)


def add_collision_capsule(parent_tag, radius, length, rpy=[0, 0, 0], xyz=[0, 0, 0],
                          material_contact_name=''):
    collision_tag = add_collision(parent_tag, xyz, rpy, material_contact_name)

    add_geometry_capsule(collision_tag, radius, length)


def add_collision_sphere(parent_tag, radius, rpy=[0, 0, 0], xyz=[0, 0, 0],
                         material_contact_name=''):
    collision_tag = add_collision(parent_tag, xyz, rpy, material_contact_name)

    add_geometry_sphere(collision_tag, radius)


def add_collision_cylinder(parent_tag, radius, length, rpy=[0, 0, 0], xyz=[0, 0, 0],
                           material_contact_name=''):
    collision_tag = add_collision(parent_tag, xyz, rpy, material_contact_name)

    add_geometry_cylinder(collision_tag, radius, length)


def add_collision_box(parent_tag, size=[1, 1, 1], rpy=[0, 0, 0], xyz=[0, 0, 0],
                      material_contact_name=''):
    collision_tag = add_collision(parent_tag, xyz, rpy, material_contact_name)

    add_geometry_box(collision_tag, size)


def add_link_element(root, name):
    link = ET.SubElement(root, 'link', {'name': name})
    return link


def add_joint(root, name, orig_xyz, axis_xyz, parent_link_name,
              child_link_name, effort='1000', limits=None, joint_type='revolute', is_obj=False):
    joint = ET.SubElement(root, 'joint', {'name': name, 'type': joint_type})
    add_origin(joint, xyz=orig_xyz)
    axis = ET.SubElement(joint, 'axis', {'xyz': list_to_str(axis_xyz)})

    parent = ET.SubElement(joint, 'parent', {'link': parent_link_name})
    child = ET.SubElement(joint, 'child', {'link': child_link_name})


    if type != 'fixed' and limits is not None:
        # http://wiki.ros.org/pr2_controller_manager/safety_limits
        if joint_type=='prismatic':
            dynamics = ET.SubElement(joint, 'dynamics', {'damping': '10.0', 'friction': '0.0001'})
        else:
            dynamics = ET.SubElement(joint, 'dynamics', {'damping': '1.0', 'friction': '1.0'})

        limit = ET.SubElement(joint, 'limit', {  # TODO how to set effort and velocity ? effort=1000 in d-grasp
            'effort': '0.1',  # Nm
            'velocity': '0.1',  # rad/s; if zero joints can (almost) not move!
            'lower': val_to_str(limits[0]), 'upper': val_to_str(limits[1])
        })


def approx_mano_collision(main_link, child, joints_dict, bounding_cylinder, material_contact_name=''):
    # child eg: 'right_index1'
    if is_finger_joint(child):
        # capsule = joint_mesh_data[child]['bounding_cylinder']
        # radius = capsule['radius']
        radius = get_finger_part_radius(name=child)
        # radius is the capsule radius of a finger part according to mano_mean.urdf

        finger_joint_child = get_child_finger_joint_name(child)

        direction = joints_dict[finger_joint_child] - joints_dict[child]

        #  slightly shift the tip of finger position, since it is on top of the finger
        if is_tip_of_finger(finger_joint_child):
            if is_finger_part_thumb(child):
                shift = np.asarray([0.002, -0.002, -0.002])
                if is_right_body_joint(child):
                    shift = np.asarray([-0.005, -0.004, -0.003])
            else:
                shift = np.asarray([0, -0.003, 0])

            direction = direction + shift
            #direction = bounding_cylinder['direction']
            #length = bounding_cylinder['height']

        length = np.linalg.norm(direction)
        direction = direction / np.linalg.norm(direction)

        capsule_vect = [0, 0, 1]  # capsule points up
        rpy = from_to_as_rot(direction,
                             capsule_vect) * -1.0  # TODO why does it not work with from_to_vect_to_rot(capsule_vect, direction)

        add_collision_capsule(main_link,
                              radius=radius,
                              length=length - radius,
                              rpy=rpy,
                              xyz=direction * length / 2 - direction * radius / 2,
                              material_contact_name=material_contact_name)

    elif is_wrist(child):
        # bbox_length, bbox_height, bbox_width = joint_mesh_data[child]['bounding_box']['size']
        # bbox_height /= 2
        # d-grasp mano_mean.urdf values
        bbox_length, bbox_height, bbox_width = [0.08780, 0.027, 0.07]

        body_side = child.split("_")[0]

        pinky_pos = joints_dict[body_side + "_" + 'pinky1']
        index_pos = joints_dict[body_side + "_" + 'index1']

        hand_tip = pinky_pos + (index_pos - pinky_pos) / 2

        hand_direction = hand_tip - joints_dict[child]
        hand_length = np.linalg.norm(hand_direction)
        hand_direction /= hand_length

        if is_right_body_joint(child):  # adjust direction of the box for right and left hand
            box_vect = [-1, 0, 0]  # negative x direction
        else:
            box_vect = [1, 0, 0]  # positive x direction
        rpy = from_to_as_rot(box_vect, hand_direction)

        xyz = hand_direction * hand_length / 2
        # shift box up
        xyz = xyz + np.asarray([0, 0, 0.005])

        add_collision_box(main_link, size=[bbox_length, bbox_height, bbox_width],
                          rpy=rpy,
                          xyz=xyz,
                          material_contact_name=material_contact_name)

        # add heel of hand
        # radius from d-grasp mano_mean.urdf
        radius = 0.015
        wrist_to_thumb_vect = joints_dict[body_side + "_" + "thumb1"] - joints_dict[child]
        length = np.linalg.norm(wrist_to_thumb_vect) - radius

        rpy = from_to_as_rot(wrist_to_thumb_vect, [0, 0, 1]) * -1
        xyz = wrist_to_thumb_vect * 0.6

        add_collision_capsule(main_link, radius=radius, length=length,
                              rpy=rpy,
                              xyz=xyz,
                              material_contact_name=material_contact_name)


def create_urdf_tree(hand_name, joints_dict, joint_mesh_data=None, pose_limits=None,
                     is_rhand=True, delta=math.pi / 64,
                     mesh_relative_out_path='./meshes',
                     mano_collision_approx=True,
                     rough_body_collision_approx=False):
    
    # For fingers, there are five links for each finger.
    # Finger tip link can be egnored at first.
    # Fixed finger link can also be ignored at first
    # Each other links contains three links, x, y, z
    parent_dict = get_kinematic_order_mano(is_rhand)
    dir_x_axis = 1 #if is_rhand else -1 # temporarily use, may not rely much on it

    root = ET.Element('robot', {'name': hand_name})
    material = ET.SubElement(root, 'material', {'name': 'hand_color'})
    color = ET.SubElement(material, 'color', {'rgba': '0.91796875 0.765 0.5234 1'})
    world = add_link_element(root, 'world')
    add_joint(root, 'world_to_base', orig_xyz=[0, 0, 0], axis_xyz=[0, 0, 0], 
              parent_link_name='world', child_link_name='sliderBar', 
              limits=None, joint_type='fixed')
    sliderBar = add_link_element(root, 'sliderBar')
    add_inertial(sliderBar, mass=0, inertia=[1.0, 0.0, 0.0, 1.0, 0.0, 1.0])

    joint_order = []
    part_number = get_mano_part_number(is_rhand)
    joint_limit_finger = get_limit(is_rhand) 

    for child, parent in parent_dict.items():
        if child[-1] == '1':
            fixed_link = add_link_element(root, f'{child[:-1]}_fixed')
            add_inertial(fixed_link, rpy=[0, 0, 0], xyz=[0, 0, 0], mass=0, 
                         inertia=[0, 0, 0, 0, 0, 0])
            # fixed joint
            add_joint(root, f'{child}_fixed', orig_xyz=[0, 0, 0], axis_xyz=[0, 0, 0],
                      parent_link_name=f'{parent}_rz', child_link_name=fixed_link.attrib['name'],
                      joint_type='fixed')
        joint_order += [child]
        main_link_name = f'{child}'
        main_link = add_link_element(root, main_link_name)
        bounding_cylinder = joint_mesh_data[child]['bounding_cylinder']
        inertial_data = joint_mesh_data[child]
        add_inertial(main_link,
                     rpy=inertial_data['origin']['rpy'],
                     # subtract absolute position from center of mass
                     xyz=inertial_data['origin']['xyz'] - joints_dict[child],
                     mass=inertial_data['mass']['value'],
                     inertia=list(inertial_data['inertia'].values()))
        if is_wrist(child):
            y_name = f'{child}_y'
            y_link = add_link_element(root, y_name)
            add_inertial(y_link, rpy=[0, 0, 0], xyz=[0, 0, 0],
                         mass=0.0, inertia=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            z_name = f'{child}_z'
            z_link = add_link_element(root, z_name)
            add_inertial(z_link, rpy=[0, 0, 0], xyz=[0, 0, 0],
                         mass=0.0, inertia=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            rx_name = f'{child}_rx'
            rx_link = add_link_element(root, rx_name)
            add_inertial(rx_link, rpy=[0, 0, 0], xyz=[0, 0, 0],
                         mass=0.0, inertia=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            ry_name = f'{child}_ry'
            ry_link = add_link_element(root, ry_name)
            add_inertial(ry_link, rpy=[0, 0, 0], xyz=[0, 0, 0],
                         mass=0.0, inertia=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            rz_name = f'{child}_rz'
            rz_link = add_link_element(root, rz_name)
            add_inertial(rz_link, rpy=[0, 0, 0], xyz=[0, 0, 0],
                         mass=0.0, inertia=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            mesh_fname = os.path.join(mesh_relative_out_path, child)
            add_visual(rz_link, mesh_fname=mesh_fname + '.stl', material_name='hand_color', xyz=-joints_dict[child])
            if mano_collision_approx:
                approx_mano_collision(main_link=rz_link, child=child, joints_dict=joints_dict, 
                                      bounding_cylinder=bounding_cylinder, 
                                      material_contact_name='finger')
            else:
                add_collision_mesh(rz_link, mesh_fname=mesh_fname + '.obj', xyz=-joints_dict[child],
                                   material_contact_name=material_contact_name)
        else:
            y_link_name = f'{child}_y'
            y_link = add_link_element(root, y_link_name)
            add_inertial(y_link, rpy=[0, 0, 0], xyz=[0, 0, 0],
                         mass=0.0, inertia=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            z_link_name = f'{child}_z'
            z_link = add_link_element(root, z_link_name)
            add_inertial(z_link, rpy=[0, 0, 0], xyz=[0, 0, 0], 
                         mass=0.0, inertia=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            bounding_cylinder = joint_mesh_data[child]['bounding_cylinder']
            mesh_fname = os.path.join(mesh_relative_out_path, child)
            # the mesh uses global coordinates -> subtract global joint origin to get relative coordinates
            add_visual(z_link, mesh_fname=mesh_fname + '.stl', material_name='hand_color', xyz=-joints_dict[child])
            if mano_collision_approx:
                approx_mano_collision(main_link=z_link, child=child, joints_dict=joints_dict, 
                                      bounding_cylinder=bounding_cylinder, 
                                      material_contact_name='finger')
            else:
                add_collision_mesh(z_link, mesh_fname=mesh_fname + '.obj', xyz=-joints_dict[child],
                                   material_contact_name=material_contact_name)

        # child eg: 'right_index1'
        if not is_wrist(child):
            pose_limit = [joint_limit_finger[0][part_number[child]*3:part_number[child]*3+3],
                          joint_limit_finger[1][part_number[child]*3:part_number[child]*3+3]]

        if is_wrist(child):
            pose_limit = [[-0.8    , 0.8    ], 
                          [-6.28   , 6.28   ]]
            add_joint(root, f'{child}_0x', orig_xyz=[0, 0, 0], axis_xyz=[1, 0, 0],
                      parent_link_name='sliderBar', child_link_name=f'{child}', 
                      effort='100', limits=pose_limit[0], joint_type='prismatic')
            add_joint(root, f'{child}_0y', orig_xyz=[0, 0, 0], axis_xyz=[0, dir_x_axis, 0],
                      parent_link_name=f'{child}', child_link_name=f'{child}_y', 
                      effort='100', limits=pose_limit[0], joint_type='prismatic')
            add_joint(root, f'{child}_0z', orig_xyz=[0, 0, 0], axis_xyz=[0, 0, dir_x_axis],
                      parent_link_name=f'{child}_y', child_link_name=f'{child}_z', 
                      effort='100', limits=pose_limit[0], joint_type='prismatic')
            add_joint(root, f'{child}_0rx', orig_xyz=[0, 0, 0], axis_xyz=[1, 0, 0],
                      parent_link_name=f'{child}_z', child_link_name=f'{child}_rx',
                      limits=pose_limit[1])
            add_joint(root, f'{child}_0ry', orig_xyz=[0, 0, 0], axis_xyz=[0, dir_x_axis, 0],
                      parent_link_name=f'{child}_rx', child_link_name=f'{child}_ry',
                      limits=pose_limit[1])
            add_joint(root, f'{child}_0rz', orig_xyz=[0, 0, 0], axis_xyz=[0, 0, dir_x_axis],
                      parent_link_name=f'{child}_ry', child_link_name=f'{child}_rz',
                      limits=pose_limit[1])
        else:
            # There are three joints for each link except for fixed link and tip
            # Three joints are for x, y and z direction
            # There is a fixed joint for fixed link and for tip
            relative_vect = joints_dict[child] - joints_dict[parent]
            parent_link_name = f'{parent}_z' if not child[-1] == '1' else f'{child[:-1]}_fixed'
            add_joint(root, f'{child}_x', orig_xyz=relative_vect, axis_xyz=[1, 0, 0],
                      parent_link_name=parent_link_name, child_link_name=f'{child}',
                      limits=[pose_limit[0][0],pose_limit[1][0]])
            add_joint(root, f'{child}_y', orig_xyz=[0, 0, 0], axis_xyz=[0, dir_x_axis, 0],
                      parent_link_name=f'{child}', child_link_name=y_link_name,
                      limits=[pose_limit[0][1],pose_limit[1][1]])
            # add main link (contains collision, ...) to z joint
            add_joint(root, f'{child}_z', orig_xyz=[0, 0, 0], axis_xyz=[0, 0, dir_x_axis],
                      parent_link_name=y_link_name, child_link_name=z_link_name,
                      limits=[pose_limit[0][2],pose_limit[1][2]])

        if child[-1] == '3':
            tip_link = add_link_element(root, f'{child[:-1]}_tip')
            add_inertial(tip_link, rpy=[0, 0, 0], xyz=[0, 0, 0], mass=0, 
                         inertia=[0, 0, 0, 0, 0, 0])
            # fixed joint
            origin_tip = joints_dict[child[:-1]] - joints_dict[child]
            add_joint(root, f'{child[:-1]}_tip', orig_xyz=origin_tip, axis_xyz=[0, 0, 0],
                      parent_link_name=z_link_name, child_link_name=f'{child[:-1]}_tip',
                      joint_type='fixed')
    return root, joint_order


def create_urdf_obj(obj_name, mesh_data, pose_limits=None, fixed_base=False):
    root = ET.Element('robot', {'name': obj_name})
    material = ET.SubElement(root, 'material', {'name': 'obj_color'})
    color = ET.SubElement(material, 'color', {'rgba': '1.0 0.423529411765 0.0392156862745 1.0'})
    top = add_link_element(root, 'top')
    bottom = add_link_element(root, 'bottom')
    if fixed_base:
        world = add_link_element(root, 'world')
        add_joint(root, 'world_to_base', orig_xyz=[0, 0, 0], axis_xyz=[0, 0, 0], 
              parent_link_name='world', child_link_name='bottom', 
              limits=None, joint_type='fixed')
    add_inertial(top, 
             rpy=mesh_data[0]['origin']['rpy'],
             xyz=mesh_data[0]['origin']['xyz'],
             mass=mesh_data[0]['mass']['value'],
             inertia=list(mesh_data[0]['inertia'].values()))
    add_inertial(bottom, 
                 rpy=mesh_data[1]['origin']['rpy'],
                 xyz=mesh_data[1]['origin']['xyz'],
                 mass=mesh_data[1]['mass']['value'],
                 inertia=list(mesh_data[1]['inertia'].values()))
    add_visual(top, mesh_fname=mesh_data[0]['path']+".stl",material_name='obj_color')
    add_visual(bottom, mesh_fname=mesh_data[1]['path']+".stl", material_name='obj_color')
    add_collision_mesh(top, mesh_fname=mesh_data[0]['path']+".obj")
    add_collision_mesh(bottom, mesh_fname=mesh_data[1]['path']+".obj")
    if pose_limits is None:
        pose_limits = [0, math.pi]
    add_joint(root, 'rotation', orig_xyz=[0, 0, 0], axis_xyz=mesh_data[0]['z_axis'],
              parent_link_name=bottom.attrib['name'], child_link_name=top.attrib['name'],
              limits=pose_limits, is_obj=True)

    return root

def create_urdf_obj_new(obj_name, mesh_data, pose_limits=None, fixed_base=False, no_bottom=True):
    root = ET.Element('robot', {'name': obj_name})
    material = ET.SubElement(root, 'material', {'name': 'obj_color'})
    color = ET.SubElement(material, 'color', {'rgba': '1.0 0.423529411765 0.0392156862745 1.0'})
    top = add_link_element(root, 'top')
    bottom = add_link_element(root, 'bottom')
    if fixed_base:
        world = add_link_element(root, 'world')
        add_joint(root, 'world_to_base', orig_xyz=[0, 0, 0], axis_xyz=[0, 0, 0],
                  parent_link_name='world', child_link_name='bottom',
                  limits=None, joint_type='fixed')
    add_inertial(top,
                 rpy=mesh_data[0]['origin']['rpy'],
                 xyz=mesh_data[0]['origin']['xyz'],
                 mass=mesh_data[0]['mass']['value'],
                 inertia=list(mesh_data[0]['inertia'].values()))
    add_inertial(bottom,
                 rpy=mesh_data[1]['origin']['rpy'],
                 xyz=mesh_data[1]['origin']['xyz'],
                 mass=mesh_data[1]['mass']['value'],
                 inertia=list(mesh_data[1]['inertia'].values()))
    add_visual(top, mesh_fname=mesh_data[0]['path']+".stl",material_name='obj_color')
    add_visual(bottom, mesh_fname=mesh_data[1]['path']+".stl", material_name='obj_color')
    add_collision_mesh(top, mesh_fname=mesh_data[0]['path']+".obj")
    if not no_bottom:
        add_collision_mesh(bottom, mesh_fname=mesh_data[1]['path']+".obj")
    if pose_limits is None:
        pose_limits = [0, math.pi]
    add_joint(root, 'rotation', orig_xyz=[0, 0, 0], axis_xyz=mesh_data[0]['z_axis'],
              parent_link_name=bottom.attrib['name'], child_link_name=top.attrib['name'],
              limits=pose_limits, is_obj=True)

    return root

def create_urdf_obj_free(obj_name, mesh_data, pose_limits=None, fixed_base=False, no_bottom=True):
    root = ET.Element('robot', {'name': obj_name})
    material = ET.SubElement(root, 'material', {'name': 'obj_color'})
    color = ET.SubElement(material, 'color', {'rgba': '1.0 0.423529411765 0.0392156862745 1.0'})
    top = add_link_element(root, 'top')
    bottom = add_link_element(root, 'bottom')
    world = add_link_element(root, 'world')
    add_joint(root, 'world_to_base', orig_xyz=[0, 0, 0], axis_xyz=[0, 0, 0],
              parent_link_name='world', child_link_name='bottom',
              limits=None, joint_type='fixed')

    add_inertial(top,
                 rpy=mesh_data[0]['origin']['rpy'],
                 xyz=mesh_data[0]['origin']['xyz'],
                 mass=mesh_data[0]['mass']['value'],
                 inertia=list(mesh_data[0]['inertia'].values()))
    add_inertial(bottom,
                 rpy=mesh_data[1]['origin']['rpy'],
                 xyz=mesh_data[1]['origin']['xyz'],
                 mass=mesh_data[1]['mass']['value'],
                 inertia=list(mesh_data[1]['inertia'].values()))
    add_visual(top, mesh_fname=mesh_data[0]['path']+".stl",material_name='obj_color')
    add_visual(bottom, mesh_fname=mesh_data[1]['path']+".stl", material_name='obj_color')
    add_collision_mesh(top, mesh_fname=mesh_data[0]['path']+".obj")
    if not no_bottom:
        add_collision_mesh(bottom, mesh_fname=mesh_data[1]['path']+".obj")
    if pose_limits is None:
        pose_limits = [0, math.pi]

    if not fixed_base:
        pose_limit = [[-0.8, 0.8],
                      [-6.28, 6.28]]
        y_name = 'fake_y'
        y_link = add_link_element(root, y_name)
        add_inertial(y_link, rpy=[0, 0, 0], xyz=[0, 0, 0],
                     mass=0.0, inertia=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        z_name = 'fake_z'
        z_link = add_link_element(root, z_name)
        add_inertial(z_link, rpy=[0, 0, 0], xyz=[0, 0, 0],
                     mass=0.0, inertia=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        rx_name = 'fake_rx'
        rx_link = add_link_element(root, rx_name)
        add_inertial(rx_link, rpy=[0, 0, 0], xyz=[0, 0, 0],
                     mass=0.0, inertia=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        ry_name = 'fake_ry'
        ry_link = add_link_element(root, ry_name)
        add_inertial(ry_link, rpy=[0, 0, 0], xyz=[0, 0, 0],
                     mass=0.0, inertia=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        r_name = 'fake'
        r_link = add_link_element(root, r_name)
        add_inertial(r_link, rpy=[0, 0, 0], xyz=[0, 0, 0],
                     mass=0.0, inertia=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        add_joint(root, 'fake_0x', orig_xyz=[0, 0, 0], axis_xyz=[1, 0, 0],
                  parent_link_name=bottom.attrib['name'], child_link_name='fake',
                  effort='100', limits=pose_limit[0], joint_type='prismatic')
        add_joint(root, 'fake_0y', orig_xyz=[0, 0, 0], axis_xyz=[0, 1, 0],
                  parent_link_name='fake', child_link_name='fake_y',
                  effort='100', limits=pose_limit[0], joint_type='prismatic')
        add_joint(root, 'fake_0z', orig_xyz=[0, 0, 0], axis_xyz=[0, 0, 1],
                  parent_link_name='fake_y', child_link_name='fake_z',
                  effort='100', limits=pose_limit[0], joint_type='prismatic')
        add_joint(root, 'fake_0rx', orig_xyz=[0, 0, 0], axis_xyz=[1, 0, 0],
                  parent_link_name='fake_z', child_link_name='fake_rx',
                  limits=pose_limit[1])
        add_joint(root, 'fake_0ry', orig_xyz=[0, 0, 0], axis_xyz=[0, 1, 0],
                  parent_link_name='fake_rx', child_link_name='fake_ry',
                  limits=pose_limit[1])
        add_joint(root, 'fake_0rz', orig_xyz=[0, 0, 0], axis_xyz=[0, 0, 1],
                  parent_link_name='fake_ry', child_link_name=top.attrib['name'],
                  limits=pose_limit[1])
    else:
        add_joint(root, 'rotation', orig_xyz=[0, 0, 0], axis_xyz=mesh_data[0]['z_axis'],
                  parent_link_name=bottom.attrib['name'], child_link_name=top.attrib['name'],
                  limits=pose_limits, is_obj=True)

    return root


def write_urdf_file(root, out_path, file_name):
    fname = file_name + '.urdf'
    out_path_file = os.path.abspath(os.path.join(out_path, fname))

    # pretty print
    xml = minidom.parseString(ET.tostring(root))
    xml = xml.toprettyxml(indent="   ")
    with open(out_path_file, "w") as f:
        f.write(xml)

    print(f"Saved to {out_path_file}")


def export_mano2urdf(hand_name, out_path, joints_dict, joint_mesh_data=None, is_rhand=True,
                      pose_limits=None):
    root, joint_order = create_urdf_tree(hand_name=hand_name, joints_dict=joints_dict,
                                         joint_mesh_data=joint_mesh_data, is_rhand=is_rhand,
                                         pose_limits=pose_limits)

    write_urdf_file(root=root, out_path=out_path, file_name=hand_name)

    out_path_file = os.path.abspath(os.path.join(out_path, hand_name + '_info.json'))
    with open(out_path_file, 'w') as f:
        json.dump({"joint_order": joint_order}, f, indent=2)


def export_obj2urdf(obj_name, model_path, out_path, mesh_data, pose_limits=None, fixed_base=False):
    root = create_urdf_obj(obj_name=obj_name, mesh_data=mesh_data, pose_limits=pose_limits, fixed_base=fixed_base)
    if fixed_base:
        obj_name += "_fixed_base"
    write_urdf_file(root=root, out_path=out_path, file_name=obj_name)
