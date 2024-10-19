import os.path
import json
import shutil

import numpy as np
import trimesh
import trimesh.exchange.obj
from pymeshlab.pmeshlab import MeshSet, Mesh
from scipy.spatial.transform import Rotation as R
from trimesh import Trimesh, convex, creation


def process_with_pymeshlab(vertices, density):
    m = Mesh(vertex_matrix=vertices)
    ms = MeshSet()
    ms.add_mesh(m)

    # compute the convex hull
    ms.generate_convex_hull()

    gm = ms.get_geometric_measures()

    volume = gm['mesh_volume']
    mass = density * volume

    center_of_mass = gm['center_of_mass']
    axis_momenta = gm['axis_momenta'] * density  # scale inertia with density

    inertia_matrix = gm['inertia_tensor']
    inertia_matrix = inertia_matrix * density  # scale inertia with density

    # boundingbox = ms.current_mesh().bounding_box()
    # diag = boundingbox.diagonal()
    # dim_x = boundingbox.dim_x()
    # dim_y = boundingbox.dim_y()

    joint_data = to_joint_data_dict(axis_momenta=axis_momenta, center_of_mass=center_of_mass,
                                    inertia_matrix=inertia_matrix, mass=mass)

    return ms, joint_data, mass, volume


def process_mesh_with_trimesh(verts, density, show=False):
    hull_mesh = trimesh.convex.convex_hull(verts, qhull_options='QbB Pp Qt')
    hull_mesh.density = density  # set density -> updates mass, inertia, ...

    # hull_mesh.mass_properties
    volume = hull_mesh.volume
    mass = hull_mesh.mass

    center_of_mass = hull_mesh.center_mass

    axes_inertia = hull_mesh.principal_inertia_components
    inertia_matrix = hull_mesh.moment_inertia

    # sanity check that the center of mass shifts with the centroid
    inert_shifted = trimesh.convex.convex_hull(hull_mesh.vertices - hull_mesh.centroid, qhull_options='QbB Pp Qt')
    inert_shifted.density = density
    cm = inert_shifted.center_mass
    assert (np.allclose(cm + hull_mesh.centroid, center_of_mass))

    # get bounding cylinder data
    bounding_primitive = hull_mesh.bounding_cylinder  # .bounding_box_oriented.bounding_cylinder
    direction = bounding_primitive.direction
    radius = bounding_primitive._kwargs['radius']
    height = bounding_primitive._kwargs['height']
    transform = bounding_primitive.primitive.transform

    # min_cylinder = trimesh.bounds.minimum_cylinder(hull_mesh, sample_count=10)
    # radius_, height_ = min_cylinder['radius'], min_cylinder['height']
    # min_cylinder_inst = trimesh.primitives.Cylinder(radius=radius_, height=height_, transform=min_cylinder['transform'])
    # assert (radius_ == radius)
    # assert (height_ == height)

    if show:
        capsule = trimesh.creation.capsule(radius=radius, height=height - 2 * radius)
        capsule = capsule.apply_transform(transform)
        capsule.visual.face_colors = capsule.visual.face_colors - 100
        (capsule + bounding_primitive).show()
        (hull_mesh + bounding_primitive).show()  # w on keyboard to show wireframe

    joint_data = to_joint_data_dict(axis_momenta=axes_inertia, center_of_mass=center_of_mass,
                                    inertia_matrix=inertia_matrix, mass=mass)

    # append cylinder bounding box to output
    joint_data['bounding_cylinder'] = dict({
        'direction': direction,
        'rotation': R.from_matrix(transform[:3, :3]).as_euler('XYZ'),
        'translation': transform[:3, 3],
        'centroid': hull_mesh.bounding_cylinder.centroid,
        'radius': radius,
        'height': height
    })

    bbox = hull_mesh.bounding_box
    joint_data['bounding_box'] = dict({
        'size': bbox.extents
    })

    return hull_mesh, joint_data, mass, volume


def _save_mesh(mesh, out_path: str, fname: str):
    out_path_file = os.path.join(out_path, fname)
    if isinstance(mesh, MeshSet):
        if 'stl' in fname:
            mesh.save_current_mesh(out_path_file, save_face_color=False, colormode=False)
        else:
            mesh.save_current_mesh(out_path_file, save_face_color=False)
    elif isinstance(mesh, Trimesh):
        if 'stl' in fname:
            mesh.export(out_path_file)
        else:
            vn = mesh.vertex_normals  # dummy call to compute vertex normals
            mesh.export(out_path_file, include_color=False)
    else:
        raise ValueError('Mesh type not supported.')

    print(f"Saved body part to {out_path_file}.")


def to_joint_data_dict(axis_momenta, center_of_mass, mass, inertia_matrix):
    joint_data = dict()
    joint_data['origin'] = {'rpy': axis_momenta,
                            'xyz': center_of_mass}
    joint_data['mass'] = {'value': mass}
    joint_data['inertia'] = {'ixx': inertia_matrix[0][0], 'ixy': inertia_matrix[0][1], 'ixz': inertia_matrix[0][2],
                             'iyy': inertia_matrix[1][1], 'iyz': inertia_matrix[1][2], 'izz': inertia_matrix[2][2]}

    return joint_data


def export_body_part_meshes(out_path, lbs_weight_matrix, vertices, joint_names,
                            decimation_factor_obj=1.0,
                            density=980,  # kg/m^3
                            is_rhand=True):
    out_path = os.path.abspath(os.path.join(out_path, 'meshes'))

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    argmax_lbs_matrix = np.argmax(lbs_weight_matrix, axis=-1)

    data = dict()

    total_mass = 0.0
    total_volume = 0.0
    joint_name2idx = dict((name, i) for i, name in enumerate(joint_names))
    for idx, name in enumerate(joint_names):
        argmax_idxs = np.empty(0, dtype=np.int)

        # get vertices belonging to joint i
        argmax_idxs = np.append(argmax_idxs, np.where(argmax_lbs_matrix == idx)[0])

        # process mesh for joints
        ms, joint_data, mass, volume = process_mesh_with_trimesh(vertices[argmax_idxs], density=density)

        # ms, joint_data, mass, volume = process_with_pymeshlab(vertices[argmax_idxs], density=density)

        total_mass += mass
        total_volume += volume

        data[name] = joint_data  # add to output dict

        # save the mesh for each joint
        fname = name.lower() + '.stl'
        _save_mesh(ms, out_path, fname)

        # decimate the mesh # TODO decimate mesh before or after computing geometric measures like mass, inertia, ...
        if decimation_factor_obj != 1.0:
            # pymeshlab
            # n_faces = ms.current_mesh().face_matrix().shape[0]
            # ms.meshing_decimation_quadric_edge_collapse(targetfacenum=int(n_faces * decimation_factor))

            ms = ms.simplify_quadratic_decimation(
                face_count=ms.area_faces.size * decimation_factor_obj)

        fname = name.lower() + '.obj'
        _save_mesh(ms, out_path, fname)

    print(f"The total volume is {total_volume} m^3 and total mass {total_mass} kg/m^3")

    return data


def get_obj_data(model_path, out_path, density=980, no_bottom=True, decimated=True): # density in kg/m^3
    # model_path_top = os.path.join(model_path, 'convex_top.obj')
    # model_path_bottom = os.path.join(model_path, 'convex_bottom.obj')
    model_path_top = os.path.join(model_path, 'top.obj')
    model_path_bottom = os.path.join(model_path, 'bottom.obj')

    assert os.path.exists(model_path_top), f"Not found: {model_path_top}"
    assert os.path.exists(model_path_bottom), f"Not found: {model_path_bottom}"

    mesh_t = trimesh.exchange.load.load_mesh(model_path_top, process=False)
    mesh_b = trimesh.exchange.load.load_mesh(model_path_bottom, process=False)

    vertices_t = mesh_t.vertices
    vertices_b = mesh_b.vertices
    faces_t = mesh_t.faces
    faces_b = mesh_b.faces

    mesh_t_meter = trimesh.Trimesh(vertices=vertices_t, faces=faces_t)
    mesh_b_meter = trimesh.Trimesh(vertices=vertices_b, faces=faces_b)

    mesh_t_meter.density = density * 2
    mesh_b_meter.density = density

    mass_t = mesh_t_meter.mass
    mass_b = mesh_b_meter.mass
    center_of_mass_t = mesh_t_meter.center_mass # / 1000 # mm -> m
    center_of_mass_b = mesh_b_meter.center_mass # / 1000
    axes_inertia_t = mesh_t_meter.principal_inertia_components
    inertia_matrix_t = mesh_t_meter.moment_inertia # / 1e6 # kg*mm^2 -> kg*m^2
    axes_inertia_b = mesh_b_meter.principal_inertia_components
    inertia_matrix_b = mesh_b_meter.moment_inertia # /1e6

    t_data = to_joint_data_dict(axis_momenta=axes_inertia_t, center_of_mass=center_of_mass_t,
                                inertia_matrix=inertia_matrix_t, mass=mass_t)
    b_data = to_joint_data_dict(axis_momenta=axes_inertia_b, center_of_mass=center_of_mass_b,
                                inertia_matrix=inertia_matrix_b, mass=mass_b)

    t_data['z_axis'] = [0, 0, -1]
    b_data['z_axis'] = [0, 0, -1]

    name_t = "top_watertight_tiny"
    name_b = "bottom_watertight_tiny"

    t_data['path'] = name_t
    b_data['path'] = name_b

    mesh_data = [t_data, b_data]


    if decimated:
        model_path_top_visual = os.path.join(model_path, 'decimated_top.obj')
        model_path_bottom_visual = os.path.join(model_path, 'decimated_bottom.obj')
    else:
        model_path_top_visual = os.path.join(model_path, 'top.obj')
        model_path_bottom_visual = os.path.join(model_path, 'bottom.obj')

    mesh_t_visual = trimesh.exchange.load.load_mesh(model_path_top_visual, process=False)
    mesh_b_visual = trimesh.exchange.load.load_mesh(model_path_bottom_visual, process=False)

    vertices_t_visual = mesh_t_visual.vertices
    vertices_b_visual = mesh_b_visual.vertices
    faces_t_visual = mesh_t_visual.faces
    faces_b_visual = mesh_b_visual.faces

    mesh_t_meter_visual = trimesh.Trimesh(vertices=vertices_t_visual, faces=faces_t_visual)
    mesh_b_meter_visual = trimesh.Trimesh(vertices=vertices_b_visual, faces=faces_b_visual)

    mesh_t_meter_visual.density = density
    mesh_b_meter_visual.density = density

    _save_mesh(mesh_t_meter_visual, out_path, name_t + ".obj")
    _save_mesh(mesh_t_meter_visual, out_path, name_t + ".stl")
    _save_mesh(mesh_b_meter_visual, out_path, name_b + ".obj")
    _save_mesh(mesh_b_meter_visual, out_path, name_b + ".stl")

    # return mesh_data, params
    return mesh_data