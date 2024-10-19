import os
import shutil
import argparse

import numpy as np

from export_meshes_shapenet import get_obj_data
from urdf_affordpose import create_urdf_obj, write_urdf_file, create_urdf_obj_new, create_urdf_obj_free


# Define source and destination folders
source_folder = "temp"
destination_parent_folder = f'formated_temp_obj'

# Create destination parent folder if it doesn't exist
if not os.path.exists(destination_parent_folder):
    os.makedirs(destination_parent_folder)

# Iterate through each .obj file in the source folder
for root, dirs, files in os.walk(source_folder):
    for filename in files:
        if filename.endswith('.obj'):
            # Construct source .obj file path
            obj_path = os.path.join(root, filename)

            # Extract folder name from the filename
            folder_name = root.split('/')[-1]

            # Construct destination folder path
            destination_folder = os.path.join(destination_parent_folder, folder_name, filename.split('.')[0])

            # Create destination folder if it doesn't exist
            if not os.path.exists(destination_folder):
                os.makedirs(destination_folder)

            # Destination file path with the new name (top.obj)
            top_obj_path = os.path.join(destination_folder, 'top.obj')

            # Copy .obj file to the destination folder and rename to top.obj
            shutil.copyfile(obj_path, top_obj_path)

            # Copy cube.obj to the destination folder and rename it to bottom.obj
            bottom_obj_path = os.path.join(destination_folder, 'bottom.obj')
            shutil.copyfile('cube.obj', bottom_obj_path)

            print(f"File 'cube.obj' copied and renamed to 'bottom.obj' in folder '{folder_name}' under '{destination_parent_folder}'")

input_directory = 'formated_temp_obj'
output_directory = 'formated_temp_obj_urdf/'

no_bottom = False
decimated = False

for root, dirs, files in os.walk(input_directory):
    for filename in files:
        if filename.endswith("top.obj"):

            obj_name = root.split('/')[-1]
            print(f"Processing {obj_name}")

            label_dir = os.path.join(output_directory + obj_name)
            os.makedirs(label_dir, exist_ok=True)
            mesh_data = get_obj_data(model_path=root, out_path=label_dir, decimated=decimated)

            root1 = create_urdf_obj_new(obj_name=obj_name, mesh_data=mesh_data, pose_limits=[0, 0.001], fixed_base=False, no_bottom=no_bottom)
            write_urdf_file(root=root1, out_path=label_dir, file_name=obj_name)

            root2 = create_urdf_obj_new(obj_name=obj_name, mesh_data=mesh_data, pose_limits=[0, 0.001], fixed_base=True, no_bottom=no_bottom)
            obj_name += "_fixed_base"
            write_urdf_file(root=root2, out_path=label_dir, file_name=obj_name)


shutil.move(output_directory, "./../../../../rsc/")
