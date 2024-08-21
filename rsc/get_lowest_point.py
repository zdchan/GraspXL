import os
import trimesh
import numpy as np

def sample_surface_points(obj_path, num_samples=10000):
    # Load the OBJ file using trimesh
    mesh = trimesh.load(obj_path)

    # Sample points on the surface of the mesh
    points, face_indices = trimesh.sample.sample_surface(mesh, num_samples)

    # Get the minimum z-coordinate (lowest point) among the sampled points
    min_z_s = np.min(points[:, 2])
    
    min_z_v = mesh.bounds[0][2]
    
    min_z = min(min_z_s, min_z_v)

    return min_z

def process_objects(root_folder):
    # Iterate through folders in the root directory
    for folder_name in os.listdir(root_folder):
        folder_path = os.path.join(root_folder, folder_name)

        # Check if the path is a directory
        if os.path.isdir(folder_path):
            # Construct paths for the top and bottom OBJ files
            top_obj_path = os.path.join(folder_path, 'top_watertight_tiny.obj')
            bottom_obj_path = os.path.join(folder_path, 'bottom_watertight_tiny.obj')

            # Check if both files exist
            if os.path.exists(top_obj_path) and os.path.exists(bottom_obj_path):
                # Get the lowest points for both top and bottom OBJ files
                lowest_point_top = sample_surface_points(top_obj_path)
                lowest_point_bottom = sample_surface_points(bottom_obj_path)

                # Find the overall lowest point between the two files
                overall_lowest_point = min(lowest_point_top, lowest_point_bottom)

                # Create a text file in the same folder as the OBJ files
                txt_file_path = os.path.join(folder_path, 'lowest_point.txt')

                # Delete the text file if it already exists
                if os.path.exists(txt_file_path):
                    os.remove(txt_file_path)
                txt_file_path = os.path.join(folder_path, 'lowest_point_new.txt')
                # Write to the text file
                with open(txt_file_path, 'w') as txt_file:
                    txt_file.write(f"{overall_lowest_point}\n")

# Replace 'path/to/mixed_unseen_test' with the actual path to your root folder
root_folder_path = 'ycb_urdf/'
process_objects(root_folder_path)

