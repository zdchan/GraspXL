## Why use GraspXL?

### Summary on the dataset:
- It contains 10M+ diverse grasping motions for 500k+ objects of different dexterous hands.
- All the grasping motions are generated with a physics simulation, which makes sure the physical plausibility of the generated motions.
- Each motion contains accurate object and hand poses of each frame.

### Potential tasks with GraspXL:
- Generating general grasping motions for [full-body motion generation](https://eth-ait.github.io/phys-fullbody-grasp/)
- Achieving zero-shot text-to-motion generation with off-the-shelf [text-to-mesh](https://dreamfusion3d.github.io/) generation methods
- Generating large scale of pseudo 3D RGBD grasping motions with [texture generation](https://mq-zhang1.github.io/HOIDiffusion/) methods and help with further downstream applications such as [training a general pose estimation model](https://nvlabs.github.io/FoundationPose/)
- [Simulating general human hand motions](https://eth-ait.github.io/synthetic-handovers/) for human-robot interaction
- Serving as expert demonstrations for robot imitation learning

## Dataset Instruction

The dataset is composed of several .zip files, which contain the generated diverse grasping motion sequences for different hands on the [Objaverse](https://objaverse.allenai.org/), and the processed (scaled and decimated) object mesh files. To make the dataset easier to download, we split the recorded motion sequences into several .zip files (5 sequences for most objects in each .zip file) so that users can choose which to download. The formats are like this :

### Objects
```
object_dataset.zip
    ├── small
        ├── <object_id>
           ├── <object_id>.obj
        ...
    ├── medium
        ├── <object_id>
           ├── <object_id>.obj
        ...
    ├── large
        ├── <object_id>
           ├── <object_id>.obj
        ...
```
Small, medium, and large contain object meshes with different scales (Check our paper for more details) used by the recorded sequences. 

### Allegro sequences
```
allegro_dataset_1.zip
    ├── small
        ├── <object_id>
           ├── allegro_1.npy
           ├── allegro_2.npy
           ├── allegro_3.npy
            ...
        ...
    ├── medium
        ├── <object_id>
           ├── allegro_1.npy
            ...
        ...
    ├── large
        ├── <object_id>
           ├── allegro_1.npy
            ...
        ...
```
Not every object has the same amount of sequences recorded.

Each .npy file contains a single motion sequence with the following format:
```
data = np.load("allegro_x.npy", allow_pickle=True).item()
data['right_hand']['trans']: a numpy array with the shape (frame_num, 3), which is the position sequence of the wrist.
data['right_hand']['rot']: a numpy array with the shape (frame_num, 3), which is the orientation (in axis angle) sequence of the wrist.
data['right_hand']['pose']: a numpy array with the shape (frame_num, 22), where the first 6 dimensions of each frame are 0, and the remaining 16 dimensions are the joint angles.

data['object_id']['trans']: a numpy array with the shape (frame_num, 3), which is the position sequence of the object.
data['object_id']['rot']: a numpy array with the shape (frame_num, 3), which is the orientation (in axis angle) sequence of the object.
data['object_id']['angle']: not used.
```

```
allegro_dataset_2.zip
```
Same format as above. Another group of recorded motion sequences. 

### MANO sequences
```
mano_dataset_1.zip
    ├── small
        ├── <object_id>
           ├── mano_1.npy
           ├── mano_2.npy
           ├── mano_3.npy
            ...
        ...
    ├── medium
        ├── <object_id>
           ├── mano_1.npy
            ...
        ...
    ├── large
        ├── <object_id>
           ├── mano_1.npy
            ...
        ...
```
Not every object has the same amount of sequences recorded.

Each .npy file contains a single motion sequence with the following format:
```
data = np.load("mano_x.npy", allow_pickle=True).item()
data['right_hand']['trans']: a numpy array with the shape (frame_num, 3), which is the position sequence of the wrist.
data['right_hand']['rot']: a numpy array with the shape (frame_num, 3), which is the orientation (in axis angle) sequence of the wrist (the first 3 dimensions for MANO parameter).
data['right_hand']['pose']: a numpy array with the shape (frame_num, 45), which is the sequence of the remaining 45 dimensions of MANO parameter.

data['object_id']['trans']: a numpy array with the shape (frame_num, 3), which is the position sequence of the object.
data['object_id']['rot']: a numpy array with the shape (frame_num, 3), which is the orientation (in axis angle) sequence of the object.
data['object_id']['angle']: not used.
```
```
mano_dataset_2.zip
mano_dataset_3.zip
```
Same as above. Another group of recorded motion sequences. 

### Additional data
``objaverse_urdf.zip`` contains the URDF files of the 500k+ objaverse objects, which are used by the code to generate the motions in simulation.

``leap_dataset_1.zip`` contains the LEAP hand grasping motions for a subset of the 500k+ objects. The dataset structure and data format are the same as the data for Allegro Hand, which is explained above.
