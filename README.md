# GraspXL: Generating Grasping Motions for Diverse Objects at Scale

## [Paper](https://arxiv.org/pdf/2403.19649.pdf) | [Project Page](https://eth-ait.github.io/graspxl/) | [Video](https://youtu.be/z7axE9F7d6s)

<img src="/tease_more.jpg" /> 

## Update
**The large-scale generated motions for 500k+ objects, each with diverse objectives and currently MANO and Allegro hand models, are ready to download! If you are interested, just fill [this form](https://forms.gle/dNwaGvtb4ppi1HZt5) to get access!**

**The code will be released soon. We will also continuously enrich the dataset (e.g., motions generated with more hand models, more grasping motions generated with different objectives, etc) and keep you updated! Please also fill [this form](https://forms.gle/dNwaGvtb4ppi1HZt5) if you want to get the notification for any update!**

## Dataset
The dataset is composed of several .zip files, which contain the generated diverse grasping motion sequences for different hands on the [Objaverse](https://objaverse.allenai.org/), and the processed (scaled and decimated) object mesh files. To make the dataset easier to download, we split the generated motions into several .zip files so that users can choose which to download. The formats are like this :

```
object.zip
    ├── small
        ├── object_id
           ├── object_id.obj
        ...
    ├── medium
        ├── object_id
           ├── object_id.obj
        ...
    ├── large
        ├── object_id
           ├── object_id.obj
        ...
```
Small, medium, and large contain object meshes with different scales (Check our paper for more details) used by the recorded sequences. 
```
allegro_dataset_1.zip
    ├── small
        ├── object_id
           ├── allegro_1.npy
           ├── allegro_2.npy
           ├── allegro_3.npy
        ...
    ├── medium
        ├── object_id
           ├── allegro_1.npy
        ...
    ├── large
        ├── object_id
           ├── allegro_1.npy
        ...
```

Most objects have 5 sequences recorded. But not all.
Each .npy file contains the following info:
```
data = np.load("allegro_x.npy", allow_pickle=True).item()
data['right_hand']['trans']: a numpy array with the shape (frame_num, 3), which is the position sequence of the wrist.
data['right_hand']['rot']: a numpy array with the shape (frame_num, 3), which is the orientation (in axis angle) sequence of the wrist.
data['right_hand']['pose']: a numpy array with the shape (frame_num, 22), where the first 6 dimensions of each frame are 0, and the remaining 16 dimensions are the joint angles.

data['object_id']['trans']: a numpy array with the shape (frame_num, 3), which is the position sequence of the wrist.
data['object_id']['rot']: a numpy array with the shape (frame_num, 3), which is the orientation (in axis angle) sequence of the wrist.
data['object_id']['angle']: not used.
```

```
allegro_dataset_2.zip
```
Same as above.

```
mano_dataset_1.zip
    ├── small
        ├── object_id
           ├── mano_1.npy
           ├── mano_2.npy
           ├── mano_3.npy
        ...
    ├── medium
        ├── object_id
           ├── mano_1.npy
        ...
    ├── large
        ├── object_id
           ├── mano_1.npy
        ...
```

Most objects have 5 sequences recorded. But not all.
Each .npy file contains the following info:
```
data = np.load("mano_x.npy", allow_pickle=True).item()
data['right_hand']['trans']: a numpy array with the shape (frame_num, 3), which is the position sequence of the wrist.
data['right_hand']['rot']: a numpy array with the shape (frame_num, 3), which is the orientation (in axis angle) sequence of the wrist (the first 3 dimensions for MANO parameter).
data['right_hand']['pose']: a numpy array with the shape (frame_num, 45), which is the sequence of the remaining 45 dimensions of MANO parameter.

data['object_id']['trans']: a numpy array with the shape (frame_num, 3), which is the position sequence of the wrist.
data['object_id']['rot']: a numpy array with the shape (frame_num, 3), which is the orientation (in axis angle) sequence of the wrist.
data['object_id']['angle']: not used.
```
```
mano_dataset_2.zip
mano_dataset_3.zip
```
Same as above.

## BibTeX Citation
```
@inProceedings{zhang2024graspxl,
  title={{GraspXL}: Generating Grasping Motions for Diverse Objects at Scale},
  author={Zhang, Hui and Christen, Sammy and Fan, Zicong and Hilliges, Otmar and Song, Jie},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2024}
}
```
