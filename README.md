# GraspXL: Generating Grasping Motions for Diverse Objects at Scale

## [Paper](https://arxiv.org/pdf/2403.19649.pdf) | [Project Page](https://eth-ait.github.io/graspxl/) | [Video](https://youtu.be/0-dRbxmX2PI)

<img src="/tease_more.jpg" /> 

### Contents

1. [News](#News)
2. [Dataset](#Dataset)
3. [Code](#Code)
4. [Installation](#installation)
5. [Demo](#Demo)
6. [Citation](#citation)
7. [License](#license)

## News
[2024.10] **We released the script to pre-process the new objects in [URDF_gen_from_obj](./raisimGymTorch/raisimGymTorch/helper/URDF_gen_from_obj). Put the .obj files you want to grasp (make sure they have meaningful sizes for grasping) under [temp](./raisimGymTorch/raisimGymTorch/helper/URDF_gen_from_obj/temp) and run [urdf_gen.py](./raisimGymTorch/raisimGymTorch/helper/URDF_gen_from_obj/urdf_gen.py), it will generate a folder with the proccessed objects and the urdf files in [rsc](./rsc), which you can further utilize to generate grasping motions with any environment scripts.**

[2024.08] **Data example & viewer released!**

[2024.08] **Code released!**

[2024.07] **The large-scale generated motions for 500k+ objects, each with diverse objectives and currently MANO and Allegro hand models, are ready to download! If you are interested, just fill [this form](https://forms.gle/dNwaGvtb4ppi1HZt5) to get access!**

**We will continuously enrich the dataset (e.g., motions generated with more hand models, more grasping motions generated with different objectives, etc) and keep you updated!**

[2024.03] **~~The code will be released soon.~~ Please fill out [this form](https://forms.gle/dNwaGvtb4ppi1HZt5) if you want to get the notification for any update!**



## Dataset
The dataset has been released, including the grasping motion sequences of different robot hands for 500k+ objects. Check [docs/DATASET.md](./docs/DATASET.md) for details and instructions. 

For an easier trial of the dataset, we give some examples (30 objects) of the data in the [dataset_example](./dataset_example) subfolder.

We also provide a viewer for the grasping motions. Check [GraspXL_visualization](https://github.com/zdchan/GraspXL_visualization) for more details.

**For texture**. We use decimated and texture-free Objaverse meshes in our dataset for smaller space consumption. However, the original Objaverse object ids are still included in the dataset (<object_id>). You can download the original Objaverse objects with textures according to their official download tutorial. The only thing to notice is the meshes we use in our dataset are scaled from original meshes, so you should calculate the scaling factor of each object by the bounding box size, and scale the downloaded original Objaverse mesh accordingly while keeping the textures. It should be quite convenient to be done with a Python script using trimesh or/and pymeshlab. After this, you can replace the objects in the dataset with the textured meshes for visualization.

**Note** The MANO hand poses in our dataset align with the original MANO model. Note that [manopth](https://github.com/hassony2/manopth) and [manotorch](https://github.com/lixiny/manotorch) have different joint orders. For more details, check [manotorch](https://github.com/lixiny/manotorch).



## Code

The repository comes with all the features of the [RaiSim](https://raisim.com/) physics simulation, as GraspXL is integrated into RaiSim.

The GraspXL related code can be found in the [raisimGymTorch](./raisimGymTorch) subfolder. There are 12 environments (see [envs](./raisimGymTorch/raisimGymTorch/env/envs/)) for Allegro Hand ("allegro\_"), Mano Hand ("ours\_") and Shadow Hand ("shadow\_"), 4 for each. "\_fixed" and "\_floating" represent the environments for the first and second training phase respectively. "\_test" represents the test environments and contain different test scripts for different test sets (PartNet, ShapeNet, Objaverse, and Generated/Reconstructed objects). "\_demo" represents the visualization environments which also record the generated motions.



## Installation


For good practice for Python package management, it is recommended to use virtual environments (e.g., `virtualenv` or `conda`) to ensure packages from different projects do not interfere with each other. The code is tested under Python 3.8.10.

### RaiSim setup

GraspXL is based on RaiSim simulation. For the installation of RaiSim, see and follow our documentation under [docs/INSTALLATION.md](./docs/INSTALLATION.md). Note that you need to get a valid, free license for the RaiSim physics simulation and an activation key via this [link](https://docs.google.com/forms/d/e/1FAIpQLSc1FjnRj4BV9xSTgrrRH-GMDsio_Um4DmD0Yt12MLNAFKm12Q/viewform). 

### GraspXL setup

After setting up RaiSim, the last part is to set up the GraspXL environments.

```
$ cd raisimGymTorch 
$ python setup.py develop
```

All the environments are run from this raisimGymTorch folder. 

Note that every time you change the environment.hpp, you need to run `python setup.py develop` again to build the environments.

Then install pytorch with (Check your CUDA version and make sure they match)

```
$ pip3 install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu118
```

Install required packages

```
$ pip install scipy
$ pip install scikit-learn scipy matplotlib
```

### Other alternative requirements

1. (Only for Mano policy training) GraspXL uses [manotorch](https://github.com/lixiny/manotorch) [Anatomy Loss](https://github.com/lixiny/manotorch#anatomy-loss) during the training (for Mano Hand only), so if you want to train Mano Hand policies (run [ours_fixed/runner.py](./raisimGymTorch/raisimGymTorch/env/envs/ours_fixed/runner.py) or [ours_floating/runner.py](./raisimGymTorch/raisimGymTorch/env/envs/ours_floating/runner.py)), you need to install manotorch. Please follow the official guideline in [manotorch](https://github.com/lixiny/manotorch).

​	After installation, replace the mano_assets_root in [mano_amano.py](https://github.com/zdchan/GraspXL/blob/1e239242082ec2bae9b9eddb4895f9f4f1d640af/raisimGymTorch/raisimGymTorch/helper/mano_amano.py#L10-L13) to your own path.

2. (Only for ShapeNet test set) If you want to generate motions for the objects from the ShapeNet test set, download [ShapeNet.zip](https://1drv.ms/u/s!ArIwHmrYW4HkoO0tm1D48rVudC4Bnw?e=DyEtsL), upzip and put the folder named large_scale_obj in [rsc](./rsc) (The original object meshes are from [ShapeNet](https://www.shapenet.org/))
3. (Only for 500k+ Objaverse test set) If you want to generate motions for the 500k+ objaverse objects, fill [this form](https://forms.gle/dNwaGvtb4ppi1HZt5) to get access to objaverse_urdf.zip. Unzip it and put the subset you want in [rsc](./rsc).

You should be all set now. Try to run the demo!



## Demo

We provide some pre-trained models to view the output of our method. They are stored in [this folder](./raisimGymTorch/data_all/). 

+ For interactive visualizations, you need to run

  ```Shell
  ./../raisimUnity/linux/raisimUnity.x86_64
  ```

  and check the Auto-connect option.

+ To randomly choose an object and visualize the generated sequences in simulation (use Mano Hand as an example), run

  ```Shell
  python raisimGymTorch/env/envs/ours_demo/demo.py
  ```

You can indicate the objects or the objectives of the generated motions in the visualization environments

+ The object is by default a random object from the training set, which you can change to a specified object. You can specify the object set by the variable cat_name (e.g., for [ours_demo](https://github.com/zdchan/GraspXL/blob/1e239242082ec2bae9b9eddb4895f9f4f1d640af/raisimGymTorch/raisimGymTorch/env/envs/ours_demo/demo.py#L76)), and choose a specific object by the variable obj_list (e.g., for [ours_demo](https://github.com/zdchan/GraspXL/blob/1e239242082ec2bae9b9eddb4895f9f4f1d640af/raisimGymTorch/raisimGymTorch/env/envs/ours_demo/demo.py#L90)). 

  The object sets include [mixed_train](./rsc/mixed_train) (the training set from [PartNet](https://partnet.cs.stanford.edu/)), [affordance_level](./rsc/affordance_level) (the PartNet test set), [large_scale_obj](./rsc/large_scale_obj) (the ShapeNet test set which you can download with [ShapeNet.zip](https://1drv.ms/u/s!ArIwHmrYW4HkoO0tm1D48rVudC4Bnw?e=DyEtsL)),  [YCB](./rsc/YCB) (reconstructed YCB objects), [gt](./rsc/gt) (groundtruth of the reconstructed YCB objects), [wild](./rsc/wild) (reconstructed in-the-wild objects), [gen](./rsc/gen) (objects generated with [DreamFusion](https://dreamfusion3d.github.io/))

+ The objectives are by default randomly sampled with the function get_initial_pose. You can also specify a desired objective with the function get_initial_pose_set.  [ours_demo](https://github.com/zdchan/GraspXL/blob/1e239242082ec2bae9b9eddb4895f9f4f1d640af/raisimGymTorch/raisimGymTorch/env/envs/ours_demo/demo.py#L198-L201) shows an example.



## BibTeX Citation

To cite us, please use the following:

```bibtex
@inProceedings{zhang2024graspxl,
  title={{GraspXL}: Generating Grasping Motions for Diverse Objects at Scale},
  author={Zhang, Hui and Christen, Sammy and Fan, Zicong and Hilliges, Otmar and Song, Jie},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2024}
}
```



## License

This work and the dataset are licensed under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).
