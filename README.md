# GraspXL: Generating Grasping Motions for Diverse Objects at Scale

## [Paper](https://arxiv.org/pdf/2403.19649.pdf) | [Project Page](https://eth-ait.github.io/graspxl/) | [Video](https://youtu.be/0-dRbxmX2PI)

<img src="/tease_more.jpg" /> 


The code and generated grasping motions of 500k+ objects with different hands will be released soon.

**If you are interested in the code and data and want to get the notification at the first time, you can fill [this form](https://forms.gle/NyU8Gk83M1itVvBQA). We will keep you updated**

## Abstract
Human hands possess the dexterity to interact with diverse objects such as grasping specific parts of the objects and/or approaching them from desired directions. More importantly, humans can grasp objects of any shape without object-specific skills. Recent works synthesize grasping motions following single objectives such as a desired approach heading direction or a grasping area. Moreover, they usually rely on expensive 3D hand-object data during training and inference, which limits their capability to synthesize grasping motions for unseen objects at scale. In this paper, we unify the generation of hand-object grasping motions across multiple motion objectives, diverse object shapes and dexterous hand morphologies in a policy learning framework GraspXL. The objectives are composed of the graspable area, heading direction during approach, wrist rotation, and hand position. Without requiring any 3D hand-object interaction data, our policy trained with 58 objects can robustly synthesize diverse grasping motions for more than 500k unseen objects with a success rate of 82.2%. At the same time, the policy adheres to objectives, which enables the generation of diverse grasps per object. Moreover, we show that our framework can be deployed to different dexterous hands and work with reconstructed or generated objects. We quantitatively and qualitatively evaluate our method to show the efficacy of our approach. Our model and code will be available.

## BibTeX Citation
```
@inProceedings{zhang2024graspxl,
  title={{GraspXL}: Generating Grasping Motions for Diverse Objects at Scale},
  author={Zhang, Hui and Christen, Sammy and Fan, Zicong and Hilliges, Otmar and Song, Jie},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2024}
}
```
