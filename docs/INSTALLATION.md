# INSTALLATION GUIDE

## General Info

This is an installation guide for Ubuntu. For Mac/Windows, please follow the official RaiSim [installation guide](https://raisim.com/sections/Installation.html).

## Physics Simulation Installation

Install prerequisites
```
$ sudo apt install cmake libeigen3-dev
```

Create the workspace in your desired directory:
```
$ mkdir raisim && cd raisim
$ mkdir raisim_build
```

Clone the GraspXL repository:

```
$ git clone https://github.com/zdchan/GraspXL.git
$ cd GraspXL
```

Build the simulation backbone:
```
$ mkdir build  && cd build
```
Run the following command and replace ```$LOCAL_INSTALL``` by the path to the raisim_build folder created above (e.g. ~/raisim/raisim_build) and ```$(python3 -c "import sys; print(sys.executable)")``` by the path to the python executable of your virtual environment:
```
$ cmake .. -DCMAKE_INSTALL_PREFIX=$LOCAL_INSTALL -DRAISIM_EXAMPLE=ON \ 
-DRAISIM_PY=ON \ 
-DPYTHON_EXECUTABLE=$(python3 -c "import sys; print(sys.executable)")
$ make install -j4
$ cd ..
```


Add the following lines to your ~/.bashrc file:
``` 
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:<PATH_TO_RAISIM_BUILD_FOLDER>/lib
export PYTHONPATH=$PYTHONPATH:<PATH_TO_RAISIM_BUILD_FOLDER>/lib
```
Do not use ```~``` to reference your home directory. Write the full path.

## Visualizations (Unity) Installation

First we need to install ```minizip``` and ```ffmpeg```:

```
$ sudo apt install minizip ffmpeg
```

Next, we need to install [Vulkan](https://linuxconfig.org/install-and-test-vulkan-on-linux):

```
$ sudo add-apt-repository ppa:oibaf/graphics-drivers
$ sudo apt update
$ sudo apt upgrade
$ sudo apt install libvulkan1 mesa-vulkan-drivers vulkan-utils
```

Make sure that you run raisimUnity executable in ```raisimUnity/<OS>/RaiSimUnity``` before you run the examples.

In Ubuntu 22.04, the following line should be executed to correctly link the package:
```
sudo ln -s /lib/x86_64-linux-gnu/libdl.so.2 /lib/x86_64-linux-gnu/libdl.so
```

## Activation Key

Rename the activation key that you received by email to activation.raisim. Save that file in ```/home/<YOUR-USERNAME>/.raisim```. 

RaiSim will check the path you set by ```raisim::World::setActivationKey()```. If the file is not found, it will search in the user directory, where you saved your activation.raisim file.
