# Deploy with ROS1 Noetic

This is for deploying the `uneven_humanoid` package on G1 with Jetson Orin.

## 📦 Installation and Configuration

**First**, install `unitree_sdk2` by following the instructions [here](https://github.com/unitreerobotics/unitree_sdk2). We are stongly recommend building and running the tests in sdk before deploying the policy.
   
**Secnd**, install ROS1 Noetic by following the instructions [here](https://wiki.ros.org/noetic/Installation/Ubuntu).

**Third**, install torch for Jetson by following the instructions [here](https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048).

**Finally**, install the required packages by running the following commands:

```
sudo apt update
```

## ⚙️ Building the ROS1 Package
```
git clone https://github.com/WongChoiLam/uneven_humanoid
cd uneven_humanoid/deploy/deloy_real_ros1
catkin_make
```

## 🚀 Running the Policy

```
source devel/setup.bash
roslaunch rl_controller controller.launch
```
