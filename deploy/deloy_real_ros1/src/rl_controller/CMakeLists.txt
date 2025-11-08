cmake_minimum_required(VERSION 3.16)
project(rl_controller)

# set(CMAKE_CXX_STANDARD 17)
ADD_COMPILE_OPTIONS(-std=c++11 )
ADD_COMPILE_OPTIONS(-std=c++14 )

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")

# set(CMAKE_BUILD_TYPE "Debug")  
# set(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g -ggdb")

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  roslib
  robot_msgs
)

find_package(yaml-cpp REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(unitree_sdk2 REQUIRED)

set(CMAKE_PREFIX_PATH /home/unitree/miniconda3/envs/rl/lib/python3.8/site-packages/torch)
set(Torch_DIR /home/unitree/miniconda3/envs/rl/lib/python3.8/site-packages/torch)
find_package(Torch REQUIRED)

catkin_package(
 INCLUDE_DIRS include
#  LIBRARIES rl_controller
 CATKIN_DEPENDS roscpp roslib robot_msgs
)

include_directories(
  include
  SYSTEM
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${TORCH_INCLUDE_DIRS}
)

add_executable (controller src/controller.cpp )
target_link_libraries(controller PRIVATE
  ${catkin_LIBRARIES}
  unitree_sdk2 
  # ddsc 
  # ddscxx 
  ${TORCH_LIBRARIES} 
  yaml-cpp
  pthread
)
