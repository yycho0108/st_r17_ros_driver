# ST-R17 ROS Driver

## Installation

1. Clone the Repositories:

    ```bash
    cd ~/catkin_ws/src
    git clone http://github.com/yycho0108/st_r17_ros_driver.git
    ```

2. Install the Dependencies:

    ```bash
    rosdep update
    rosdep install st_r17_ros_driver --ignore-src
    ```

3. Build the project:

    ```bash
    catkin build st_r17_ros_driver
    ```

## Documentation

### Generating URDF:

Refer to the [instructions]().
The CAD design files are available [here]().

### Generating IKFast:

([Reference](http://docs.ros.org/kinetic/api/moveit_tutorials/html/doc/ikfast_tutorial.html))

```bash
roscd st_r17_ikfast_plugin
# options : decimal rounding
./generate_ikfast_solver.sh
```

## Run

The following are the most basic demonstrations of functions.

See the included packages such as [st\_r17\_calibration](st_r17_calibration) or [st\_r17\_gazebo](st_r17_gazebo) for more involved demos.

### Rviz Simulation Demo

```bash
roslaunch st_r17_moveit_config demo.launch 
roslaunch st_r17_moveit_config move_group_interface.launch
```

### Hardware Demo

```bash
roslaunch st_r17_moveit_config hardware.launch
roslaunch st_r17_moveit_config demo.launch sim:=false
roslaunch st_r17_moveit_config move_group_interface.launch
```

### Stereo Vision Testing Configuration

```bash
roscore

# run hardware ...
roslaunch st_r17_moveit_config hardware.launch rate:=20.0
ROS_NAMESPACE=stereo rosrun uvc_camera uvc_stereo_node _left/device:=/dev/video1 _right/device:=/dev/video2

# configure camera ...
uvcdynctrl -v -d video1 --set='Focus, Auto' 0
uvcdynctrl -v -d video2 --set='Focus, Auto' 0
uvcdynctrl -v -d video1 --set='Focus (absolute)' 0
uvcdynctrl -v -d video2 --set='Focus (absolute)' 0

# moveit ...
roslaunch st_r17_moveit_config demo.launch sim:=false
roslaunch st_r17_moveit_config move_group_interface.launch 

# stereo ...
ROS_NAMESPACE=stereo rosrun stereo_filter stereo_filter left:=left/image_raw right:=right/image_raw left_info:=left/camera_info right_info:=right/camera_info _baseline:=0.63
```
