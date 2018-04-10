# ST-R17 ROS Driver

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

### Simulation Demo

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
roslaunch st_r17_moveit_config real.launch 
roslaunch st_r17_moveit_config move_group_interface.launch 

# transforms ...
rosrun tf static_transform_publisher 0 0.0315 0 0 0 0 jaw_link camera_ros 100
rosrun tf static_transform_publisher -1 0 0 -1.57 0 -1.57 camera_ros camera 100

# stereo ...
ROS_NAMESPACE=stereo rosrun stereo_filter stereo_filter left:=left/image_raw right:=right/image_raw left_info:=left/camera_info right_info:=right/camera_info _baseline:=0.63
```
