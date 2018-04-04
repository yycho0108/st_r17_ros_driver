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
