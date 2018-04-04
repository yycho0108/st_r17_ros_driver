# ST-R17 ROS Driver

### Generating URDF:

Refer to the [instructions]().
The CAD design files are available [here]().

### Generating IKFast:

([Reference](http://docs.ros.org/kinetic/api/moveit_tutorials/html/doc/ikfast_tutorial.html))

```bash
# format collada ...
openrave-robot.py ...

IKOUT=/tmp/ik.cpp
python $(openrave-config --python-dir)/openravepy/_openravepy_/ikfast.py --robot=$(rospack find st_r17_description)/urdf/robot_3.dae --iktype=translationdirection5d --baselink=0 --eelink=6 --savefile=${IKOUT}
```

### Simulation Dmo

```bash
roslaunch st_r17_moveit_config demo.launch 
```

### Hardware Demo

```bash
roslaunch st_r17_moveit_config hardware.launch
roslaunch st_r17_moveit_config real.launch
```
