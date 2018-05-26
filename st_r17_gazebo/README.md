# ST R17 Gazebo Simulation

## Run

```bash
roscore
rosrun tf static_transform_publisher 0 0 0 0 0 0 odom base_link 1
roslaunch st_r17_gazebo gazebo.launch
roslaunch st_r17_description control.launch
roslaunch st_r17_moveit_config move_group.launch allow_trajectory_execution:=true fake_execution:=false info:=true debug:=false
roslaunch st_r17_moveit_config moveit_rviz.launch
```
