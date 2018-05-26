# ST R17 Calibration

DH Parameter Calibration with April Tags

Currently, the seeds are configured for the ST R17 Arm, but the implementation is universal for all robot arms.

## Run DH Calibration (Virtual)

1. Run the perception stack for detecting target markers.

    The below node publishes a virtual target at a fixed offset from `base_link`:
    
    ```bash
    rosrun st_r17_calibration target_publisher.py _num_markers:=4 _zero:=false _rate:=100
    ```

2. Run the Calibrator.
    ```bash
    rosrun st_r17_calibration dh_calibrator.py _num_markers:=4 _noise:=False
    ```

## Run DH Calibration (Gazebo + Vision):

1. Setup the World:

    ```bash
    roscore
    roslaunch st_r17_gazebo gazebo.launch
    roslaunch st_r17_gazebo spawn_tags.launch
    rosrun st_r17_calibration gazebo_target_initializer.-y _num_markers:=4 tag_size:=0.5
    ```

2. Setup the Controls:

    ```bash
    roslaunch st_r17_description control.launch
    roslaunch st_r17_moveit_config move_group.launch allow_trajectory_execution:=true fake_execution:=false info:=true debug:=false
    roslaunch st_r17_moveit_config moveit_rviz.launch
    ```

3. Start perception stack and calibration:

    ```bash
    ROS_NAMESPACE=/left rosrun image_proc image_proc
    roslaunch st_r17_calibration apriltags.launch
    rosrun st_r17_calibration dh_calibrator.py _num_markers:=4 stereo_to_target:=/tag_detections
    ```
