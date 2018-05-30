# ST R17 Calibration

![Gazebo Sim](figs/sim.gif)

DH Parameter Calibration with April Tags.

Currently, the seeds are configured for the ST R17 Arm, but the implementation is universal for all robot arms.

## Results

### Gazebo simulation

[Youtube Link](https://youtu.be/T6hWMyOZmXE)

![gz\_dh\_err](figs/gz_dh_err.png)

#### DH Parameters

Nominal:

&alpha;  | a        |d         | &Delta;q |
:-------:|:--------:|:--------:|:--------:|
&pi;     | 0        | -0.355   | 0        |
&pi;/2   | 0        | 0        | -&pi;/2  |
0        | 0.375    | 0        | 0        |
0        | 0.375    | 0.024    | &pi;/2   |
&pi;/2   | 0        | 0.042    | 1.176    |
0        | -0.012   | 0.159    | &pi;     |

Initial:

&alpha;  | a        |d         | &Delta;q |
:-------:|:--------:|:--------:|:--------:|
3.06     | 5.13e-2  | -3.51e-1 | -1.76e-2 |
1.68     | -1.34e-2 | 7.94e-3  | -1.61    |
-2.72e-2 | 3.74e-1  | 3.72e-2  | 2.53e-2  |
4.51e-2  | 3.83e-1  | 2.02e-2  | 1.57     |
1.53     | 1.81e-3  | 9.19e-2  | 1.15     |
-3.55e-2 | -2.13e-2 | 1.46e-1  | 3.17     |

Calibrated:

&alpha;  | a        |d         | &Delta;q |
:-------:|:--------:|:--------:|:--------:|
3.14     | 2.85e-3  | -3.66e-1 | 2.53e-3  |
1.57     | -2.64e-3 | -1.96e-2 | -1.60    |
-4.00e-3 | 3.79e-1  | 2.39e-2  | 1.79e-2  |
-2.27e-3 | 3.81e-1  | 2.21e-2  | 1.58     |
1.57     | 1.77e-2  | 7.99e-02 | 1.16     |
-5.80e-4 | 1.88e-3  | 1.11e-1  | 3.17     |

#### Kinematic Error

![pos\_err.png](figs/pos_err.png)

![kin\_err.png](figs/kin_err.png)

Mean Absolute Error over 1000 Samples:

x (m)  | y (m) | z(m)  |R (deg)|P (deg)|Y (deg)|
:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
1.29e-2|1.21e-2|1.73e-2|3.10e-1|2.58e-1|4.54e-1|


## Run DH Calibration (Virtual)

![calib\_virtual.gif](figs/calib_virtual.gif)

[Youtube Link](https://www.youtube.com/watch?v=DozXbHvRHp8)

1. Run the perception stack for detecting target markers.

    The below node publishes a virtual target at a fixed offset from `base_link`:
    
    ```bash
    roscore
    roslaunch st_r17_description urdf.launch use_kinect:=false
    rosrun st_r17_calibration target_publisher.py _num_markers:=4 _zero:=false _rate:=100
    rviz -d $(rospack find st_r17_description)/rviz/dh.rviz
    ```

2. Run the Calibrator.
    ```bash
    rosrun st_r17_calibration dh_calibrator.py _num_markers:=4 _noise:=False
    ```

## Run DH Calibration (Gazebo + Vision):

![calib.gif](figs/calib.gif)

1. Setup the World:

    ```bash
    roscore
    roslaunch st_r17_gazebo gazebo.launch
    roslaunch st_r17_gazebo spawn_tags.launch tag_size:=0.5
    rosrun st_r17_calibration gazebo_target_initializer _num_markers:=4 tag_size:=0.5 _min_Y:=-1.0 _max_Y:=1.0
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
    roslaunch st_r17_calibration calibrate.launch num_markers:=4 slop:=0.01
    roslaunch st_r17_calibration scouter.launch
    ```

4. Evaluate Performance:

    ```bash
    rosrun st_r17_calibration evaluate_dh.py
    ```
