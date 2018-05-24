# ST R17 Description

URDF Description for the ST R17 5DoF Robot Arm

## Run DH Calibration:

1. Run the perception stack for detecting target markers.

    The below node publishes a virtual target at a fixed offset from `base_link`:
    
    ```bash
    rosrun st_r17_description target_publisher.py _num_markers:=4 _zero:=false _rate:=100
    ```

2. Run the Calibrator.
    ```bash
    rosrun st_r17_description dh_calibrator.py _num_markers:=4 _noise:=False
    ```

