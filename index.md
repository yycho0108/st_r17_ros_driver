# ST R17 ROS Driver

---

## Intro

This project is a C++ ROS driver for the ST-R17 5DoF Robot Arm from [ST Robotics](https://www.strobotics.com/).

The hardware for development are based on the Olin College Robo-Lab with Professor Dave Barrett.

With the above due credits, however, the works presented here are entirely of my own.

This project was initially my development under the Edwin, or now Interactive Robotics Lab(IRL), but ultimately branched out as an independent project.

## Content

The project includes the following packages, each of which implement a crucial feature in the development of any robotic arm platform:

- **st_r17_ros_driver** : `ros_control` compatible C++ ROS Driver for the ST-R17 Robot Arm through Serial protocol
- **st_r17_moveit_config** : `MoveIt!` Interface for dynamic trajectory generation and target-pose following
- **st_r17_description** : URDF (Universal Robot Description Format) of the ST-R17 Robot Arm for description of internal transformations, collisions, and dynamic parameters
- **st_r17_gazebo** : Gazebo simulation of the ST-R17 Robot Arm for SITL(Software-In-The-Loop) Development
- **st_r17_calibration** : DH Parameter Calibration
- **st_r17_ikfast_plugin** : Inverse Kinematics Plugin for efficient analytical solutions for joint-angle computation