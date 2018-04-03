### STR17 Moveit Config

## Install

1. Clone the Repositories:

```bash
cd ~/catkin_ws/src
git clone http://github.com/olinrobotics/st_r17_description.git
git clone http://github.com/olinrobotics/st_r17_moveit_config.git
```

2. Install the Dependencies:

```bash
rosdep update
rosdep install st_r17_description
rosdep install st_r17_moveit_config --ignore-src
```

3. Build the project:

```bash
catkin_make --pkg st_r17_description
catkin_make --pkg st_r17_moveit_config
```

### Running

If you have built the project immediately before this, be sure to run these in the new terminal.

#### Simulation

```bash
roslaunch st_r17_moveit_config demo.launch
```

#### With Hardware

```bash
roslaunch st_r17_moveit_config hardware.launch
roslaunch st_r17_moveit_config real.launch
```

### TO-DOS For Documentation:

- [ ] How to setup URDF Packages
- [ ] How to configure MoveIt! For a custom robot
	- [ ] Setting up the hardware interface
	- [ ] Setting up move\_group interface`
- [ ] How to develop custon analytical IK solutions with IKFast
