#!/usr/bin/env bash

OUTDIR="/tmp"
OUTCOL="${OUTDIR}/robot.dae"
IKOUT="${OUTDIR}/ik.cpp"
ROUND=8

pushd ${PWD}
source ~/.bashrc
roscd st_r17_description/urdf
rosrun collada_urdf urdf_to_collada robot_3.urdf ${OUTCOL}
rosrun moveit_kinematics round_collada_numbers.py ${OUTCOL} ${OUTCOL} ${ROUND}
python $(openrave-config --python-dir)/openravepy/_openravepy_/ikfast.py --robot=${OUTCOL} --iktype=translationdirection5d --baselink=0 --eelink=5 --savefile=${IKOUT}
popd
