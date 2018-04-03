cmake_minimum_required(VERSION 2.8.3)
project(st_r17_moveit_config)

add_compile_options(-std=c++11)

find_package(Threads)
find_package(catkin REQUIRED
	roscpp
	control_msgs
	trajectory_msgs
	actionlib
	urdf
	joint_limits_interface
	transmission_interface
	hardware_interface
	controller_manager	
	moveit_ros_move_group
	moveit_ros_planning
	moveit_ros_planning_interface
	moveit_core
	pluginlib
	roscpp
	tf_conversions
	)

include_directories(
	include/
	${catkin_INCLUDE_DIRS}
	#${Gflags_INCLUDE_DIRS}
	)

catkin_package(
	LIBRARIES
	CATKIN_DEPENDS
	moveit_core
	pluginlib
	roscpp
	tf_conversions
	)

add_executable(st_r17_hardware_node src/st_r17_interface.cpp src/st_arm.cpp src/utils.cpp src/Mutex.cpp)
add_dependencies(st_r17_hardware_node ${catkin_EXPORTED_TARGETS})
add_executable(st_r17_move_group_interface src/move_group_interface.cpp)

target_link_libraries(st_r17_hardware_node
	${catkin_LIBRARIES}
	${CMAKE_THREAD_LIBS_INIT}
    serial
	)

target_link_libraries(st_r17_move_group_interface
	${catkin_LIBRARIES}
	)

#set(IKFAST_LIBRARY_NAME st_r17_arm_group_moveit_ikfast_plugin)
#find_package(LAPACK REQUIRED)
#add_library(${IKFAST_LIBRARY_NAME} src/st_r17_arm_group_ikfast_moveit_plugin.cpp)
#target_link_libraries(${IKFAST_LIBRARY_NAME} ${catkin_LIBRARIES} ${Boost_LIBRARIES} ${LAPACK_LIBRARIES})

install(DIRECTORY launch DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
	PATTERN "setup_assistant.launch" EXCLUDE)
install(DIRECTORY config DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})

#install(TARGETS ${IKFAST_LIBRARY_NAME} LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION})

#install(
#	FILES
#	st_r17_arm_group_moveit_ikfast_plugin_description.xml
#	DESTINATION
#	${CATKIN_PACKAGE_SHARE_DESTINATION}
#	)
