#include <ros/ros.h>
#include <controller_manager/controller_manager.h>
#include <std_msgs/String.h>
#include <sensor_msgs/JointState.h>

#include "utils.h"
#include "st_r17_interface.h"
#include "Mutex.h"

#include <algorithm>
#include <string>

const std::string joints[] = {"waist","shoulder","elbow","hand","wrist"};
Mutex mtx;

STR17Interface::STR17Interface(ros::NodeHandle nh, const std::string& dev):st(dev), nh(nh){

	// initialize arm
    ROS_INFO("Initializing ST Arm!");
	st.initialize();
	st.set_speed(10000);
	st.home();
    ROS_INFO("Initialization Complete.");

	for(int i=0; i<N_JOINTS; ++i){
        vel[i] = eff[i] = -1;
		// connect and register the joint state interface
		hardware_interface::JointStateHandle state_handle(joints[i], &pos[i], &vel[i], &eff[i]);
		jnt_state_interface.registerHandle(state_handle);
		// connect and register the joint position interface
		hardware_interface::JointHandle pos_handle(jnt_state_interface.getHandle(joints[i]), &cmd[i]);
		jnt_pos_interface.registerHandle(pos_handle);
	}
    registerInterface(&jnt_state_interface);
	registerInterface(&jnt_pos_interface);

}
STR17Interface::~STR17Interface(){

}
ros::Time STR17Interface::get_time(){
	return ros::Time::now();
}

void cvtJ(std::vector<double>& j){
	// convert to degrees
	j[0] *= 90./B_RATIO;
	j[1] *= 90./S_RATIO;
	j[2] *= 90./E_RATIO;
	j[3] *= 90./W_RATIO;
	j[4] *= 90./T_RATIO;
	j[4] -= j[3];

	for(std::vector<double>::iterator it = j.begin(); it!=j.end();++it){
		double& l = *it;
		l = d2r(l);
	}
}

void cvtJ_i(std::vector<double>& j){
	//invert the conversion
	for(std::vector<double>::iterator it = j.begin(); it!=j.end();++it){
		double& l = *it;
		l = r2d(l);
	}
	j[4] += j[3];

	j[0] = int(j[0] * B_RATIO / 90.);
	j[1] = int(j[1] * S_RATIO / 90.);
	j[2] = int(j[2] * E_RATIO / 90.);
	j[3] = int(j[3] * W_RATIO / 90.);
	j[4] = int(j[4] * T_RATIO / 90.);
}

void STR17Interface::read(const ros::Time& time){
	//alias with reference
	std::vector<double> loc;
	
	// ##### MUTEX #####
	mtx.lock();	
	st.where(loc);
	mtx.unlock();

	if(loc.size() != 5){
		std::cerr << "WARNING :: INVALID LOCATION READ !!! " << std::endl;
		std::cerr << loc.size() << std::endl;
		return;
	}
	// #################

	//reformat based on scale and direction
	cvtJ(loc);

	// fill data -- let controller manager know
	for(int i=0; i<N_JOINTS;++i){
		pos[i] = loc[i];
	}

}

void STR17Interface::write(const ros::Time& time){
	//st.move ... joints
	// info to st-r17, i.e. desired joint states
	std::vector<double> cmd_pos(N_JOINTS);
	for(int i=0; i<N_JOINTS; ++i){
		cmd_pos[i] = cmd[i];
	}

	cvtJ_i(cmd_pos); // invert conversion

	// ##### MUTEX #####
	mtx.lock();
    auto move_flag=false;
    for(int i=0; i<N_JOINTS; ++i){
        float dp = fabs(cmd[i] - pos[i]);
        if(dp > d2r(1.0)){
            move_flag=true;
            break;
        }
    }
    if(move_flag)
        st.move(cmd_pos);
	//for(int i=0; i<N_JOINTS; ++i){
	//	float dp = fabs(cmd[i] - pos[i]); // target-position
	//	if(dp > d2r(1.0)){ // more than 1 degrees different
	//		// TODO : apply scaling factors?
	//		st.move(joints[i], cmd_pos[i]);
	//	}
	//}
	mtx.unlock();
	// #################
};

int main(int argc, char* argv[]){
	ros::init(argc,argv,"st_r17_hardware");
	std::vector<double> j;

	ros::NodeHandle nh;
	ros::NodeHandle nh_priv("~");

	std::string dev;
    float rate;

	nh_priv.param("dev", dev, std::string("/dev/ttyUSB0")); 
    nh_priv.param("rate", rate, 10.0f);

	ROS_INFO("Get Param %s", ros::param::get("/dev", dev)?"SUCCEEDED":"FAILED");
	ROS_INFO("initialized with device %s", dev.c_str());

	STR17Interface st_r17(nh, dev);
	controller_manager::ControllerManager cm(&st_r17, nh);

	ros::AsyncSpinner spinner(1);
	spinner.start();

	ros::Time then = st_r17.get_time();
	ros::Rate r = ros::Rate(rate); // 10 Hz

	while(ros::ok()){
        ROS_INFO_THROTTLE(5.0, "ALIVE");
		ros::Time now = st_r17.get_time();
		ros::Duration period = ros::Duration(now-then);
        then = now;

		st_r17.read(now);
		cm.update(now, period);
		st_r17.write(now);

		r.sleep();
	}
	return 0;
}
