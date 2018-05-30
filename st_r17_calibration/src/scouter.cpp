#include <moveit/move_group_interface/move_group.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>

#include <moveit_msgs/DisplayRobotState.h>
#include <moveit_msgs/DisplayTrajectory.h>

#include <moveit_msgs/AttachedCollisionObject.h>
#include <moveit_msgs/CollisionObject.h>
#include <moveit_msgs/ApplyPlanningScene.h>
#include <moveit_msgs/Grasp.h>

#include <moveit_visual_tools/moveit_visual_tools.h>

#include "std_msgs/Float64MultiArray.h"
#include "std_msgs/Empty.h"
#include <geometry_msgs/Pose.h>
#include <geometry_msgs/PoseArray.h>
#include <tf/tf.h>


#include <iostream>
#include <memory>

#ifndef M_PI
#define M_PI 3.14159265358979
#endif

typedef std::vector<double> d_vec;

bool check_moveit(moveit::planning_interface::MoveItErrorCode c){
    return c == moveit::planning_interface::MoveItErrorCode::SUCCESS; 
}

struct solutionSort{
	const d_vec& seed_pose;
	solutionSort(const d_vec& seed_pose):seed_pose(seed_pose){
	}
	bool operator()(const d_vec& v_1, const d_vec& v_2){
		int n = v_1.size();
		float d_1 = 0.0;
		float d_2 = 0.0;
		//ASSERT n == 5
		for(int i=0; i<n; ++i){
			d_1 += fabs(v_1[i] - seed_pose[i]);
			d_2 += fabs(v_2[i] - seed_pose[i]);
		}	
		return d_1 < d_2;
	}	
};

class ScouterMoveGroupInterface{
	private:
		const ros::NodeHandle& nh;
        tf::TransformListener _tfl;
		visualization_msgs::Marker marker_msg;

		moveit::planning_interface::MoveGroupInterface group;
		const moveit::core::JointModelGroup& j_group;
        ros::Subscriber gt_sub;
        std::vector<geometry_msgs::Point> ps;

        float noise;
        float delay;

	public:
		ScouterMoveGroupInterface(ros::NodeHandle& nh);
		bool moveToPose(const geometry_msgs::Pose& target_pose);
        void test();
        void move();
        void gt_cb(const geometry_msgs::PoseArrayConstPtr& msg);
};

float random_uniform(float mn, float mx){
    float x = float(rand()) / RAND_MAX;
    return mn + x * (mx - mn);
}

//tf::Quaternion vqv(tf::Vector3& src, tf::Vector3& dst, eps = 1e-3){
//    tf::Vector3 v0 = src.normalized();
//    tf::Vector3 v1 = dst.normalized();
//    
//    float s = v0.dot(v1);
//
//    if(s > (1.0 - eps)){
//        //xyzw
//        return tf::Quaternion(0,0,0,1);
//    }
//    else if (s < -(1.0 - eps)){
//        return tf::Quaternion(1,0,0,0);
//    }
//
//    tf::Vector3 c = v0.cross(v1);
//
//    tf::Quaternion q(c.x, c.y,  c.z, 1.0 + s);
//    q.normalize();
//    return q;
//}

void ScouterMoveGroupInterface::test(){

    float deg = M_PI / 180;
    float r = random_uniform(0.5, 1.2);
    float phi = random_uniform(0 * deg, 90 * deg);
    float theta = random_uniform(-M_PI, M_PI);

    float x = r * cos(phi) * cos(theta);
    float y = r * cos(phi) * sin(theta);
    float z = r * sin(phi);

    tf::Quaternion q;
    q = tf::createQuaternionFromRPY(0.0, -phi, theta);
    
    geometry_msgs::Pose target_pose;

    tf::quaternionTFToMsg(q, target_pose.orientation);

    target_pose.position.x = x;
    target_pose.position.y = y;
    target_pose.position.z = z;

	group.setStartStateToCurrentState();
    group.setGoalJointTolerance(0.034);
    group.setGoalOrientationTolerance(0.034);
    group.setGoalPositionTolerance(0.05);
    moveit::planning_interface::MoveGroupInterface::Plan my_plan;
    // plan #1 : simply try to plan for target pose
    group.setPoseTarget(target_pose);
    bool success = check_moveit(group.plan(my_plan));
    std::cout << int(success) << ',' << r << ',' << phi << ',' << theta << std::endl;
}

void ScouterMoveGroupInterface::move(){
    float deg = M_PI / 180;

    float r = random_uniform(0.75, 1.2);
    float phi = random_uniform(15 * deg, 45 * deg);
    float theta = random_uniform(-M_PI, M_PI);
    //float theta = random_uniform(-1.0, 1.0);

    tf::Quaternion q;
    if(ps.size() > 0){
        int i = rand() % ps.size();
        auto& p = ps[i];
        
        theta = atan2(p.y, p.x);
        theta += random_uniform(-noise, noise);
        phi   = atan2(p.z, sqrt(p.y*p.y+p.x*p.x));
        phi   += random_uniform(-noise ,noise);
    }else{
        ROS_INFO_THROTTLE(1.0, "%d", ps.size());
        return;
    }

    float x = r * cos(phi) * cos(theta);
    float y = r * cos(phi) * sin(theta);
    float z = r * sin(phi);
     geometry_msgs::Pose target_pose;

    // set orientation
    q = tf::createQuaternionFromRPY(0.0, -phi, theta);
    //tf::Quaternion q;
    //q.setEuler(0.0, -phi, theta);
    tf::quaternionTFToMsg(q, target_pose.orientation);

    //ste position
    target_pose.position.x = x;
    target_pose.position.y = y;
    target_pose.position.z = z;

    bool success = moveToPose(target_pose);
    if(!success){
        ROS_INFO("Failed to go to target : (%.2f), %.2f,%.2f,%.2f | %.2f,%.2f,%.2f", r, x,y,z,0.0,phi,theta);
    }else{
        ros::Duration(delay).sleep();
    }

    //geometry_msgs::Pose next_pose = current_pose;
    //bool success = moveToPose(next_pose);
    //if(success){
    //    current_pose = next_pose;
    //}
}

bool ScouterMoveGroupInterface::moveToPose(const geometry_msgs::Pose& target_pose){

	group.setStartStateToCurrentState();
    group.setGoalJointTolerance(0.034);
    group.setGoalOrientationTolerance(0.034);
    group.setGoalPositionTolerance(0.05);

    moveit::planning_interface::MoveGroupInterface::Plan my_plan;

    // plan #1 : simply try to plan for target pose
    group.setPoseTarget(target_pose);
    bool success = check_moveit(group.plan(my_plan));
    if(success){
        success = check_moveit(group.move());
        return success;
    }

    // plan #2 : use IK Solver directly to search for more solutions
	const kinematics::KinematicsBaseConstPtr& solver_ptr = j_group.getSolverInstance();

	std::vector<geometry_msgs::Pose> target_pose_v;
	target_pose_v.push_back(target_pose);

	d_vec seed_pose;
	group.getCurrentState()->copyJointGroupPositions(group.getCurrentState()->getRobotModel()->getJointModelGroup(group.getName()), seed_pose);
	std::vector<d_vec> solutions;

	kinematics::KinematicsResult result;
	kinematics::KinematicsQueryOptions options;

	options.return_approximate_solution = true;
	options.discretization_method = kinematics::DiscretizationMethods::NO_DISCRETIZATION; // TODO : play with this

	if(solver_ptr->getPositionIK(target_pose_v,seed_pose,solutions,result,options)){
		ROS_INFO("ALTERNATIVE IK SOLUTION CANDIDATES FOUND! ");
		std::cout << std::setprecision(4);

		std::vector<std::string> link_names;
		link_names.push_back(group.getEndEffectorLink());

		// sorting with least difference as beginning, most difference as end
		// "prefer" a solution closer to the current state
		std::sort(solutions.begin(), solutions.end(), solutionSort(seed_pose));

		for(std::vector<d_vec>::const_iterator it = solutions.begin(); it != solutions.end(); ++it){
			const d_vec& sol = (*it);
			float pitch = fabs(sol[1] + sol[2] + sol[3]);
			ROS_INFO("PITCH : %f\n", pitch);
			group.setJointValueTarget(sol);
			bool success = check_moveit(group.plan(my_plan));
			if(success){
				success = check_moveit(group.move());
                ROS_INFO(" Goal %s", (success? "SUCCESS" : "FAIL"));
				return success;
			}
		}
	}
	return false;
}

void ScouterMoveGroupInterface::gt_cb(const geometry_msgs::PoseArrayConstPtr& msg){
    if(msg->poses.size() > 0){
        for(auto& p : msg->poses){
            ps.push_back(p.position);
        }
        gt_sub.shutdown();
    }
    //if(ps.size() > 0){
}

ScouterMoveGroupInterface::ScouterMoveGroupInterface(ros::NodeHandle& nh):
	nh(nh),
	_tfl(ros::Duration(10.0)),
	group("arm_group"),
	j_group(*group.getRobotModel()->getJointModelGroup("arm_group"))
{
    if(! ros::param::get("~noise", noise)){
        noise = (M_PI / 180) * 1.0; // +-1 deg.
    }
    if(! ros::param::get("~delay", delay)){
        delay = 0.0;
    }
	// **** CONFIGURE GROUP **** //
	group.setNumPlanningAttempts(8); // attempt three times

	group.setGoalPositionTolerance(0.05); // 5cm tolerance
	group.setGoalOrientationTolerance(1.0 * M_PI / 180); // 5 deg. tolerance
	group.setGoalJointTolerance(1.0 * M_PI / 180); // 5 deg. tolerance

	group.setSupportSurfaceName("table");
	group.setStartStateToCurrentState();
	group.setWorkspace(-2,-2,0,2,2,2);
	group.setPlannerId("RRTConnectkConfigDefault");

	// **** SETUP ENVIRONMENT **** //
	moveit::planning_interface::PlanningSceneInterface planning_scene_interface;  
	//spawnObject(planning_scene_interface);

	// **** DISPLAY RELEVANT INFO **** //
	ROS_INFO("Reference frame (Plan): %s", group.getPlanningFrame().c_str());
	ROS_INFO("Reference frame (End Effector): %s", group.getEndEffectorLink().c_str());
	geometry_msgs::Pose p = group.getCurrentPose().pose;
	ROS_INFO("Current Pose: %f %f %f | %f %f %f %f\n", p.position.x, p.position.y, p.position.z, p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w);

	// **** FORMAT MARKER MESSAGE **** //
	marker_msg.header.frame_id = "base_link";
	marker_msg.scale.x = marker_msg.scale.y = marker_msg.scale.z = 0.05;
	marker_msg.type = marker_msg.SPHERE;
	marker_msg.id = 0;
	marker_msg.pose.orientation.w = 1.0; // believe auto filled to 0
	marker_msg.color.a = 1;
	marker_msg.color.r = 1;
	marker_msg.color.g = 0;
	marker_msg.color.b = 1;

    //wonder?
    gt_sub = nh.subscribe("/base_to_target", 10, &ScouterMoveGroupInterface::gt_cb, this);
}

int main(int argc, char **argv)
{
	ros::init(argc, argv, "scouter_move_group_interface");
	ros::NodeHandle nh;  

	ScouterMoveGroupInterface scouter(nh);

	ros::AsyncSpinner spinner(1);
	spinner.start();

	// from here listen to callbacks
	ros::Rate r = ros::Rate(50.0);
	while(ros::ok()){
		ros::spinOnce();
        //scouter.test();
        scouter.move();
		r.sleep();
	}

	ros::shutdown();  
	return 0;
}
