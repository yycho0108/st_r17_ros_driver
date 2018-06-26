#include <Eigen/StdVector>
#include <Eigen/Core>

#include <iostream>
#include <fstream>

#include <stdint.h>

#include "g2o/core/sparse_optimizer.h"
#include "g2o/core/block_solver.h"
#include "g2o/core/linear_solver.h"
#include "g2o/core/solver.h"

#include "g2o/core/optimization_algorithm_levenberg.h"
#include "g2o/core/robust_kernel_impl.h"
#include "g2o/solvers/cholmod/linear_solver_cholmod.h"
#include "g2o/solvers/dense/linear_solver_dense.h"

#include "g2o/types/slam3d_addons/types_slam3d_addons.h"
#include "g2o/types/slam3d/types_slam3d.h"
#include "g2o/core/factory.h"

#include "g2o/stuff/macros.h"
#include "g2o/stuff/command_args.h"

/* ROS STUFF */
#include <sensor_msgs/JointState.h>
#include <geometry_msgs/PoseArray.h>

#include <apriltags2_ros/AprilTagDetectionArray.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/subscriber.h>

#include <tf/transform_datatypes.h>
#include <tf/transform_listener.h>
#include <tf_conversions/tf_eigen.h>

#include <st_r17_calibration/DH.h>

using namespace Eigen;
using namespace g2o;
using namespace std;

struct DH{
	float alpha, a, d, dq;
};

struct Estimate{
	Eigen::Isometry3d value;
	bool seen=false;
};

void dh2T(const DH& dh,
		const float q,
		Eigen::Matrix4d& dst
		){
	float cq = cos(q + dh.dq);
	float sq = sin(q + dh.dq);
	float ca = cos(dh.alpha);
	float sa = sin(dh.alpha);
	dst.setZero();
	dst << cq, -sq, 0, dh.a,
		sq*ca, cq*ca, -sa, -sa*dh.d,
		sq*sa, cq*sa, ca, ca*dh.d,
		0, 0, 0, 1;
}

void forward_kinematics(
		const std::vector<DH>& dh,
		const std::vector<float>& j,
		Eigen::Matrix4d& T,
		Eigen::Matrix4d& tmp
		){
	T.setIdentity();

	//Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
	int n = dh.size();
	//assert(n == j.size());
	for(int i=0; i<n; ++i){
		dh2T(dh[i], j[i], tmp);
		T *= tmp;
	}
}

// IMPORTANT : Eigen::Quaternion is in (wxyz) order
void iso_from_pose(
		geometry_msgs::Pose& src,
		Eigen::Isometry3d& dst){
	auto& p = src.position;
	auto& q = src.orientation;
	dst.linear() = Eigen::Quaterniond(q.w,q.x,q.y,q.z).normalized().toRotationMatrix();
	dst.translation() = Eigen::Vector3d(p.x,p.y,p.z);
}

void pose_from_iso(
		Eigen::Isometry3d& src,
		geometry_msgs::Pose& dst){
	Eigen::Quaterniond q(src.rotation());
	dst.orientation.x = q.x();
	dst.orientation.y = q.y();
	dst.orientation.z = q.z();
	dst.orientation.w = q.w();

	Eigen::Vector3d p(src.translation());
	dst.position.x = p.x();
	dst.position.y = p.y();
	dst.position.z = p.z();
}

class Slam{
	typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::JointState,
	apriltags2_ros::AprilTagDetectionArray> MySyncPolicy;

  private:
	// parameters
	int _num_markers;
	int _batch_size;
	float _tol;
	bool _initialized;
	bool _has_edge_info;
	int _max_iter;
	std::vector<DH> _dh;
	std::vector<std::string> _joints;
	std::string _frame_o, _frame_c;
	bool _verbose;
	float _noise;

	// handles
	ros::NodeHandle& _nh;
	tf::TransformListener _tfl;
	g2o::SparseOptimizer _opt;
	message_filters::Subscriber<sensor_msgs::JointState> _j_sub;
	message_filters::Subscriber<apriltags2_ros::AprilTagDetectionArray> _d_sub;
	message_filters::Synchronizer<MySyncPolicy> _sync;
	ros::Publisher _l_pub;
	ros::Publisher _l_viz_pub;
	ros::Subscriber _dh_sub;

	// temporaries ...
	Eigen::Matrix4d T;
	Eigen::Matrix4d tmp;
	int _m_idx;
	int _last_update;

	// vertices ...
	std::vector<Estimate> estimates;

	// testing - noise handlers
	std::default_random_engine r_gen;
	std::normal_distribution<double> n_dist;

  public:
	Slam(ros::NodeHandle& nh):
		_nh(nh), _tfl(_nh), _sync(MySyncPolicy(20), _j_sub, _d_sub){
			_has_edge_info = false;
			
			// load parameters		
			ros::param::param<int>("~num_markers", _num_markers, 4);
			_initialized = true;
			_initialized &= (_num_markers > 0);
			ros::param::param<int>("~batch_size", _batch_size, 32);
			ros::param::param<float>("~tol", _tol, 1e-6); //NOT USED FOR NOW
			ros::param::param<int>("~max_iter", _max_iter, 100);
			ros::param::param<std::string>("~frame_o", _frame_o, "base_link");
			ros::param::param<std::string>("~frame_c", _frame_c, "stereo_optical_link");
			ros::param::param<bool>("~verbose", _verbose, false);
			ros::param::param<float>("~noise", _noise, 0.0);

			// testing -- initialize random engine
			r_gen.seed( time(0) );
			n_dist.param(std::normal_distribution<double>::param_type(0, _noise));

			_initialized &= (_batch_size > 0);
			std::vector<float> dhv;

			_initialized &= _nh.getParam("dh_flat", dhv); // dh parameter must be supplied!
			ROS_INFO("DH Parameter Input Size : %d", dhv.size());
			_initialized &= set_dh(dhv, _noise); // dh parameter must be valid!
			_initialized &= _nh.getParam("joints", _joints); // joint order

			// initialize g2o
			g2o::BlockSolver_6_3::LinearSolverType* linearSolver = new g2o::LinearSolverCholmod<g2o::BlockSolver_6_3::PoseMatrixType>();
			g2o::BlockSolver_6_3* solver_ptr = new g2o::BlockSolver_6_3(linearSolver);
			g2o::OptimizationAlgorithmLevenberg* alg = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
			_opt.setAlgorithm(alg);
			_opt.setVerbose(_verbose);
			this->reset();

			// initialize handles
			_l_pub = nh.advertise<apriltags2_ros::AprilTagDetectionArray>("/base_to_target", 10);
			_l_viz_pub = nh.advertise<geometry_msgs::PoseArray>("/base_to_target_viz", 10);
			_j_sub.subscribe(nh, "/joint_states", 10);
			_d_sub.subscribe(nh, "/stereo_to_target", 10);
			_dh_sub = nh.subscribe("/dh_params", 10, &Slam::dh_cb, this);

			//_sync = message_filters::Synchronizer<MySyncPolicy>(MySyncPolicy(20), _j_sub, _d_sub);
			_sync.registerCallback(
					boost::bind(&Slam::data_cb, this, _1, _2));
		}

	~Slam(){
		// necessary?
	}
	void dh_cb(const st_r17_calibration::DHConstPtr& msg){
		set_dh(msg->data, 0.0);
	}

	void run(){
		if(!_initialized){
			ROS_ERROR("Initialization Failed; Aborting");
			return;
		}

		ros::Rate rate(50);
		while(ros::ok()){
			ros::spinOnce();
			this->step();
			rate.sleep();
		}
	}

	void publish(){
		apriltags2_ros::AprilTagDetectionArray msg;
		geometry_msgs::PoseArray viz_msg;
		ros::Time stamp = ros::Time::now();
		// 1. build message
		msg.header.stamp = stamp;	
		msg.header.frame_id = this->_frame_o;

		viz_msg.header.stamp = stamp;
		viz_msg.header.frame_id = this->_frame_o;

		for(int i=0; i<_num_markers;++i){
			geometry_msgs::Pose p;

			apriltags2_ros::AprilTagDetection det;
			det.id.push_back(i);
			det.size.push_back(0.0); // not really used
			det.pose.header.stamp = stamp;
			det.pose.header.frame_id = this->_frame_o; // TODO : check if this is consistent
			pose_from_iso(estimates[i].value, det.pose.pose.pose); // TODO : fill in covariance information
			pose_from_iso(estimates[i].value, p);
			msg.detections.push_back(det);
			viz_msg.poses.push_back(p);
		}

		// 2. publish message
		_l_pub.publish(msg);
		_l_viz_pub.publish(viz_msg);
	}

	void step(){
		if( this->_m_idx > this->_batch_size ){
			for(auto& e : this->estimates){
				if(!e.seen){
					ROS_WARN_THROTTLE(1.0, "Landmarks Haven't Been Fully Initialized");
					// some of the landmarks have not been seen yet
					return; 
				}
			}
			_opt.initializeOptimization();
			int steps = _opt.optimize(this->_max_iter, false); // see if online=true will work
			if(steps > 0){
				ROS_INFO("Took %d steps", steps);
				this->_has_edge_info = true;
			}else{
				ROS_WARN("Optimization Failed");
			}
			for(int i=0;i<_num_markers;++i){
				auto v = _opt.vertex(i+1); // account for 1-offset
				estimates[i].value = static_cast<VertexSE3*>(v)->estimate();
			}
			this->publish();
			this->reset();
			this->_last_update = this->_m_idx;
		}
	}

	void reset(){
		if(this->_has_edge_info){
			// perform marginalization + assignment
			std::vector<g2o::OptimizableGraph::Vertex*> mVs;
			for(int i=0; i< 1+_num_markers; ++i){
				mVs.push_back(_opt.vertex(i));
			}
			g2o::SparseBlockMatrixXd spinv;
			_opt.computeMarginals(spinv, mVs);
			ROS_INFO_STREAM("Marginals : " << spinv);
		}else{
			// initialize containers ...
			estimates.resize(_num_markers);

			// add base_link vertex
			auto v0 = new g2o::VertexSE3();
			Eigen::Quaterniond q(1.0,0.0,0.0,0.0); //wxyz
			Eigen::Vector3d t(0,0,0);
			v0->setEstimate(g2o::SE3Quat(q,t));
			v0->setFixed(true);
			v0->setId(0);
			_opt.addVertex(v0);

			// landmarks (initialized with guesses if estimates exist)
			for(int i=0; i<_num_markers;++i){
				auto v = new g2o::VertexSE3();
				v->setId(1+i);
				if(estimates[i].seen){
					v->setEstimate(estimates[i].value);
				}
				_opt.addVertex(v);
			}
			this->_m_idx = 1; //current motion index
		}

		//if(estimates.size() <= 0){
		//	// == first initialization
		//	this->_m_idx = 1;

		//	// add base_link vertex
		//	auto v0 = new g2o::VertexSE3();
		//	Eigen::Quaterniond q(1.0,0.0,0.0,0.0); //wxyz
		//	Eigen::Vector3d t(0,0,0);
		//	v0->setEstimate(g2o::SE3Quat(q,t));
		//	v0->setFixed(true);
		//	v0->setId(0);
		//	_opt.addVertex(v0);

		//	// add landmark vertices
		//	for(int i=0; i<_num_markers;++i){
		//		auto v = new g2o::VertexSE3();
		//		v->setId(1+i);
		//		_opt.addVertex(v);
		//	}
		//	estimates.resize(_num_markers);
		//}else{
		//	for(auto& v : _opt.vertices()){
		//		if(v.first >= (1+_num_markers)){
		//			static_cast<VertexSE3*>(v.second)->setMarginalized(true);
		//		}
		//	}
		//}


		//_opt.clear(); // automatically deletes all vertices/edges
		//// TODO : investigate if other side-effects exist

		//// add base_link vertex
		//auto v0 = new g2o::VertexSE3();
		//Eigen::Quaterniond q(1.0,0.0,0.0,0.0); //wxyz
		//Eigen::Vector3d t(0,0,0);
		//v0->setEstimate(g2o::SE3Quat(q,t));
		//v0->setFixed(true);
		//v0->setId(0);
		//_opt.addVertex(v0);

		//// landmarks (initialized with guesses if estimates exist)
		//for(int i=0; i<_num_markers;++i){
		//	auto v = new g2o::VertexSE3();
		//	v->setId(1+i);
		//	if(estimates[i].seen){
		//		v->setEstimate(estimates[i].value);
		//	}
		//	_opt.addVertex(v);
		//}

		//this->_m_idx = 1; //current motion index
	}

	g2o::VertexSE3* add_motion(Eigen::Isometry3d& x){
		int i0 = (1 + _num_markers);
		auto v = new g2o::VertexSE3();
		v->setEstimate(x);
		v->setId(i0 + this->_m_idx);
		_opt.addVertex(v);

		auto e = new g2o::EdgeSE3();
		e->setVertex(0, _opt.vertex(0));
		e->setVertex(1, v);
		e->setInformation(100.0 * Eigen::Matrix<double,6,6>::Identity()); //
		
		e->setMeasurement(x);
		_opt.addEdge(e);

		++this->_m_idx;
		return v; // note, edge doesn't get returned
	}

	g2o::VertexSE3* add_landmark(int l_idx, Eigen::Isometry3d& x){
		// only adds vertex
		g2o::VertexSE3* v = dynamic_cast<g2o::VertexSE3*>(_opt.vertex(1 + l_idx));
		//ROS_INFO_STREAM(x.matrix());

		if(!estimates[l_idx].seen){
			// initialize with guess
			v->setEstimate(x);
			estimates[l_idx].value = x;
			estimates[l_idx].seen = true;
		}
		return v;
	}

	void data_cb(const sensor_msgs::JointStateConstPtr& j_msg, const apriltags2_ros::AprilTagDetectionArrayConstPtr& d_msg){
		// enforce at least 1 visible detection
		if(d_msg->detections.size() <= 0){
			return;
		}

		int n = j_msg->position.size();

		// NOTE: n-th joint is hard-coded to be fixed to account for final static transformation.(ee -> camera)
		std::vector<float> jpos(n+1);

		// reorder joint values ...
		for(int i=0; i<n; ++i){
			for(int j=0; j<n; ++j){
				if(_joints[i] == j_msg->name[j]){
					jpos[i] = j_msg->position[j];
				}
			}
		}

		if( jpos.size() != this->_dh.size()){
			ROS_ERROR_THROTTLE(1.0, "Invalid number of Joints or DH Parameters");
			return;
		}

		forward_kinematics(this->_dh, jpos, T, tmp);
		Eigen::Isometry3d T_oc;

		T_oc.matrix() = T; //??

		//	= T; // set from matrix, TODO : just set directly? idk
		// T_oc = Transformation from origin(base_link) -> camera
		
		geometry_msgs::PoseStamped tfm0, tfm1;

		auto v = this->add_motion(T_oc);
		try{
			for(auto& d : d_msg->detections){
				int idx = d.id[0]; // landmark index

				// transform pose to reference frame_c
				tfm0.header.stamp = d_msg->header.stamp;
				tfm0.header.frame_id = d_msg->header.frame_id; // TODO : check valid
				tfm0.pose = d.pose.pose.pose;
				_tfl.transformPose(this->_frame_c, tfm0, tfm1);

				// --> tfm1
				Eigen::Isometry3d T_cz; // Transformation from camera -> landmark
				iso_from_pose(tfm1.pose, T_cz);
				Eigen::Isometry3d T_oz = T_oc * T_cz; // transformation from base_link -> landmark
				auto v_z = this->add_landmark(idx, T_oz); // add guess if it doesn't exist yet

				// add corresponding landmark edge
				auto e = new g2o::EdgeSE3();
				e->setVertex(0, v);
				e->setVertex(1, v_z);
				e->setInformation(100.0 * Eigen::Matrix<double,6,6>::Identity());
				// TODO : configurable information matrix
				e->setMeasurement(T_cz);
				_opt.addEdge(e);
			}
		}catch(const tf::LookupException& e){
			ROS_WARN("TF Lookup Exception");
			return;
		}catch(const tf::ExtrapolationException& e){
			ROS_WARN("TF Extrapolation Exception");
			return;
		}catch(const tf::TransformException& e){
			ROS_WARN("TF Transform Exception");
			return;
		}
	}

	bool set_dh(const std::vector<float>& src, float noise){
		int n = src.size();

		if(! (n%4 == 0)){
			ROS_WARN_THROTTLE(1.0, "Invalid DH Parameters; ignored");
			return false;
		}

		if(noise > 0){
			n_dist.param(std::normal_distribution<double>::param_type(0, _noise));
		}

		this->_dh.clear();
		for(int i0=0; i0<n; i0+=4){
			DH dh_i{src[i0], src[i0+1], src[i0+2], src[i0+3]};
			if(noise > 0){
				dh_i.alpha += n_dist(r_gen);
				dh_i.a     += n_dist(r_gen);
				dh_i.d     += n_dist(r_gen);
				dh_i.dq     += n_dist(r_gen);
			}
			this->_dh.push_back(dh_i);
		}
		return true;
	}
};

int main(int argc, char* argv[]){
	ros::init(argc, argv, "slam");
	ros::NodeHandle nh;
	Slam slam(nh);
	slam.run();
	return 0;
}

// REFERENCE
//int main(int argc, const char* argv[]){
//	auto* factory = g2o::Factory::instance();
//	factory->registerType("VERTEX_SE3:QUAT", new g2o::HyperGraphElementCreator<g2o::VertexSE3>); 
//	factory->registerType("EDGE_SE3:QUAT", new g2o::HyperGraphElementCreator<g2o::EdgeSE3>); 
//	factory->printRegisteredTypes(cout, true);
//
//	g2o::SparseOptimizer opt;
//	opt.setVerbose(false);
//
//	g2o::BlockSolver_6_3::LinearSolverType* linearSolver = new g2o::LinearSolverCholmod<g2o::BlockSolver_6_3::PoseMatrixType>();
//	g2o::BlockSolver_6_3* solver_ptr = new g2o::BlockSolver_6_3(linearSolver);
//
//	g2o::OptimizationAlgorithmLevenberg* alg = new g2o::OptimizationAlgorithmLevenberg(
//			std::move(solver_ptr)
//			);
//	opt.setAlgorithm(alg);
//
//	if (!opt.load("/tmp/joint_graph.g2o")){
//		cerr << "Error loading Graph" << endl;
//		return 2;
//	}
//
//	g2o::OptimizableGraph::Vertex* v = static_cast<OptimizableGraph::Vertex*>(opt.vertices().find(0)->second);
//	v->setFixed(true);
//
//	opt.setVerbose(true);
//	opt.initializeOptimization();
//	opt.optimize(1000); // give it a significant number of iterations for it to work
//
//	int n = opt.vertices().size();
//	for(int i=n-8; i<n; ++i){ // 8 == num_markers
//		auto v = opt.vertex(i);
//		g2o::OptimizableGraph::Vertex* ov = static_cast<OptimizableGraph::Vertex*>(v);
//		opt.saveVertex(cout, ov);
//	}
//	std::cout << "??" << std::endl;
//	return 0;
//}
