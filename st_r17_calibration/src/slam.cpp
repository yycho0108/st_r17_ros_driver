#include <Eigen/StdVector>
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

#include <sensor_msgs/JointState.h>
#include <apriltags2_ros/AprilTagDetectionArray.h>

using namespace Eigen;
using namespace g2o;
using namespace std;

class Slam{
	private:
		g2o::SparseOptimizer _opt;

	public:
		Slam(){
			// initialize g2o
			g2o::BlockSolver_6_3::LinearSolverType* linearSolver = new g2o::LinearSolverCholmod<g2o::BlockSolver_6_3::PoseMatrixType>();
			g2o::BlockSolver_6_3* solver_ptr = new g2o::BlockSolver(linearSolver)
				g2o::OptimizationAlgorithmLevenberg* alg = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
			_opt.setAlgorithm(alg);

			// add node 0 (base_link)
			g2o::VertexSE3 v;
			Eigen::Quaterniond q(1.0,0.0,0.0,0.0);
			Eigen::Vector3d t(0,0,0);
			v->setEstimate(g2o::SE3Quat(q0,t0));
			v->setFixed(true);
			_opt.addVertex(v);
		}

		~Slam(){
			// necessary?
		}

		void data_cb(){

		}
};

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
