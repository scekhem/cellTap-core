#pragma once
#pragma warning(disable:4996)

#include <pcl/point_types.h>
#include <pcl/common/common.h>

#include <Eigen/Eigen>
#include <Eigen/Core>
#include <Eigen/Dense>

#include <opencv2/opencv.hpp>

#include <vector>
#include <string>

#include <ptcloud.h>
#include <ptframe.h>
#include <PtRobot.h>
#include <frameio.h>
#include "SimpleIni.h"

//USING_NAMESPACE_POINT_CLOUD_DATA

class  Reprojector
{
public:
	// initialize
	void _allocateMemory();
	Reprojector();
	~Reprojector(void);

	PtCloud loadPTData(std::string& filename);

	bool iniparams(const std::string & ini_file);
	bool generate(const PtCloud& pt_data); // reprojet points xyz to matrix
	bool process();

	bool glueAOI();

	bool generateLT(const PtCloud& source_pt_data, bool direction_x = true); // reprojet points xyz to matrix
	bool generateLT(const std::vector<PtFrame>& source_pt_frames, bool direction_x = true);
	bool generateLTS(const std::vector<PtFrame>& source_pt_frames, bool direction_x = true,  int show_pt_num = 0);

	std::vector < std::vector<PtRobot>> getResult();

	std::vector<PtRobot> getResultLT();

	std::vector<PtRobot> getResultLTS(const int points_num);

	void setdebug(const bool value);

private:
	
	pcl::PointCloud<pcl::PointXYZ>::Ptr _updatePoints(Eigen::MatrixXi* depth_mask); // process & show gray img
	pcl::PointCloud<pcl::PointXYZ>::Ptr _updatePointsLT(Eigen::MatrixXi * depth_mask);

	bool debug_model = false;
	pcl::PointXYZ min_p;
	pcl::PointXYZ max_p;

	int depth_wide;
	int depth_height;
	double gap_wide;
	double gap_height;
	float depth_range;

	// ini config params
	double crop_z_min = 0.6;
	int hist_blur_size = 7;
	int hist_threshold_blocksize = 15;

	double winblur_thresh = 3.0;
	double zblur_thresh = 0.5;
	//int margin_x = 50;
	int margin_x = 0;
	int margin_y = 0;
	double compress_x = 5;
	double compress_y = 2;
	bool use_border = false;
	bool from_left = false;
	int crop_head = 0;
	int crop_tail = 0;
	int output_points_num = 100;

	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in;
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_centered;

	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_selected;
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_result;
	std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> resultGM;

	Eigen::MatrixXi matrix_index;
	cv::Mat image_hist_8UC1, image_hist_16UC1;

};	//reprojector