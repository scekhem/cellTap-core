#include "p2cv.h"
#include <pcl/filters/filter.h>
#include <pcl/common/transforms.h>
#include <pcl/visualization/cloud_viewer.h>
#include<pcl/filters/statistical_outlier_removal.h>
#include <pcl/io/pcd_io.h>

#include <pcl/surface/mls.h>
#include <pcl/features/normal_3d.h>

#include <opencv2/highgui/highgui_c.h>
#include <opencv2/core/eigen.hpp>
#include <opencv2/xfeatures2d.hpp>

using namespace std;

std::vector<cv::Point2i> winBlur(std::vector<cv::Point2i> pix_line, const int channel=0, const double blur_size = 2, const int iter_time = 5)
{
	std::vector<cv::Point2i> res;
	try {
		res.push_back(pix_line[0]);
		Eigen::MatrixXd matrix_line = Eigen::MatrixXd::Zero(1, pix_line.size());
		for (int i = 1; i < pix_line.size(); i++)
		{
			if (channel == 0) { matrix_line(0, i) = pix_line[i].x * 10; }
			else if (channel == 1) { matrix_line(0, i) = pix_line[i].y; }
			//std::cout << pix_line[i].x << endl;
		}
		cv::Mat line;
		cv::eigen2cv(matrix_line, line);
		line.convertTo(line, CV_64FC1);
		cv::Mat robert = (cv::Mat_<double>(1, 23) << 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05,
			0, 0, 0,
			0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05);

		for (int i = 0; i < iter_time; i++)
		{
			cv::filter2D(line, line, -1, robert, cv::Point(-1, -1));
		}

		cv::cv2eigen(line, matrix_line);
		for (int i = 0; i < pix_line.size(); i++)
		{

			if (channel == 0)
			{
				//std::cout << pix_line[i].x << " - ";
				//pix_line[i].x = matrix_line(0, i)/10;
				if (abs(matrix_line(0, i) / 10 - pix_line[i].x < blur_size)) {
					//pix_line[i].x = matrix_line(0, i) / 10;
					res.push_back(pix_line[i]);
				}
				//std::cout << matrix_line(0, i) / 10 << endl;
			}
			else if (channel == 1)
			{
				pix_line[i].y = matrix_line(0, i);
				res.push_back(pix_line[i]);
			}

		}
	}
	catch (cv::Exception& e)
	{
		const char* err_msg = e.what();
		std::cout << err_msg << endl;
		return res;
	}
	res.push_back(pix_line[pix_line.size()-1]);
	return res;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr pointsFilter(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_unfiltered, const int channel, const double thresh = 1)
{
	pcl::PointCloud<pcl::PointXYZ>::Ptr res(new pcl::PointCloud<pcl::PointXYZ>());
	Eigen::MatrixXd matrix_line = Eigen::MatrixXd::Zero(1, cloud_unfiltered->points.size());
	for (int i = 0; i < cloud_unfiltered->points.size(); i++)
	{
		if (channel == 0) { matrix_line(0, i) = cloud_unfiltered->points[i].x*100; }
		else if (channel == 1) { matrix_line(0, i) = cloud_unfiltered->points[i].y*100; }
		else if (channel == 2) { matrix_line(0, i) = cloud_unfiltered->points[i].z*100; }
		//std::cout << pix_line[i].x << endl;
	}
	cv::Mat line;
	cv::eigen2cv(matrix_line, line);
	line.convertTo(line, CV_64FC1);
	cv::Mat robert = (cv::Mat_<double>(1, 15) << 0.1, 0.1, 0.1, 0.1, 0.1, -0.2, -0.2, -0.2, -0.2, -0.2, 0.1, 0.1, 0.1, 0.1, 0.1);
	cv::filter2D(line, line, -1, robert, cv::Point(-1, -1));

	cv::cv2eigen(line, matrix_line);
	for (int i = 0; i < cloud_unfiltered->points.size(); i++)
	{
		//std::cout << matrix_line(0, i) << " - " << cloud_unfiltered->points[i].z << endl;
		if (abs(matrix_line(0, i)) < thresh) {
			
			res->points.push_back(cloud_unfiltered->points[i]);
		}
	}
	return res;
}

class Vistool
{
public:
	boost::shared_ptr<pcl::visualization::PCLVisualizer> pcl_viewer;
};

void Reprojector::_allocateMemory()
{
	cloud_in.reset(new pcl::PointCloud< pcl::PointXYZ >());
	cloud_centered.reset(new pcl::PointCloud< pcl::PointXYZ >());
	cloud_result.reset(new pcl::PointCloud< pcl::PointXYZ >());
	cloud_selected.reset(new pcl::PointCloud< pcl::PointXYZ >());
}

Reprojector::~Reprojector(void)
{
	image_hist_8UC1.release();
	image_hist_16UC1.release();
	matrix_index.setRandom();
}

Reprojector::Reprojector()
{
	_allocateMemory();
}

PtCloud Reprojector::loadPTData(std::string& filename)
{
	PtCloud container;
	std::string full_path = filename;
	//
	//auto start = clock();
	std::cout << "loading..." << std::endl;
	container = restoreCloudDataBinary(full_path);

	__int64 data_size = 0;

	std::cout << "[p2cv] load frames from :" << full_path.c_str() << std::endl
		<< "[p2cv] countPerLine :" << container.getNodeCountPerLine() << std::endl
		<< "[p2cv] lineCount :" << container.getLineCount() << std::endl;
	return container;
}

bool Reprojector::generate(const PtCloud & pt_data)
{
	int line_count = pt_data.getLineCount();
	int cols_count = pt_data.getNodeCountPerLine();
	if (line_count*cols_count < 32000)
	{
		std::cout << "[error] ptdata size error:" << line_count << " x " << cols_count << std::endl;
		return false;
	}
	std::cout << "[p2cv] get ptdata:" << line_count << " x " << cols_count << std::endl;
	pcl::PointXYZ temp_point;
	pcl::PointCloud< pcl::PointXYZ > temp_cloud;
	std::vector<int> intensity;
	std::vector<PtNode> pt_nodes = pt_data.getNodes();
	std::cout << "get Nodes:" << pt_nodes.size() << std::endl;
	double max_x = 0, max_y = 0, min_x = 99999, min_y = 99999;
	for (int i = 0; i < pt_nodes.size(); i++)
	{
		if (!isnan(pt_nodes[i].laser.x()))
		{
			if (pt_nodes[i].laser.x() > max_x) max_x = pt_nodes[i].laser.x();
			else if (pt_nodes[i].laser.x() < min_x) min_x = pt_nodes[i].laser.x();
		}
		if (!isnan(pt_nodes[i].laser.y()))
		{
			if (pt_nodes[i].laser.y() > max_y) max_y = pt_nodes[i].laser.y();
			else if (pt_nodes[i].laser.y() < min_y) min_y = pt_nodes[i].laser.y();
		}
		if (pt_nodes[i].laser.hasNaN()) continue;
		else {

			temp_point.x = pt_nodes[i].laser.x();
			temp_point.y = pt_nodes[i].laser.y();
			temp_point.z = -pt_nodes[i].laser.z();

			intensity.push_back(i);
			temp_cloud.push_back(temp_point);
		}
	}
	
	*cloud_in = temp_cloud;
	pcl::getMinMax3D(*cloud_in, min_p, max_p);				// get points max & min range in xyz
	std::cout << "[p2cv] frame range:" << max_x << " - " << min_x << " ; " << max_y << " - " << min_y << endl;
	gap_wide = (max_x - min_x) / (cols_count - 1) * compress_x;
	gap_height = (max_y - min_y) / (line_count - 1) * compress_y;
	depth_range = max_p.z - min_p.z;
	if (gap_wide <= 0 || gap_height <= 0 || depth_range <=0) {
		std::cout << "[error] pt points axis error !" << std::endl;
		return false;
	}
	line_count = (max_p.y - min_p.y) / gap_height + 1;
	cols_count = (max_p.x - min_p.x) / gap_wide + 1;
	Eigen::MatrixXd matrix_depth_8bit = Eigen::MatrixXd::Zero(line_count + 100, cols_count + 100);
	Eigen::MatrixXd matrix_depth_16bit = Eigen::MatrixXd::Zero(line_count + 100, cols_count + 100);
	matrix_index = Eigen::MatrixXi::Zero(line_count + 100, cols_count + 100);

	std::cout << "[p2cv] point range:" << max_p << min_p << std::endl;
	std::cout << "[p2cv] depth img size:" << cols_count + 100 << " - " << line_count + 100 << std::endl;
	std::cout << "[p2cv] depth range:" << depth_range << std::endl;
	Eigen::Affine3f transform_2 = Eigen::Affine3f::Identity();
	transform_2.translation() << -min_p.x, -min_p.y, 0.0;
	transform_2.rotate(Eigen::AngleAxisf(0, Eigen::Vector3f::UnitZ())); // 0 deg in rotation as default
	pcl::transformPointCloud(*cloud_in, *cloud_centered, transform_2);
	for (int i = 0; i < cloud_centered->points.size(); i++)
	{
		int pos_x = cloud_centered->points[i].x / gap_wide + 50;
		int pos_y = cloud_centered->points[i].y / gap_height + 50;
		//matrix_depth_8bit(pos_y, pos_x) = (cloud_centered->points[i].z - min_p.z - (depth_range * crop_z_min)) / (depth_range * (1 - crop_z_min)) * 255;
		//matrix_depth_16bit(pos_y, pos_x) = (cloud_centered->points[i].z - min_p.z - (depth_range * crop_z_min)) / (depth_range * (1 - crop_z_min)) * 32767;
		matrix_depth_8bit(pos_y, pos_x) = (cloud_centered->points[i].z - min_p.z -  crop_z_min) / (depth_range - crop_z_min) * 255;
		matrix_depth_16bit(pos_y, pos_x) = (cloud_centered->points[i].z - min_p.z - crop_z_min) / (depth_range - crop_z_min) * 32767;
		matrix_index(pos_y, pos_x) = i;
	}

	cv::Mat img_temp;
	cv::eigen2cv(matrix_depth_8bit, img_temp);
	img_temp.convertTo(this->image_hist_8UC1, CV_8UC1);
	cv::medianBlur(this->image_hist_8UC1, this->image_hist_8UC1, 3);
	cv::normalize(this->image_hist_8UC1, this->image_hist_8UC1, 0, 255, cv::NORM_MINMAX);

	cv::eigen2cv(matrix_depth_16bit, img_temp);
	img_temp.convertTo(img_temp, CV_16SC1);
	cv::medianBlur(img_temp, img_temp, 3);
	cv::Sobel(img_temp, img_temp, CV_16S, 1, 0, 5, 2.5, 0.0);
	cv::normalize(img_temp, this->image_hist_16UC1, -20, 255, cv::NORM_MINMAX);
	this->image_hist_16UC1.convertTo(this->image_hist_16UC1, CV_8UC1);

	return true;
}

bool Reprojector::process()
{
	try {
		cv::Mat img_temp, result_GE, result_UP;
		img_temp = image_hist_8UC1.clone();
		cv::adaptiveThreshold(img_temp, result_UP, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY_INV, hist_threshold_blocksize, 2);

		/// 去除第一层外轮廓
		if (!use_border) {
			cv::Mat roElement = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(10, 10)); // 内核
			cv::Mat coElement = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(30, 30)); // 内核
			result_GE = result_UP.clone();		// 边缘图
			cv::Rect ccomp;
			cv::floodFill(result_UP, cv::Point(1, 1), cv::Scalar(255), &ccomp, cv::Scalar(20), cv::Scalar(20));  // 填充轮廓外区域
			cv::bitwise_not(result_UP, result_UP, result_GE);													 // 去除填充图上的轮廓线条
			cv::threshold(result_UP, result_UP, 10, 255, cv::THRESH_BINARY_INV);
			cv::erode(result_UP, result_UP, coElement);
			cv::erode(result_UP, result_UP, coElement);
			cv::bitwise_and(result_GE, result_UP, result_UP);
			cv::erode(result_UP, result_UP, roElement);
			cv::erode(result_UP, result_UP, roElement);
			cv::dilate(result_UP, result_UP, roElement);
			cv::dilate(result_UP, result_UP, roElement);
		}
		///
		//cv::namedWindow("result_UP", CV_WINDOW_NORMAL);
		//cv::imshow("result_UP", result_UP);
		//cv::waitKey(0);
		Eigen::MatrixXi depth_mask;
		cv::cv2eigen(result_UP, depth_mask);
		_updatePoints(&depth_mask);

		if (debug_model) {
			Vistool* vistool;
			vistool = new Vistool();
			vistool->pcl_viewer.reset(new pcl::visualization::PCLVisualizer("3D Viewer"));
			pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZ> fildColor(cloud_centered, "z");
			pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> high_light(cloud_selected, 255, 255, 255);
			vistool->pcl_viewer->addPointCloud<pcl::PointXYZ>(cloud_centered, fildColor, "centered cloud");
			vistool->pcl_viewer->addPointCloud<pcl::PointXYZ>(cloud_selected, high_light, "trajectory");
			vistool->pcl_viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 8, "trajectory");
			vistool->pcl_viewer->spin();
			delete vistool;
		}
	}
	catch (cv::Exception& e)
	{
		const char* err_msg = e.what();
		std::cout << err_msg << endl;
		return false;
	}
	return true;
}

bool Reprojector::glueAOI()
{
	try {
		cv::Mat img_temp, result_GE, result_UP;
		img_temp = image_hist_16UC1.clone();

		cv::namedWindow("image_hist_8UC1", CV_WINDOW_NORMAL);
		cv::imshow("image_hist_8UC1", img_temp);
		cv::waitKey(0);

		cv::Sobel(img_temp, result_UP, CV_8UC1, 1, 0, 3);
		cv::Sobel(img_temp, result_GE, CV_8UC1, 0, 1, 3);
		cv::bitwise_or(result_UP, result_GE, result_GE);
		cv::namedWindow("result_UP", CV_WINDOW_NORMAL);
		cv::imshow("result_UP", result_GE);
		///
		cv::Mat glue_mask = cv::Mat::zeros(result_GE.size(),CV_8UC1);
		std::vector<cv::Point2i> glue_points;
		for (int i = 2; i < cloud_selected->points.size() - 2; i++)
		{
			cv::Point2i temp_point;
			temp_point.x = cloud_selected->points[i].x / gap_wide + 50;
			temp_point.y = cloud_selected->points[i].y / gap_height + 50;
			glue_points.push_back(temp_point);
		}

		/// draw out glue way
		cv::polylines(glue_mask,glue_points,false,cv::Scalar(255),150);
		cv::bitwise_and(result_GE,glue_mask, result_GE);
		cv::threshold(result_GE, result_GE, 150, 255, cv::THRESH_BINARY);
		cv::namedWindow("glue_mask", CV_WINDOW_NORMAL);
		cv::imshow("glue_mask", result_GE);
		cv::waitKey(0);
		//Eigen::MatrixXi depth_mask;
		//cv::cv2eigen(result_UP, depth_mask);
		//_updatePoints(&depth_mask);

	}
	catch (cv::Exception& e)
	{
		const char* err_msg = e.what();
		std::cout << err_msg << endl;
		return false;
	}
	return true;
}

bool Reprojector::generateLT(const PtCloud & pt_data, bool direction_x)
{
	int line_count = pt_data.getLineCount();
	int cols_count = pt_data.getNodeCountPerLine();
	if (line_count*cols_count < 32000)
	{
		std::cout << "[error] ptdata size error:" << line_count << " x " << cols_count << std::endl;
		return false;
	}
	std::cout << "[p2cv] get ptdata:" << line_count << " x " << cols_count << std::endl;
	pcl::PointXYZ temp_point;
	pcl::PointCloud< pcl::PointXYZ > temp_cloud;
	std::vector<int> intensity;
	std::vector<PtNode> pt_nodes = pt_data.getNodes();
	std::cout << "get Nodes:" << pt_nodes.size() << std::endl;
	double max_x = 0, max_y = 0, min_x = 99999, min_y = 99999;
	for (int i = 0; i < pt_nodes.size(); i++)
	{
		if (!isnan(pt_nodes[i].laser.x()))
		{
			if (pt_nodes[i].laser.x() > max_x) max_x = pt_nodes[i].laser.x();
			else if (pt_nodes[i].laser.x() < min_x) min_x = pt_nodes[i].laser.x();
		}
		if (!isnan(pt_nodes[i].laser.y()))
		{
			if (pt_nodes[i].laser.y() > max_y) max_y = pt_nodes[i].laser.y();
			else if (pt_nodes[i].laser.y() < min_y) min_y = pt_nodes[i].laser.y();
		}
		if (pt_nodes[i].laser.hasNaN()) continue;
		else {

			temp_point.x = pt_nodes[i].laser.x();
			temp_point.y = pt_nodes[i].laser.y();
			temp_point.z = -pt_nodes[i].laser.z();

			intensity.push_back(i);
			temp_cloud.push_back(temp_point);
		}
	}
	std::cout << "[p2cv] frame range:" << max_x << " - " << min_x << " ; " << max_y << " - " << min_y << endl;
	*cloud_in = temp_cloud;
	pcl::getMinMax3D(*cloud_in, min_p, max_p);				// get points max & min range in xyz

	gap_wide = (max_x - min_x) / (cols_count - 1) * compress_x;
	gap_height = (max_y - min_y) / (line_count - 1) * compress_y;
	depth_range = max_p.z - min_p.z;
	std::cout << "[p2cv] depth range:" << depth_range << std::endl;
	if (gap_wide <= 0 || gap_height <= 0 || depth_range <= 0) {
		std::cout << "[error] pt points axis error !" << std::endl;
		return false;
	}
	line_count = (max_p.y - min_p.y) / gap_height + 1;
	cols_count = (max_p.x - min_p.x) / gap_wide + 1;
	Eigen::MatrixXd matrix_depth = Eigen::MatrixXd::Zero(line_count + 100, cols_count + 100);
	matrix_index = Eigen::MatrixXi::Zero(line_count + 100, cols_count + 100);

	std::cout << "[p2cv] point range:" << max_p << min_p << std::endl;
	std::cout << "[p2cv] depth img size:" << cols_count + 1 << " - " << line_count + 1 << std::endl;

	
	Eigen::Affine3f transform_2 = Eigen::Affine3f::Identity();
	/* move points to cameraframe center */
	transform_2.translation() << -min_p.x, -min_p.y, 0.0;
	transform_2.rotate(Eigen::AngleAxisf(0, Eigen::Vector3f::UnitZ())); // 0 deg in rotation as default
	pcl::transformPointCloud(*cloud_in, *cloud_centered, transform_2);

	for (int i = 0; i < cloud_centered->points.size(); i++)
	{
		int pos_x = cloud_centered->points[i].x / gap_wide+50;
		int pos_y = cloud_centered->points[i].y / gap_height+50;
		matrix_depth(pos_y, pos_x) = (cloud_centered->points[i].z - min_p.z - (depth_range * crop_z_min)) / (depth_range * (1- crop_z_min)) * 255;
		//matrix_depth(pos_y, pos_x) = (cloud_centered->points[i].z - min_p.z- crop_z_min) / (depth_range) * 255;
		//matrix_depth(pos_y, pos_x) = (cloud_centered->points[i].z - min_p.z - crop_z_min) / (depth_range) * 32767;
		matrix_index(pos_y, pos_x) = i;
	}
	try {
		cv::Mat result_GE, result_UP;
		cv::Mat img_temp, image_hist, image_hist_x, image_hist_y;
		cv::eigen2cv(matrix_depth, img_temp);


		img_temp.convertTo(image_hist, CV_8UC1);
		cv::medianBlur(image_hist, image_hist, hist_blur_size);
		cv::normalize(image_hist, image_hist, 0, 255, cv::NORM_MINMAX);


		img_temp = image_hist.clone();

		cv::Mat roElement = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 10)); // 内核
		cv::Mat coElement = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3)); // 内核
		//cv::adaptiveThreshold(img_temp, result_UP, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, 15, 6);
		cv::adaptiveThreshold(img_temp, result_UP, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY_INV, hist_threshold_blocksize, 2);
		//cv::erode(result_UP, result_UP, dilateElement);
		result_GE = result_UP.clone();

		cv::Rect ccomp;
		cv::floodFill(result_UP, cv::Point(1, 1), cv::Scalar(255), &ccomp, cv::Scalar(20), cv::Scalar(20));

		cv::bitwise_not(result_UP, result_UP, result_GE);
		cv::threshold(result_UP, result_UP, 100, 255, cv::THRESH_BINARY_INV);

		//cv::erode(result_UP, result_UP, roElement);
		//cv::erode(result_UP, result_UP, roElement);
		//cv::erode(result_UP, result_UP, coElement);

		//cv::erode(result_UP, result_GE, dilateElement);
		//cv::bitwise_not(result_UP, result_UP, result_GE);

		Eigen::MatrixXi depth_mask;
		cv::cv2eigen(result_UP, depth_mask);
		_updatePointsLT(&depth_mask);
		//_updatePointsLT_INV(&depth_mask);
		//getResultLT(40);
		if (debug_model) {
			Vistool* vistool;
			vistool = new Vistool();
			vistool->pcl_viewer.reset(new pcl::visualization::PCLVisualizer("3D Viewer"));
			pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZ> fildColor(cloud_centered, "z");
			pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> high_light(cloud_selected, 255, 255, 255);
			vistool->pcl_viewer->addPointCloud<pcl::PointXYZ>(cloud_centered, fildColor, "centered cloud");
			vistool->pcl_viewer->addPointCloud<pcl::PointXYZ>(cloud_selected, high_light, "trajectory");
			vistool->pcl_viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 8, "trajectory");
			cv::namedWindow("result_UP", CV_WINDOW_NORMAL);
			cv::imshow("result_UP", result_UP);
			cv::namedWindow("image_hist", CV_WINDOW_NORMAL);
			cv::imshow("image_hist", image_hist);
			vistool->pcl_viewer->spin();
			delete vistool;
		}
	}
	catch (cv::Exception& e)
	{
		const char* err_msg = e.what();
		std::cout << err_msg << endl;
		return false;
	}
	return true;
}

bool Reprojector::generateLT(const std::vector<PtFrame>& source_pt_frames, bool direction_x)
{
	PtCloud source_pt_data(source_pt_frames);
	if(!generateLT(source_pt_frames, direction_x)) return false;
	return true;
}
bool Reprojector::generateLTS(const std::vector<PtFrame>& source_pt_frames, bool direction_x, int show_pt_num)
{
	PtCloud source_pt_data(source_pt_frames);
	generateLTS(source_pt_frames, direction_x, show_pt_num);
	return false;
}

std::vector <std::vector<PtRobot>> Reprojector::getResult()
{
	std::vector<PtRobot> temp_group;
	std::vector < std::vector<PtRobot>> result;
	if (cloud_result->points.size() < 1) return result;
	int diff_z = 0.3;
	PtRobot temp_point;
	temp_point.setX(cloud_result->points[0].y + 3);
	temp_point.setY(cloud_result->points[0].x);
	temp_point.setZ(-cloud_result->points[0].z - diff_z);
	temp_point.setRx(0);
	temp_point.setRy(0);
	temp_point.setRz(0);
	temp_group.push_back(temp_point);
	std::cout << "------ |" << -cloud_result->points[0].z - diff_z << " - " << cloud_result->points[0].y + 3 << std::endl;
	for (int i = 0; i < cloud_result->points.size(); i++)
	{
		PtRobot temp_point;
		temp_point.setX(cloud_result->points[i].y);
		temp_point.setY(cloud_result->points[i].x);
		temp_point.setZ(-cloud_result->points[i].z - diff_z);
		temp_point.setRx(0);
		temp_point.setRy(0);
		temp_point.setRz(0);
		temp_group.push_back(temp_point);
		std::cout << "------ |" << -cloud_result->points[i].z - diff_z << " - " << cloud_result->points[i].y << " - " << cloud_result->points[i].x << std::endl;
		//i += gap;
	}
	result.push_back(temp_group);
	std::cout << "[result] retrun points:" << result.size() << std::endl;
	return result;
}

std::vector<PtRobot> Reprojector::getResultLT()
{
	std::vector<PtRobot> result;
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_temp(new pcl::PointCloud<pcl::PointXYZ>);
	cloud_temp->points.clear();
	if (cloud_result->points.size() < 1) return result;
	int gap = 0;
	if (output_points_num < cloud_result->points.size())
		gap = cloud_result->points.size() / output_points_num;

	PtRobot temp_point;
	pcl::PointXYZ point_pre;
	point_pre.x = cloud_result->points[0].x;
	point_pre.y = cloud_result->points[0].y + 2;
	point_pre.z = cloud_result->points[0].z;
	temp_point.setX(point_pre.x);
	temp_point.setY(point_pre.y);
	temp_point.setZ(-point_pre.z);
	temp_point.setRx(0);
	temp_point.setRy(0);
	temp_point.setRz(0);
	result.push_back(temp_point);

	point_pre.x = cloud_selected->points[0].x;
	point_pre.y = cloud_selected->points[0].y + 2;
	point_pre.z = cloud_selected->points[0].z;
	cloud_temp->points.push_back(point_pre);
	int len = cloud_result->points.size();
	for (int i = 0; i < len; i++)
	{
		temp_point.setX(cloud_result->points[i].x);
		temp_point.setY(cloud_result->points[i].y);
		temp_point.setZ(-cloud_result->points[i].z);
		temp_point.setRx(0);
		temp_point.setRy(0);
		temp_point.setRz(0);

		result.push_back(temp_point);
		cloud_temp->points.push_back(cloud_selected->points[i]);

		if (debug_model) 
			std::cout << "[point] :" << -cloud_result->points[i].z <<
			" - " << cloud_result->points[i].x << 
			" - " << cloud_result->points[i].y << 
			" - " << i <<std::endl;
		i += gap;
	}
	point_pre.x = cloud_result->points[len-1].x;
	point_pre.y = cloud_result->points[len-1].y-2;
	point_pre.z = cloud_result->points[len-1].z;
	temp_point.setX(point_pre.x);
	temp_point.setY(point_pre.y);
	temp_point.setZ(-point_pre.z);
	temp_point.setRx(0);
	temp_point.setRy(0);
	temp_point.setRz(0);
	result.push_back(temp_point);

	point_pre.x = cloud_selected->points[len - 1].x;
	point_pre.y = cloud_selected->points[len - 1].y-2;
	point_pre.z = cloud_selected->points[len - 1].z;
	cloud_temp->points.push_back(point_pre);
	//cloud_selected->points.clear();
	//cloud_selected->points.push_back(cloud_temp->points[cloud_temp->points.size()-1]);
	cloud_selected = cloud_temp;

	std::cout << "[result] retrun points:" << result.size() << std::endl;
	return result;
}

std::vector<PtRobot> Reprojector::getResultLTS(const int points_num) 
{
	std::vector<PtRobot> result;
	if (cloud_result->points.size() < 1) return result;
	float pre_z = cloud_result->points[0].z;
	float pre_x = cloud_result->points[0].y;
	int gap = 0;
	if (points_num < cloud_result->points.size())
		gap = cloud_result->points.size() / points_num;
	if (cloud_result->points.size() < 1) return result;
	double y_af = cloud_result->points[1].y;
	for (int i = 0; i < cloud_result->points.size() - 1; i++)
	{
		y_af = cloud_result->points[i + 1].y;
		PtRobot temp_point;
		temp_point.setX(cloud_result->points[i].x);
		temp_point.setY(cloud_result->points[i].y);
		temp_point.setZ(-cloud_result->points[i].z);
		temp_point.setRx(0);
		temp_point.setRy(0);
		temp_point.setRz(0);
		result.push_back(temp_point);
		std::cout << "------" << cloud_result->points[i].z << " - " << cloud_result->points[i].y << " - " << cloud_result->points[i].x << std::endl;
		if (cloud_result->points[i].y - y_af > 10)
		{
			std::cout << "\n";
			break;
		}
		i += gap;
	}
	std::cout << "[result] retrun points:" << result.size() << std::endl;
	return result;

}

void Reprojector::setdebug(const bool value)
{
	debug_model = value;
}

bool Reprojector::iniparams(const std::string & ini_file)
{
	CSimpleIniA inifile;
	std::string config_file = ini_file;
	SI_Error rc = inifile.LoadFile(config_file.data());
	if (rc < 0) return false;
	const char* pv;
	pv = inifile.GetValue("check_config", "debug");
	debug_model = std::stoi(pv);
	pv = inifile.GetValue("check_config", "crop_z_min");
	crop_z_min = std::stof(pv);
	pv = inifile.GetValue("check_config", "compress_x");
	compress_x = std::stof(pv);
	pv = inifile.GetValue("check_config", "compress_y");
	compress_y = std::stof(pv);

	pv = inifile.GetValue("check_config", "hist_blur_size");
	hist_blur_size = std::stof(pv);
	pv = inifile.GetValue("check_config", "winblur_thresh");
	winblur_thresh = std::stof(pv);
	pv = inifile.GetValue("check_config", "zblur_thresh");
	zblur_thresh = std::stof(pv);

	pv = inifile.GetValue("check_config", "margin_x");
	margin_x = std::stof(pv);
	pv = inifile.GetValue("check_config", "margin_y");
	margin_y = std::stof(pv);

	pv = inifile.GetValue("check_config", "use_border");
	use_border = std::stoi(pv);
	pv = inifile.GetValue("check_config", "from_left");
	from_left = std::stoi(pv);
	pv = inifile.GetValue("check_config", "crop_head");
	crop_head = std::stoi(pv);
	pv = inifile.GetValue("check_config", "crop_tail");
	crop_tail = std::stoi(pv);
	pv = inifile.GetValue("check_config", "hist_thresh_size");
	hist_threshold_blocksize = std::stoi(pv);

	pv = inifile.GetValue("check_config", "output_points_num");
	output_points_num = std::stoi(pv);
	return true;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr Reprojector::_updatePoints(Eigen::MatrixXi* depth_mask)
{
	
	cloud_selected.reset(new pcl::PointCloud< pcl::PointXYZ >());
	cloud_result.reset(new pcl::PointCloud< pcl::PointXYZ >());
	cloud_selected->points.clear();
	cloud_result->points.clear();
	std::vector<cv::Point2i> pix_line;
	std::cout << "[p2cv] start get points ... \n";
	for (int ro =  20; ro <= depth_mask->rows() - 20; ro++)
	{
		//bool find = false;
		if (from_left) {
			for (int co = 20; co < depth_mask->cols() - 20; co++)
			{
				if (depth_mask->row(ro).col(co).value() > 50)
				{
					try
					{
						int p_index = matrix_index(ro + margin_y, co + margin_x);
						//std::cout << p_index << std::endl;
						if (p_index > cloud_centered->points.size()-1 || p_index <0)
						{
							throw (-1);
						}
						
						if (p_index > 100) {
							cv::Point2i temp_pix;
							temp_pix.x = co + margin_x;
							temp_pix.y = ro + margin_y;
							pix_line.push_back(temp_pix);
							//find = true;
							break;
						}
					}
					catch (int e)
					{
						std::cout << "[warning] index error" << std::endl;
					}
				}
				//if (!find) co += 3;
			}//ro += 20;
		}
		else {
			for (int co = depth_mask->cols() - 10; co > 10; co--)
			{
				if (depth_mask->row(ro).col(co).value() > 100)
				{
					try
					{
						if (matrix_index(ro + margin_y, co + margin_x) > cloud_centered->points.size())
						{
							throw (-1);
						}
						int p_index = matrix_index(ro + margin_y, co + margin_x);
						if (p_index > 10) {
							cv::Point2i temp_pix;
							temp_pix.x = co + margin_x;
							temp_pix.y = ro + margin_y;
							pix_line.push_back(temp_pix);
							break;
						}
					}
					catch (int e)
					{
						std::cout << "[warning] index error" << std::endl;
					}
				}
				//if (!find) co -= 3;
			}//ro -= 20;
		}
	}
	std::cout << "[p2cv] start winBlur ... \n";
	pix_line = winBlur(pix_line, 0, winblur_thresh);

	if (pix_line.size() < 10) {
		std::cout << "[p2cv] winBlur error: " << cloud_selected->points.size() << std::endl;
		return cloud_result;
	}
	for (int index = crop_head; index < (pix_line.size()- crop_tail); index++)
	{
		int p_index = matrix_index(pix_line[index].y, pix_line[index].x);
		if (p_index > 10 && p_index < cloud_centered->points.size()) {
			cloud_selected->points.push_back(cloud_centered->points[p_index]);
			cloud_result->points.push_back(cloud_in->points[p_index]);
		}
	}
	/*pcl::PointXYZ head_point, tail_point;
	pcl::PointXYZ head_point_res, tail_point_res;
	head_point = cloud_selected->points[2];
	tail_point = cloud_selected->points[cloud_selected->points.size() - 2];

	head_point_res = cloud_result->points[2];
	tail_point_res = cloud_result->points[cloud_result->points.size() - 2];*/
	std::cout << "[p2cv] start StatisticalOutlierRemoval ... \n";

	//pcl::StatisticalOutlierRemoval<pcl::PointXYZ> Statistical;
	//Statistical.setMeanK(15);//取平均值的临近点数
	//Statistical.setStddevMulThresh(10);//临近点数数目少于多少时会被舍弃

	//Statistical.setInputCloud(cloud_selected);
	//Statistical.filter(*cloud_selected);
	//Statistical.setInputCloud(cloud_result);
	//Statistical.filter(*cloud_result);

	if (cloud_result->points.size() < 10) { 
		std::cout << "[p2cv] StatisticalOutlierRemoval error: " << cloud_selected->points.size() << std::endl;
		return cloud_result;
	}

	cloud_selected = pointsFilter(cloud_selected, 2, zblur_thresh);
	cloud_result = pointsFilter(cloud_result, 2, zblur_thresh);

	if (cloud_result->points.size() < 10) {
		std::cout << "[p2cv] zblur error: " << cloud_selected->points.size() << std::endl;
		return cloud_result;
	}
	/*cloud_selected->points[0] = head_point;
	cloud_selected->points.push_back(tail_point);

	cloud_result->points[0] = head_point_res;
	cloud_result->points.push_back(tail_point_res);*/

	std::cout << "[p2cv] LT get points: " << cloud_selected->points.size() << std::endl;

	return cloud_selected;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr Reprojector::_updatePointsLT(Eigen::MatrixXi* depth_mask)
{
	cloud_selected.reset(new pcl::PointCloud< pcl::PointXYZ >());
	cloud_result.reset(new pcl::PointCloud< pcl::PointXYZ >());
	cloud_selected->points.clear();
	cloud_result->points.clear();
	std::vector<cv::Point2i> pix_line;
	for (int ro = depth_mask->rows()-10; ro >= 10; ro--)
	{
		for (int co = 10; co < depth_mask->cols()-10; co++)
		{
			if (depth_mask->row(ro).col(co).value() > 100)
			{
				try
				{
					if (matrix_index(ro + margin_y, co + margin_x) > cloud_centered->points.size())
					{
						throw (-1);
					}
					int p_index = matrix_index(ro + margin_y, co + margin_x);
					if (p_index > 10) {
						cv::Point2i temp_pix;
						temp_pix.x = co + margin_x;
						temp_pix.y = ro + margin_y;
						pix_line.push_back(temp_pix);
						break;
					}
				}
				catch (int e)
				{
					std::cout << "[warning] index error" << std::endl;
				}
			}
		}
		//ro -= 20;
	}

	pix_line = winBlur(pix_line,0, winblur_thresh);

	for (int index = crop_head; index < (pix_line.size() - crop_tail); index++)
	{
		int p_index = matrix_index(pix_line[index].y, pix_line[index].x);
		if (p_index > 10 && p_index < cloud_centered->points.size()) {
			cloud_selected->points.push_back(cloud_centered->points[p_index]);
			cloud_result->points.push_back(cloud_in->points[p_index]);
		}
	}

	cloud_selected = pointsFilter(cloud_selected,2, zblur_thresh);
	cloud_result = pointsFilter(cloud_result,2,zblur_thresh);

	pcl::StatisticalOutlierRemoval<pcl::PointXYZ> Statistical;
	Statistical.setMeanK(10);//取平均值的临近点数
	Statistical.setStddevMulThresh(7);//临近点数数目少于多少时会被舍弃

	Statistical.setInputCloud(cloud_selected);
	Statistical.filter(*cloud_selected);
	Statistical.setInputCloud(cloud_result);
	Statistical.filter(*cloud_result);

	std::cout << "[p2cv] LT get points: " << cloud_selected->points.size() << std::endl;
	
	return cloud_selected;
}