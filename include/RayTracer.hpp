#pragma once
#ifndef _RAY_TRACER_HPP
#define _RAY_TRACER_HPP

#include <Eigen/Core>
#include <opencv2/opencv.hpp>

void ray_tracing(const std::vector<std::array<Eigen::Vector3f, 3>> &faces,
                 const Eigen::Matrix4f &view_inv, const Eigen::Vector3f &O,
                 const int rows, const int cols, const float scale,
                 cv::Mat &hit_map);

#endif
