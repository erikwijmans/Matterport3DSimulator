#pragma once
#ifndef _RAY_TRACER_HPP
#define _RAY_TRACER_HPP

#include <memory>

#include <Eigen/Core>
#include <opencv2/opencv.hpp>

void ray_tracing(
    const std::vector<std::shared_ptr<std::array<Eigen::Vector3f, 3>>> &faces,
    const std::vector<int> &face_cat_id, const Eigen::Matrix4f &view,
    const Eigen::Vector3f &O, const int rows, const int cols, const float scale,
    cv::Mat &semgseg, cv::Mat &depth);

#endif
