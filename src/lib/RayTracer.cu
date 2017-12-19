#include "RayTracer.hpp"
#include <Eigen/Core>
#include <Eigen/Geometry>

#undef EIGEN_DEFAULT_DENSE_INDEX_TYPE
#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int

#define H_MULT(res_name, mat_ptr, vec)                                         \
    Eigen::Vector3f res_name;                                                  \
    do {                                                                       \
	Eigen::Vector4f h_##res_name(vec[0], vec[1], vec[2], 1);               \
	Eigen::Vector4f res_##res_name = (*mat_ptr) * h_##res_name;            \
	res_name = Eigen::Vector3f(res_##res_name[0] / res_##res_name[3],      \
				   res_##res_name[1] / res_##res_name[3],      \
				   res_##res_name[2] / res_##res_name[3]);     \
    } while (0)

__global__ void filter_faces_kernel(const Eigen::Vector3f *__restrict__ faces,
				    int n_faces, const Eigen::Matrix4f *T,
				    uint8_t *__restrict__ face_mask) {
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    const int stride = gridDim.x * blockDim.x;

    for (int i = idx; i < n_faces; i += stride) {
	bool any_in = true;
	for (int j = 0; j < 3; ++j) {
	    const auto &tmp = faces[i * 3 + j];
	    H_MULT(res, T, tmp);

	    any_in = any_in || (res[0] / res[3] <= -1 && res[0] / res[3] <= 1 &&
				res[1] / res[3] >= -1 && res[1] / res[3] <= 1 &&
				res[2] / res[3] >= -1 && res[2] / res[3] <= 1);
	}
	face_mask[i] = any_in ? 1 : 0;
    }
}

#define CROSS_PRODUCT(res_name, a, b)                                          \
    Eigen::Vector3f res_name(a[1] * b[2] - a[2] * b[1],                        \
			     a[2] * b[0] - a[0] * b[2],                        \
			     a[0] * b[1] - a[1] * b[0])

__device__ bool rayTriangleIntersect(const Eigen::Vector3f &orig,
				     const Eigen::Vector3f &dir,
				     const Eigen::Vector3f &v0,
				     const Eigen::Vector3f &v1,
				     const Eigen::Vector3f &v2, float *t) {
    Eigen::Vector3f edge1 = (v1 - v0);
    Eigen::Vector3f edge2 = (v2 - v0);

    /* Eigen::Vector3f h = dir.cross(edge2); */
    CROSS_PRODUCT(h, dir, edge2);
    float a = edge1.dot(h);
    if (a < 1e-6)
	return false;

    float f = 1.0 / a;
    Eigen::Vector3f s = orig - v0;
    float u = f * s.dot(h);
    if (u < 0.0 || u > 1.0)
	return false;

    /* Eigen::Vector3f q = s.cross(edge1); */
    CROSS_PRODUCT(q, s, edge1);
    float v = f * dir.dot(q);
    if (v < 0.0 || v + u > 1.0)
	return false;

    *t = f * edge2.dot(q);

    return *t > 1e-6;
}

__global__ void ray_tracing_kernel(
    const Eigen::Vector3f *__restrict__ faces, const uint8_t *__restrict__ mask,
    const int *__restrict__ face_cat_id, const int n_faces,
    const Eigen::Matrix4f *view_inv, const Eigen::Vector3f *O,
    const Eigen::Matrix4f *view, const int rows, const int cols,
    const float scale, int *__restrict__ semgseg, float *__restrict__ depth) {
    const int row_idx = blockIdx.x;
    const int stride = blockDim.x;

    for (int col_idx = threadIdx.x; col_idx < cols; col_idx += stride) {
	const int res_idx = row_idx * cols + col_idx;

	const float x = (2.0 * (col_idx + 0.5) / static_cast<float>(cols) - 1) *
			static_cast<float>(cols) / static_cast<float>(rows) *
			scale;
	const float y =
	    (1 - 2.0 * (row_idx + 0.5) / static_cast<float>(rows)) * scale;

	Eigen::Vector3f tmp(x, y, -1);
	tmp.normalize();
	H_MULT(pt, view_inv, tmp);

	/* if (res_idx == (rows / 2 * cols + cols / 2)) {
		printf("%f %f %f\n\n", pt[0], pt[1], pt[2]);
	} */

	Eigen::Vector3f gaze_dir = pt - *O;
	gaze_dir.normalize();

	for (int f_idx = 0; f_idx < n_faces; ++f_idx) {
	    if (mask[f_idx] == 1) {
		int offset = f_idx * 3;
		float t;
		if (rayTriangleIntersect(*O, gaze_dir, faces[offset + 0],
					 faces[offset + 1], faces[offset + 2],
					 &t)) {
		    Eigen::Vector3f hit = *O + t * gaze_dir;
		    H_MULT(c_pt, view, hit);
		    const float d = -c_pt[2];

		    bool closer = d < depth[res_idx] && depth[res_idx] > 0;
		    depth[res_idx] = closer ? d : depth[res_idx];
		    semgseg[res_idx] =
			closer ? face_cat_id[f_idx] : semgseg[res_idx];
		}
	    }
	}
    }
}

#undef EIGEN_DEFAULT_DENSE_INDEX_TYPE
#define EIGEN_DEFAULT_DENSE_INDEX_TYPE std::ptrdiff_t

template <class T>
Eigen::Matrix<T, 4, 4> perspective(double radf, double aspect, double zNear,
				   double zFar) {
    typedef Eigen::Matrix<T, 4, 4> Matrix4;

    assert(aspect > 0);
    assert(zFar > zNear);

    double tanHalfFovy = tan(radf / 2.0);
    Matrix4 res = Matrix4::Zero();
    res(0, 0) = 1.0 / (aspect * tanHalfFovy);
    res(1, 1) = 1.0 / (tanHalfFovy);
    res(2, 2) = -(zFar + zNear) / (zFar - zNear);
    res(3, 2) = -1.0;
    res(2, 3) = -(2.0 * zFar * zNear) / (zFar - zNear);
    return res;
}

void ray_tracing(
    const std::vector<std::shared_ptr<std::array<Eigen::Vector3f, 3>>> &faces,
    const std::vector<int> &face_cat_id, const Eigen::Matrix4f &view,
    const Eigen::Vector3f &O, const int rows, const int cols, const float vfov,
    cv::Mat &semgseg, cv::Mat &depth) {

    const Eigen::Vector3f *cuda_faces;
    cudaMalloc((void **)&cuda_faces,
	       faces.size() * 3 * sizeof(Eigen::Vector3f));
    for (int i = 0; i < faces.size(); ++i) {
	for (int j = 0; j < 3; ++j) {
	    cudaMemcpy((void *)(cuda_faces + i * 3 + j),
		       (void *)(&faces[i]->at(j)), sizeof(Eigen::Vector3f),
		       cudaMemcpyHostToDevice);
	}
    }

    const Eigen::Matrix4f *cuda_T;
    cudaMalloc((void **)&cuda_T, sizeof(Eigen::Matrix4f));

    Eigen::Matrix4f T =
	(perspective<double>(vfov, (double)cols / (double)rows, 0.01, 30) *
	 view.cast<double>())
	    .cast<float>();
    cudaMemcpy((void *)cuda_T, (void *)&T, sizeof(Eigen::Matrix4f),
	       cudaMemcpyHostToDevice);

    uint8_t *face_mask;
    cudaMalloc((void **)&face_mask, faces.size() * sizeof(uint8_t));

    std::cout << "Filtering" << std::endl;
    filter_faces_kernel<<<((faces.size() + 511) / 512) / 4, 512>>>(
	cuda_faces, faces.size(), cuda_T, face_mask);

    const Eigen::Matrix4f *cuda_view;
    cudaMalloc((void **)&cuda_view, sizeof(Eigen::Matrix4f));
    cudaMemcpy((void *)cuda_view, (void *)&view, sizeof(Eigen::Matrix4f),
	       cudaMemcpyHostToDevice);

    const Eigen::Matrix4f *cuda_view_inv;
    cudaMalloc((void **)&cuda_view_inv, sizeof(Eigen::Matrix4f));
    Eigen::Matrix4f view_inv = Eigen::Affine3f(view).inverse().matrix();
    cudaMemcpy((void *)cuda_view_inv, (void *)&view_inv,
	       sizeof(Eigen::Matrix4f), cudaMemcpyHostToDevice);

    const Eigen::Vector3f *cuda_O;
    cudaMalloc((void **)&cuda_O, sizeof(Eigen::Vector3f));
    cudaMemcpy((void *)cuda_O, (void *)&O, sizeof(Eigen::Vector3f),
	       cudaMemcpyHostToDevice);

    int *cuda_semgseg;
    cudaMalloc((void **)&cuda_semgseg, rows * cols * sizeof(int));
    cudaMemset((void *)cuda_semgseg, 0, rows * cols * sizeof(int));

    float *cuda_depth;
    cudaMalloc((void **)&cuda_depth, rows * cols * sizeof(float));
    cudaMemset((void *)cuda_depth, 0, rows * cols * sizeof(float));

    int *cuda_face_cat_id;
    cudaMalloc((void **)&cuda_face_cat_id, face_cat_id.size() * sizeof(int));
    cudaMemcpy((void *)cuda_face_cat_id, (void *)face_cat_id.data(),
	       face_cat_id.size() * sizeof(int), cudaMemcpyHostToDevice);

    ray_tracing_kernel<<<((rows * cols + 511) / 512) / 4, 512>>>(
	cuda_faces, face_mask, cuda_face_cat_id, faces.size(), cuda_view_inv,
	cuda_O, cuda_view, rows, cols, std::tan(vfov / 2.0), cuda_semgseg,
	cuda_depth);

    semgseg.create(rows, cols, CV_32SC1);
    cudaMemcpy((void *)semgseg.data, (void *)cuda_semgseg,
	       rows * cols * sizeof(int), cudaMemcpyDeviceToHost);

    depth.create(rows, cols, CV_32FC1);
    cudaMemcpy((void *)depth.data, (void *)cuda_depth,
	       rows * cols * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree((void *)cuda_faces);
    cudaFree((void *)cuda_view_inv);
    cudaFree((void *)cuda_view);
    cudaFree((void *)cuda_O);
    cudaFree((void *)cuda_semgseg);
    cudaFree((void *)cuda_depth);
    cudaFree((void *)cuda_face_cat_id);
    cudaFree((void *)cuda_T);
    cudaFree((void *)face_mask);
}
