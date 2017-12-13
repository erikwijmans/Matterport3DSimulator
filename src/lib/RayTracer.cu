#include "RayTracer.hpp"
#include <Eigen/Core>

#undef EIGEN_DEFAULT_DENSE_INDEX_TYPE
#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int

#define CROSS_PRODUCT(res_name, a, b)                                          \
	Eigen::Vector3f res_name(a[1] * b[2] - a[2] * b[1],                    \
				 a[2] * b[0] - a[0] * b[2],                    \
				 a[0] * b[1] - a[1] * b[0])

__device__ bool rayTriangleIntersect(const Eigen::Vector3f &orig,
				     const Eigen::Vector3f &dir,
				     const Eigen::Vector3f &v0,
				     const Eigen::Vector3f &v1,
				     const Eigen::Vector3f &v2) {
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

	float t = f * edge2.dot(q);

	return t > 1e-6;
}

__global__ void ray_tracing_kernel(const Eigen::Vector3f *__restrict__ faces,
				   const int n_faces,
				   const Eigen::Matrix4f *view_inv,
				   const Eigen::Vector3f *O, const int rows,
				   const int cols, const float scale,
				   uint8_t *__restrict__ hit_map) {
	const int row_idx = blockIdx.x;
	const int stride = blockDim.x;
	for (int col_idx = threadIdx.x; col_idx < cols; col_idx += stride) {
		const int res_idx = row_idx * cols + col_idx;
		hit_map[res_idx] = 0;

		const float x =
		    (2.0 * (col_idx + 0.5) / static_cast<float>(cols) - 1) *
		    static_cast<float>(cols) / static_cast<float>(rows) * scale;
		const float y =
		    (1 - 2.0 * (row_idx + 0.5) / static_cast<float>(rows)) *
		    scale;

		Eigen::Vector3f tmp(x, y, -1);
		tmp.normalize();
		Eigen::Vector4f h_pt(tmp[0], tmp[1], tmp[2], 1);
		Eigen::Vector4f res = (*view_inv) * h_pt;
		Eigen::Vector3f pt(res[0] / res[3], res[1] / res[3],
				   res[2] / res[3]);

		/* if (res_idx == (rows / 2 * cols + cols / 2)) {
			printf("%f %f %f\n\n", pt[0], pt[1], pt[2]);
		} */

		Eigen::Vector3f gaze_dir = 10 * (pt - *O);
		gaze_dir.normalize();

		for (int f_idx = 0; f_idx < n_faces && (hit_map[res_idx] == 0);
		     ++f_idx) {
			int offset = f_idx * 3;
			if (rayTriangleIntersect(
				*O, gaze_dir, faces[offset + 0],
				faces[offset + 1], faces[offset + 2])) {
				hit_map[res_idx] = 1;
			}
		}
	}
}

#undef EIGEN_DEFAULT_DENSE_INDEX_TYPE
#define EIGEN_DEFAULT_DENSE_INDEX_TYPE std::ptrdiff_t

void ray_tracing(const std::vector<std::array<Eigen::Vector3f, 3>> &faces,
		 const Eigen::Matrix4f &view_inv, const Eigen::Vector3f &O,
		 const int rows, const int cols, const float scale,
		 cv::Mat &hit_map) {

	const Eigen::Vector3f *cuda_faces;
	cudaMalloc((void **)&cuda_faces,
		   faces.size() * 3 * sizeof(Eigen::Vector3f));
	for (int i = 0; i < faces.size(); ++i) {
		for (int j = 0; j < 3; ++j) {
			cudaMemcpy((void *)(cuda_faces + i * 3 + j),
				   (void *)(&faces[i][j]),
				   sizeof(Eigen::Vector3f),
				   cudaMemcpyHostToDevice);
		}
	}

	const Eigen::Matrix4f *cuda_view_inv;
	cudaMalloc((void **)&cuda_view_inv, sizeof(Eigen::Matrix4f));
	cudaMemcpy((void *)cuda_view_inv, (void *)&view_inv,
		   sizeof(Eigen::Matrix4f), cudaMemcpyHostToDevice);
	const Eigen::Vector3f *cuda_O;
	cudaMalloc((void **)&cuda_O, sizeof(Eigen::Vector3f));
	cudaMemcpy((void *)cuda_O, (void *)&O, sizeof(Eigen::Vector3f),
		   cudaMemcpyHostToDevice);

	uint8_t *cuda_hit_map;
	cudaMalloc((void **)&cuda_hit_map, rows * cols * sizeof(uint8_t));

	ray_tracing_kernel<<<rows, 512>>>(cuda_faces, faces.size(),
					  cuda_view_inv, cuda_O, rows, cols,
					  scale, cuda_hit_map);

	hit_map.create(rows, cols, CV_8UC1);
	cudaMemcpy((void *)hit_map.data, (void *)cuda_hit_map,
		   rows * cols * sizeof(uint8_t), cudaMemcpyDeviceToHost);

	cudaFree((void *)cuda_faces);
	cudaFree((void *)cuda_view_inv);
	cudaFree((void *)cuda_O);
	cudaFree((void *)cuda_hit_map);
}
