#pragma once
#include <vector>
#include "KillIf.h"
#include "TensorGpuDataHelper.cuh"

template<typename T> class TensorGpu {
public:
	std::vector<int> shape;
	std::vector<int> product;
	TensorGpuDataHelper<T> __data_helper;
	void build_product_of_shape();
public:
	void setCurrentTensorToZeroes();
	void setNormalDistribution(T low, T high);
	TensorGpu(std::vector<int> shape);

	void fillWithZeroes(std::vector<int> shape);

	T& v(std::vector<int>position);

	std::vector<int> getShape();
	TensorGpu(const TensorGpu<T>& other);

	TensorGpu<T>& operator = (const TensorGpu<T>& other);
	TensorGpu<T>& operator = (TensorGpu<T>&& other) noexcept;
	void kill();
};

template class TensorGpu<float>;
template class TensorGpu<double>;
template class TensorGpu<long double>;