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

template<typename T> T L2Loss(TensorGpu<T> a, TensorGpu<T> b) {
	killIf(a.getShape() != b.getShape(), "can't compute the loss if their shape is not equal");
	if (a.shape.empty()) return 0;
	T loss = 0;
	for (int i = 0; i < a.product[0]; i++) loss += (a.__data_helper.getDataAtPosition(i) - a.__data_helper.getDataAtPosition(i)) * (a.__data_helper.getDataAtPosition(i) - a.__data_helper.getDataAtPosition(i));
	return loss;
}

template<typename T> TensorGpu<T> getL2LossDerivative(TensorGpu<T> a, TensorGpu<T> b) {
	killIf(a.getShape() != b.getShape(), "can't compute the loss if their shape is not equal");
	if (a.shape.empty()) return a;
	for (int i = 0; i < a.product[0]; i++) a.__data_helper.getDataAtPosition(i) = 2 * (a.__data_helper.getDataAtPosition(i) - b.__data_helper.getDataAtPosition(i));
	return a;
}

template class TensorGpu<float>;
template class TensorGpu<double>;
template class TensorGpu<long double>;