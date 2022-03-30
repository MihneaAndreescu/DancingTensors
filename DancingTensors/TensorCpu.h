#pragma once
#include <vector>
#include "KillIf.h"


template<typename T> class TensorCpu {
public:
	std::vector<int> shape;
	std::vector<int> product;
	T* __data;

	void build_product_of_shape();

public:
	void setCurrentTensorToZeroes();
	void setNormalDistribution(T low, T high);

	void fillWithZeroes(std::vector<int> shape);

	TensorCpu(std::vector<int> shape);
	T& v(std::vector<int>position);

	std::vector<int> getShape();
	TensorCpu(const TensorCpu<T>& other);

	TensorCpu<T>& operator = (const TensorCpu<T>& other);
	TensorCpu<T>& operator = (TensorCpu<T>&& other) noexcept;

	void kill();
};

template<typename T> T L2Loss(TensorCpu<T> a, TensorCpu<T> b) {
	killIf(a.getShape() != b.getShape(), "can't compute the loss if their shape is not equal");
	if (a.shape.empty()) return 0;
	T loss = 0;
	for (int i = 0; i < a.product[0]; i++) loss += (a.__data[i] - b.__data[i]) * (a.__data[i] - b.__data[i]);
	return loss;
}

template<typename T> TensorCpu<T> getL2LossDerivative(TensorCpu<T> a, TensorCpu<T> b) {
	killIf(a.getShape() != b.getShape(), "can't compute the loss if their shape is not equal");
	if (a.shape.empty()) return a;
	for (int i = 0; i < a.product[0]; i++) a.__data[i] = 2 * (a.__data[i] - b.__data[i]);
	return a;
}

template class TensorCpu<float>;
template class TensorCpu<double>;
template class TensorCpu<long double>;