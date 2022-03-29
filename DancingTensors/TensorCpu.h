#pragma once
#include <vector>
#include "KillIf.h"


template<typename T> class TensorCpu {
private:
	std::vector<int> shape;
	std::vector<int> product;
	T* __data;

	void build_product_of_shape();

public:
	TensorCpu(std::vector<int> shape);
	T& v(std::vector<int>position);

	std::vector<int> getShape();
	TensorCpu(const TensorCpu<T>& other);

	TensorCpu<T>& operator = (const TensorCpu<T>& other);
	TensorCpu<T>& operator = (TensorCpu<T>&& other) noexcept;
};

template class TensorCpu<float>;
template class TensorCpu<double>;
template class TensorCpu<long double>;