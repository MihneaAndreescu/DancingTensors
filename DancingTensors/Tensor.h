#pragma once
#include <vector>
#include "KillIf.h"
#include "TensorCpu.h"
#include "TensorGpu.h"

enum class DeviceType { CPU, GPU, TPU };

template<typename T> class Tensor {
private:
	DeviceType device;
	TensorCpu<T> tensorCpu;
	TensorGpu<T> tensorGpu;

public:

	void setCurrentTensorToZeroes();

	void setNormalDistribution(T low, T high);

	DeviceType getDeviceType();

	Tensor(std::vector<int> shape, DeviceType device);

	T& v(std::vector<int>position);

	std::vector<int> getShape();

	Tensor(const Tensor<T>& other);

	Tensor<T>& operator = (const Tensor<T>& other);

	Tensor<T>& operator = (Tensor<T>&& other) noexcept;

	void toDevice(DeviceType newDevice);

	void fillWithZeroes(std::vector<int> shape);

};

template class Tensor<float>;
template class Tensor<double>;
template class Tensor<long double>;