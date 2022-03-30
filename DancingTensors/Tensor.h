#pragma once
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
	DeviceType getDeviceType();

	Tensor(DeviceType device, std::vector<int> shape);
	Tensor(const Tensor<T>& other);

	T& v(std::vector<int>position);
	std::vector<int> getShape();
	
	Tensor<T>& operator = (const Tensor<T>& other);
	Tensor<T>& operator = (Tensor<T>&& other); // my code is wrong here
	
	void toDevice(DeviceType newDevice);
};

template class Tensor<float>;
template class Tensor<double>;
template class Tensor<long double>;