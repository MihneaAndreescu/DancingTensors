#pragma once
#include <vector>
#include "KillIf.h"
#include "TensorCpu.h"
#include "TensorGpu.h"

enum class DeviceType { CPU, GPU, TPU };

template<typename T> class ITensor {

};

template<typename T> class Tensor { // refactory cu pointeri classITensor
public:
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

template<typename T> T L2Loss(Tensor<T> a, Tensor<T> b) {
	killIf(a.getShape() != b.getShape(), "can't compute the loss if their shape is not equal");
	killIf(a.getDeviceType() != b.getDeviceType(), "can't compute the loss if they aren't on the same device");

	if (a.getDeviceType() == DeviceType::CPU) return L2Loss(a.tensorCpu, b.tensorCpu);
	if (a.getDeviceType() == DeviceType::GPU) return L2Loss(a.tensorGpu, b.tensorGpu);

	killIf(true, "this device isn't supported yet");
}

template<typename T> Tensor<T> getL2LossDerivative(Tensor<T> a, Tensor<T> b) {
	killIf(a.getShape() != b.getShape(), "can't compute the loss if their shape is not equal");
	killIf(a.getDeviceType() != b.getDeviceType(), "can't compute the loss if they aren't on the same device");

	if (a.getDeviceType() == DeviceType::CPU) {
		a.tensorCpu = getL2LossDerivative(a.tensorCpu, b.tensorCpu);
		return a;
	}
	if (a.getDeviceType() == DeviceType::GPU) {
		a.tensorGpu = getL2LossDerivative(a.tensorGpu, b.tensorGpu);
		return a;
	}
	killIf(true, "this device isn't supported yet");
}

template class Tensor<float>;
template class Tensor<double>;
template class Tensor<long double>;