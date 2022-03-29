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
	DeviceType getDeviceType() {
		return device;
	}
	Tensor(DeviceType device, std::vector<int> shape) :
		tensorCpu({}),
		tensorGpu({}),
		device(device) {
		if (device == DeviceType::CPU) {
			tensorCpu.fillWithZeroes(shape);
			return;
		}
		if (device == DeviceType::GPU) {
			tensorGpu.fillWithZeroes(shape);
			return;
		}
		killIf(true, "device type not supported");
	}

	T& v(std::vector<int>position) {
		if (device == DeviceType::CPU) {
			return tensorCpu.v(position);
		}
		if (device == DeviceType::GPU) {
			return tensorGpu.v(position);
		}
		killIf(true, "device type not supported");
	}

	std::vector<int> getShape() {
		if (device == DeviceType::CPU) {
			return tensorCpu.getShape();
		}
		if (device == DeviceType::GPU) {
			return tensorGpu.getShape();
		}
		killIf(true, "device type not supported");
	}
	Tensor(const Tensor<T>& other) :
		device(other.device),
		tensorCpu(other.tensorCpu),
		tensorGpu(other.tensorGpu) {

	}

	Tensor<T>& operator = (const Tensor<T>& other) {
		
		tensorCpu.kill();
		tensorGpu.kill();

		device = other.device;
		tensorCpu = other.tensorCpu;
		tensorGpu = other.tensorGpu;

		return *this;
	}
	Tensor<T>& operator = (Tensor<T>&& other) noexcept { // doubt here
		device = other.device;
	
		tensorCpu = other.tensorCpu;
		tensorGpu = other.tensorGpu;
		return *this;
	}
	void toDevice(DeviceType newDevice) {
		if (device == newDevice) return;
		if (device == DeviceType::CPU) {
			if (newDevice == DeviceType::GPU) {
				tensorGpu.shape = tensorCpu.shape;
				tensorGpu.product = tensorCpu.product;
				tensorGpu.__data_helper.kill();
				if (!tensorGpu.shape.empty()) {
					tensorGpu.__data_helper.requestDataChunk(tensorGpu.product[0]);
					for (int i = 0; i < tensorGpu.product[0]; i++) {
						tensorGpu.__data_helper.getDataAtPosition(i) = tensorCpu.__data[i];
					}
				}
				tensorCpu.kill();
				
				device = newDevice;
				return;
			}
			killIf(true, "invalid device chane type");
		}
		if (device == DeviceType::GPU) {
			if (newDevice == DeviceType::CPU) {
				tensorCpu.shape = tensorGpu.shape;
				tensorCpu.product = tensorGpu.product;
				tensorCpu.kill();
				if (!tensorCpu.shape.empty()) {
					tensorCpu.fillWithZeroes(tensorCpu.shape);
					for (int i = 0; i < tensorGpu.product[0]; i++) {
						tensorCpu.__data[i] = tensorGpu.__data_helper.getDataAtPosition(i);
					}
				}
				tensorGpu.kill();

				device = newDevice;
				return;
			}
			killIf(true, "invalid device chane type");
		}
		killIf(true, "invalid device chane type");
	}
};

template class Tensor<float>;
template class Tensor<double>;
template class Tensor<long double>;