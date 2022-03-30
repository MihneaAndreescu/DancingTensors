#include <vector>
#include "KillIf.h"
#include "TensorCpu.h"
#include "TensorGpu.h"
#include "Tensor.h"



template<typename T> void Tensor<T>::setCurrentTensorToZeroes() {
	if (device == DeviceType::CPU) {
		tensorCpu.setCurrentTensorToZeroes();
		return;
	}
	if (device == DeviceType::GPU) {
		tensorGpu.setCurrentTensorToZeroes();
		return;
	}
	killIf(true, "device type not supported");
}

template<typename T> void Tensor<T>::setNormalDistribution(T low, T high) {
	if (device == DeviceType::CPU) {
		tensorCpu.setNormalDistribution(low, high);
		return;
	}
	if (device == DeviceType::GPU) {
		tensorGpu.setNormalDistribution(low, high);
		return;
	}
	killIf(true, "device type not supported");
}

template<typename T> DeviceType Tensor<T>::getDeviceType() {
	return device;
}
template<typename T> Tensor<T>::Tensor(std::vector<int> shape, DeviceType device) :
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

template<typename T> T& Tensor<T>::v(std::vector<int>position) {
	if (device == DeviceType::CPU) {
		return tensorCpu.v(position);
	}
	if (device == DeviceType::GPU) {
		return tensorGpu.v(position);
	}
	killIf(true, "device type not supported");
}

template<typename T> std::vector<int> Tensor<T>::getShape() {
	if (device == DeviceType::CPU) {
		return tensorCpu.getShape();
	}
	if (device == DeviceType::GPU) {
		return tensorGpu.getShape();
	}
	killIf(true, "device type not supported");
}
template<typename T> Tensor<T>::Tensor(const Tensor<T>& other) :
	device(other.device),
	tensorCpu(other.tensorCpu),
	tensorGpu(other.tensorGpu) {

}

template<typename T> Tensor<T>& Tensor<T>::operator = (const Tensor<T>& other) {
	if (this == &other) return *this;

	tensorCpu.kill();
	tensorGpu.kill();

	device = other.device;
	tensorCpu = other.tensorCpu;
	tensorGpu = other.tensorGpu;

	return *this;
}

template<typename T> Tensor<T>& Tensor<T>::operator = (Tensor<T>&& other) noexcept { // doubt here
	if (this == &other) return *this;

	device = std::move(other.device);
	tensorCpu = std::move(other.tensorCpu);
	tensorGpu = std::move(other.tensorGpu);

	std::cout << "done\n";
	return *this;
}
template<typename T> void Tensor<T>::toDevice(DeviceType newDevice) {
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

template<typename T> void Tensor<T>::fillWithZeroes(std::vector<int> shape) {
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
