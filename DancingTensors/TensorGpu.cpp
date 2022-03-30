#include "TensorGpu.h"
#include "TensorGpuDataHelper.cuh"
#include <vector>
#include <iostream>
#include <random>

template<typename T> void TensorGpu<T>::setCurrentTensorToZeroes() {
	if (shape.empty()) return;
	for (int i = 0; i < product[i]; i++) {
		__data_helper.__data[i] = 0;
	}
}

template<typename T> void TensorGpu<T>::setNormalDistribution(T low, T high) {
	std::mt19937 rng(0);

	if (shape.empty()) return;
	std::normal_distribution<T> distribution(low, high);
	for (int i = 0; i < product[i]; i++) {
		__data_helper.__data[i] = distribution(rng);
	}
}
template<typename T> void TensorGpu<T>::build_product_of_shape() {
	if (shape.empty()) {
		product.clear();
		return;
	}
	product.resize((int)shape.size());
	product.back() = shape.back();
	for (int i = (int)shape.size() - 2; i >= 0; i--) {
		product[i] = product[i + 1] * shape[i];
	}
}


template<typename T> void TensorGpu<T>::fillWithZeroes(std::vector<int> newShape) {
	if (!shape.empty()) __data_helper.kill();
	shape = newShape;
	build_product_of_shape();
	if (shape.empty()) return;
	__data_helper.requestAndFillWithZeroesDataChunk(product[0]);
}

template<typename T> TensorGpu<T>::TensorGpu(std::vector<int> shape) : shape(shape) {
	build_product_of_shape();
	if (shape.empty()) return;
	__data_helper.requestAndFillWithZeroesDataChunk(product[0]);

}

template<typename T> T& TensorGpu<T>::v(std::vector<int>position) {
	killIf((int)position.size() != (int)shape.size(), "The shape of the positional argument isn't equal to the shape of the TensorGpu");
	killIf(position.empty(), "You can't get the value of an empty TensorGpu");

	int my_position = 0;

	for (int i = 0; i < (int)shape.size() - 1; i++) {
		killIf(position[i] < 0 || position[i] >= shape[i], "Positional argument is outside the range of the TensorGpu");
		my_position += position[i] * product[i + 1];
	}

	killIf(position.back() < 0 || position.back() >= shape.back(), "Positional argument is outside the range of the TensorGpu");
	my_position += position.back();

	killIf(my_position < 0 || my_position >= product[0], "My TensorGpu code is wrong!!!!!!!!!!!!!!!!");
	return __data_helper.getDataAtPosition(my_position);
}

template<typename T> std::vector<int> TensorGpu<T>::getShape() {
	return shape;
}


template<typename T> TensorGpu<T>::TensorGpu(const TensorGpu<T>& other) {
	shape = other.shape;
	product = other.product;

	if (shape.empty()) return;

	__data_helper.requestDataChunk(product[0]);

	__data_helper.copyData(other.__data_helper, product[0]);

}

template<typename T> TensorGpu<T>& TensorGpu<T>::operator = (const TensorGpu<T>& other) {
	if (this == &other) return *this;
	shape = other.shape;
	product = other.product;
	__data_helper.freeMyData();
	if (shape.empty()) return *this;

	__data_helper.requestDataChunk(product[0]);
	__data_helper.copyData(other.__data_helper, product[0]);

	return *this;
}

template<typename T>TensorGpu<T>& TensorGpu<T>::operator = (TensorGpu<T>&& other) noexcept {
	if (this == &other) return *this;

	bool is_empty = other.shape.empty();

	shape = std::exchange(other.shape, {});
	product = std::exchange(other.product, {});


	__data_helper.freeMyData();

	if (!is_empty) __data_helper.requestDataChunk(product[0]);
	if (!is_empty) __data_helper.__data = std::exchange(other.__data_helper.__data, nullptr);

	if (!is_empty) other.__data_helper.kill();

	return *this;
}

template<typename T> void TensorGpu<T>::kill() {
	if (!shape.empty()) __data_helper.kill();
	shape = {};
	product = {};
}