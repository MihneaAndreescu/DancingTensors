#pragma once
#include "cuda_runtime.h" // do I need those?
#include "device_launch_parameters.h" // do I need those? 

#include <stdio.h>
#include <vector>

template<typename T> class TensorGpuDataHelper {
private:
	bool hasData = false;
	T* __data;
public:
	void kill() {

		hasData = false;
		__data = nullptr;
	}
	void freeMyData() {
		if (!hasData) return;
		cudaFree(__data);
		hasData = false;
	}
	void requestDataChunk(int dim) {
		if (hasData) cudaFree(__data);

		if (!dim) {
			hasData = false;
			return;
		}

		cudaMallocManaged(&__data, dim * sizeof(T));
		hasData = true;
	}
	void requestAndFillWithZeroesDataChunk(int dim) {
		if (hasData) cudaFree(__data);

		if (!dim) {
			hasData = false;
			return;
		}

		cudaMallocManaged(&__data, dim * sizeof(T));
		hasData = true;

		for (int i = 0; i < dim; i++) {
			__data[i] = 0;
		}
		hasData = true;
	}
	T& getDataAtPosition(int position) {
		return __data[position];
	}
	void copyData(TensorGpuDataHelper<T> auxDataHelper, int dim) {
		for (int i = 0; i < dim; i++) {
			__data[i] = auxDataHelper.__data[i];
		}
	}
};

namespace Cuda {
	void wrapper(void);
}