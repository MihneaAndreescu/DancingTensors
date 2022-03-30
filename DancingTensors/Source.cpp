#include "Tensor.h"
#include "TensorGpu.h"
#include "TensorCpu.h"
#include <iostream>
#include <vector>
#include <string>

using namespace std;

TensorGpu<double> f(TensorGpu<double> guy) {
	return guy;
}

Tensor<double> zol(Tensor<double> shit) {
	return shit;
}

int main() {
	Tensor<double> myTensor(DeviceType::CPU, { 5, 5 });
	Tensor<double> sec(DeviceType::CPU, { 5, 5 });
	//myTensor = sec;
	cout << "done\n";
	exit(0);
	myTensor.v({ 3, 3 }) = 6;
	sec = zol(myTensor);
	sec.v({ 2, 2 }) = 9;
	sec.toDevice(DeviceType::CPU);
	sec.toDevice(DeviceType::GPU);
	for (int i = 0; i < 5; i++) {
		for (int j = 0; j < 5; j++) {
			cout << sec.v({ i, j }) << " ";
		}
		cout << "\n";
	}

	exit(0);
	myTensor.v({ 4, 2 }) = 4;

	for (int i = 0; i < 5; i++) {
		for (int j = 0; j < 3; j++) {
			cout << myTensor.v({ i, j }) << " ";
		}
		cout << "\n";
	}
	exit(0);


	TensorGpu<double> a({ 5, 2 });
	TensorGpu<double> b({ 6, 1 });
	b = f(a);
	exit(0);
	if (1) {
		TensorGpu<double> my_tensor({ 5, 2 });
		my_tensor.v({ 4, 1 }) = 1;
		for (int i = 0; i < 5; i++) {
			for (int j = 0; j < 2; j++) {
				cout << my_tensor.v({ i, j }) << " ";
			}
		}
		cout << "\n";
		TensorGpu<double> second = my_tensor;
		second.v({ 4, 1 }) = 77;
		for (int i = 0; i < 5; i++) {
			for (int j = 0; j < 2; j++) {
				cout << my_tensor.v({ i, j }) << " ";
			}
		}
		cout << "\n";
	}

	if (0) {
		TensorCpu<double> my_tensor({ 5, 2 });
		my_tensor.v({ 4, 1 }) = 1;
		for (int i = 0; i < 5; i++) {
			for (int j = 0; j < 2; j++) {
				cout << my_tensor.v({ i, j }) << " ";
			}
		}
		cout << "\n";

		TensorCpu<double> second = my_tensor;
		second.v({ 4, 0 }) = -1;

		for (int i = 0; i < 5; i++) {
			for (int j = 0; j < 2; j++) {
				cout << my_tensor.v({ i, j }) << " ";
			}
		}
		cout << "\n";
	}
}