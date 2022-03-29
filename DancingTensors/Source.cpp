#include "TensorCpu.h"
#include <iostream>
#include <vector>
#include <string>

using namespace std;

int main() {
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