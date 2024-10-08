#include "Tensor.h"
#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <chrono>
#include <cassert>
#include <iomanip>
#include <map>

using namespace std;

std::ostream& operator<< (std::ostream& os, vector<int> v) {
	os << "{ ";
	for (int i = 0; i < (int)v.size(); i++) {
		os << v[i];
		if (i + 1 < (int)v.size()) os << ", ";
	}
	os << " }";
	return os;
}

mt19937 rng(0);

template<typename T> class BoxInterface {
public:
	virtual DeviceType getDeviceType() = 0;
	virtual Tensor<T> forward(Tensor<T> x) = 0;
	virtual Tensor<T> backProp(Tensor<T> yd) = 0;
	virtual void gradientDescentSGD(T lr) = 0;
	virtual void resetDerivatives() = 0;
	virtual void setNormalDistribution(T low, T high) = 0;
};

template<typename T> pair<T, Tensor<T>> evalL2Loss(Tensor<T> a, Tensor<T> b) {
	T loss = L2Loss(a, b);
	Tensor<T> ad = getL2LossDerivative(a, b);
	return { loss, ad };
}

template<typename T> class Sequential : public BoxInterface<T> {
private:
	vector<BoxInterface*> boxes;
	DeviceType device;
public:
	
	DeviceType getDeviceType() {
		return device;
	}

	Sequential(vector<BoxInterface*> boxes, DeviceType device) :
		boxes(boxes),
		device(device) {
	}
	

	Tensor<T> forward(Tensor<T> x) override {
		for (auto& box : boxes) {
			x = box->forward(x);
		}
		return x;
	}

	Tensor<T> backProp(Tensor<T> yd) override {
		for (int i = (int)boxes.size() - 1; i >= 0; i--) {
			yd = boxes[i]->backProp(yd);
		}
		return yd;
	}
	void gradientDescentSGD(T lr) override {
		for (auto& box : boxes) {
			box->gradientDescentSGD(lr);
		}
	}
	void resetDerivatives() override {
		for (auto& box : boxes) {
			box->resetDerivatives();
		}
	}

	void setNormalDistribution(T low, T high) override {
		for (auto& box : boxes) {
			box->setNormalDistribution(low, high);
		}
	}

};

template<typename T> class DotVectorMatrixBox : public BoxInterface<T> {
private:
	int input_dim;
	int output_dim;

	bool compute_gradients;
	DeviceType device;

	Tensor<T> weights;
	Tensor<T> weightsd;
	Tensor<T> copy_of_input;

public:

	DeviceType getDeviceType() override { return device; }


	DotVectorMatrixBox(int input_dim, int output_dim, bool compute_gradients, DeviceType device) :
		input_dim(input_dim),
		output_dim(output_dim),
		compute_gradients(compute_gradients),
		device(device),
		weights({ input_dim, output_dim }, device),
		copy_of_input({}, device),
		weightsd({}, device) {
		if (compute_gradients) {
			weightsd.fillWithZeroes({ input_dim, output_dim });
		}
	}

	Tensor<T> forward(Tensor<T> x) override {
		vector<int> x_shape = x.getShape();
		killIf((int)x_shape.size() != 2, "wrong shape");
		int batch_size = x_shape[0];
		killIf(x_shape[1] != input_dim, "wrong shape");

		Tensor<T> y({ batch_size, output_dim }, x.getDeviceType());

		if (!compute_gradients) {
			for (int batch = 0; batch < batch_size; batch++) {
				for (int i = 0; i < input_dim; i++) {
					for (int j = 0; j < output_dim; j++) {
						y.v({ batch, j }) += x.v({ batch, i }) * weights.v({ batch, i });
					}
				}
			}
		}
		else {
			copy_of_input = x;
			for (int batch = 0; batch < batch_size; batch++) {
				for (int i = 0; i < input_dim; i++) {
					for (int j = 0; j < output_dim; j++) {
						y.v({ batch, j }) += x.v({ batch, i }) * weights.v({ i, j });
					}
				}
			}
		}

		return y;
	}

	Tensor<T> backProp(Tensor<T> yd) override {

		killIf(!compute_gradients, "come on man, are you serious??? U told me not to compute gradients and now u wanna do backprop?");

		vector<int> yd_shape = yd.getShape();
		killIf((int)yd_shape.size() != 2, "wrong shape");
		int batch_size = yd_shape[0];
		killIf(yd_shape[1] != output_dim, "wrong shape");

		assert(batch_size == copy_of_input.getShape()[0]);

		Tensor<T> xd({ batch_size, input_dim }, device);

		for (int batch = 0; batch < batch_size; batch++) {
			for (int i = 0; i < input_dim; i++) {
				for (int j = 0; j < output_dim; j++) {

					xd.v({ batch, i }) += yd.v({ batch, j }) * weights.v({ i, j });
					weightsd.v({ i, j }) += yd.v({ batch, j }) * copy_of_input.v({ batch, i });
				}
			}
		}

		return xd;
	}
	void gradientDescentSGD(T lr) override {

		killIf(!compute_gradients, "come on man, are you serious??? U told me not to compute gradients and now u wanna do gradient descent?");

		for (int i = 0; i < input_dim; i++) {
			for (int j = 0; j < output_dim; j++) {
				weights.v({ i, j }) -= weightsd.v({ i, j }) * lr;
			}
		}
	}
	void resetDerivatives() override {
		weightsd.setCurrentTensorToZeroes();
		//weightsd.setCurrentTensorToZeroes();
	}

	void setNormalDistribution(T low, T high) override {
		weights.setNormalDistribution(low, high);
	}

};

template<typename T> class BiasVectorBox : public BoxInterface<T> {
private:
	int dim;
	
	bool compute_gradients;
	DeviceType device;

	Tensor<T> biases;
	Tensor<T> biasesd;
	Tensor<T> copy_of_input;

public:

	DeviceType getDeviceType() override { return device; }

	BiasVectorBox(int dim, bool compute_gradients, DeviceType device) :
		dim(dim),
		compute_gradients(compute_gradients),
		device(device),
		biases({ dim }, device),
		copy_of_input({}, device),
		biasesd({}, device) {
		if (compute_gradients) {
			biasesd.fillWithZeroes({ dim });
		}
	}

	Tensor<T> forward(Tensor<T> x) override {
		// optimize gpu, gpu
		vector<int> x_shape = x.getShape();
		killIf((int)x_shape.size() != 2, "wrong shape");
		int batch_size = x_shape[0];
		killIf(x_shape[1] != dim, "wrong shape");

		Tensor<T> y({ batch_size, dim }, x.getDeviceType());

		if (!compute_gradients) {
			for (int batch = 0; batch < batch_size; batch++) {
				for (int i = 0; i < dim; i++) {
					y.v({ batch, i }) += x.v({ batch, i }) + biases.v({ i });
				}
			}
		}
		else {
			copy_of_input = x;
			for (int batch = 0; batch < batch_size; batch++) {
				for (int i = 0; i < dim; i++) {
					y.v({ batch, i }) += x.v({ batch, i }) + biases.v({ i });
				}
			}
		}

		return y;
	}

	Tensor<T> backProp(Tensor<T> yd) override {

		killIf(!compute_gradients, "come on man, are you serious??? U told me not to compute gradients and now u wanna do backprop?");

		vector<int> yd_shape = yd.getShape();
		killIf((int)yd_shape.size() != 2, "wrong shape");
		int batch_size = yd_shape[0];
		killIf(yd_shape[1] != dim, "wrong shape");

		assert(batch_size == copy_of_input.getShape()[0]);

		Tensor<T> xd({ batch_size, dim }, device);

		for (int batch = 0; batch < batch_size; batch++) {
			for (int i = 0; i < dim; i++) {

				biasesd.v({ i }) += yd.v({ batch, i });
				xd.v({ batch, i }) += yd.v({ batch, i });
			}
		}

		return xd;
	}
	void gradientDescentSGD(T lr) override {

		killIf(!compute_gradients, "come on man, are you serious??? U told me not to compute gradients and now u wanna do gradient descent?");

		for (int i = 0; i < dim; i++) {
			biases.v({ i }) += biasesd.v({ i }) * lr;
		}
	}
	void resetDerivatives() override {
		biasesd.setCurrentTensorToZeroes();
	}

	void setNormalDistribution(T low, T high) override {
		biases.setNormalDistribution(low, high);
	}

};

int main() {
	if (1) {
		
		DotVectorMatrixBox<double> b1(10, 20, true, DeviceType::CPU);
		BiasVectorBox<double> b2(20, true, DeviceType::CPU);

		BoxInterface<double>* g1 = &b1;
		BoxInterface<double>* g2 = &b2;

		
		Sequential<double> seq({new DotVectorMatrixBox<double>(10, 20, true, DeviceType::CPU), g2}, DeviceType::CPU);
		seq.setNormalDistribution(0, 1);

		Tensor<double> input({ 4, 10 }, DeviceType::CPU);
		input.setNormalDistribution(0, 1);

		Tensor<double> want({ 4, 20 }, DeviceType::CPU);
		want.setNormalDistribution(0, 1);

		for (int t = 1; 1; t++) {
			Tensor<double> output = seq.forward(input);

			cout << fixed << setprecision(10) << "loss = " << evalL2Loss<double>(output, want).first << "\n";
			seq.backProp(evalL2Loss(output, want).second);
			seq.gradientDescentSGD(5e-2);
			seq.resetDerivatives();
		}

		exit(0);
	}

	if (0) {
		BiasVectorBox<double> biasbox(10, true, DeviceType::CPU);

		exit(0);
	}
	DotVectorMatrixBox<double> learner(10, 20, true, DeviceType::CPU);
	learner.setNormalDistribution(0, 1);

	Tensor<double> input({ 4, 10 }, DeviceType::CPU);
	input.setNormalDistribution(0, 1);

	Tensor<double> want({ 4, 20 }, DeviceType::CPU);
	want.setNormalDistribution(0, 1);

	Tensor<double> output = learner.forward(input);

	cout << output.getShape() << " " << want.getShape() << "\n";

	double loss = evalL2Loss<double>(output, want).first;

	cout << loss << "\n";

	for (int tc = 1; 1; tc++) {

		learner.backProp(evalL2Loss(output, want).second);
		learner.gradientDescentSGD(6e-2);
		learner.resetDerivatives();

		output = learner.forward(input);
		cout << "new loss = " << fixed << setprecision(10) << evalL2Loss<double>(output, want).first << "\n";
	}
	exit(0);

	//Tensor<double> output = learner.forward(input);

	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 20; j++) {
			cout << output.v({ i, j }) << " ";
		}
		cout << "\n";
	}

	//cout << learner.forward(Tensor<double> { { 4, 10 }, DeviceType::CPU}).getShape() << "\n";
	map<string, Tensor<double>> lt;

}

x = (a + b) * c + d * e + f, class Tensor

x = (x + a) + b + x
|         |            |
gigi     vasile       vasile

a + b = c

gigi + vasile = add_res

lista de tensori, lista cu toate operatiile
lt, lo

() - () - () - () - ()

