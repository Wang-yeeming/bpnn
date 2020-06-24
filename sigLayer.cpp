#include "sigLayer.h"

double sigLayer::sigmoid(double x) { return (1.0 / (1 + exp(-x))); }

double sigLayer::desigmoid(double x) {
    return this->sigmoid(x) * (1.0 - this->sigmoid(x));
}

double sigLayer::forward(double x) { return this->sigmoid(x); }

double sigLayer::backforward(double dout) { return this->desigmoid(dout); }
