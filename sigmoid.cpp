#include "sigmoid.h"

double sigLayer::sigmoid(double x) { return (1.0 / (1 + exp(-x))); }

double sigLayer::dsigmoid(double x) {
    return this->sigmoid(x) * (1.0 - this->sigmoid(x));
}

matrix sigLayer::forward(const matrix& X) {
    matrix L = X;
    for (int i = 0; i < L.line; i++)
        for (int j = 0; j < L.col; j++)
            L.data[i][j] = this->sigmoid(X.data[i][j]);
    return L;
}

matrix sigLayer::backward(const matrix& dL) {
    matrix dX = dL;
    for (int i = 0; i < dX.line; i++)
        for (int j = 0; j < dX.col; j++)
            dX.data[i][j] = this->dsigmoid(dL.data[i][j]);
    return dX;
}
