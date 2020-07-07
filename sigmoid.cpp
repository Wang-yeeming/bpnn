#include "sigmoid.h"

double sigLayer::sigmoid(double x) { return (1.0 / (1.0 + exp(-x))); }

double sigLayer::dsigmoid(double x) {
    return this->sigmoid(x) * (1.0 - this->sigmoid(x));
}

matrix sigLayer::forward(const matrix& X) {
    matrix L(X.line, X.col);
    for (size_t i = 0; i < L.line; i++)
        for (size_t j = 0; j < L.col; j++)
            L.data[i][j] = this->sigmoid(X.data[i][j]);
    this->out = std::move(L);
    return this->out;
}

matrix sigLayer::backward(const matrix& dL) {
    matrix dX(dL.line, dL.col);
    for (size_t i = 0; i < dX.line; i++)
        for (size_t j = 0; j < dX.col; j++)
            dX.data[i][j] =
                dL.data[i][j] * this->dsigmoid(this->out.data[i][j]);
    return dX;
}
