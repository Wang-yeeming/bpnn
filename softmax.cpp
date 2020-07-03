#include "softmax.h"

double sofLayer::crossEntropyError(const matrix& x, const matrix& t) {
    try {
        if (x.line != 1) throw "无法进行交叉熵误差计算";
    } catch (const char* msg) {
        std::cerr << msg << std::endl;
    }
    double sum = 0;
    double delta = 1e-7;
    for (size_t i = 0; i < x.col; i++) sum += t.data[0][i] * log(x.data[0][i] + delta);
    double error = (-sum) / x.col;
    return error;
}

matrix sofLayer::softmax(const matrix& x) {
    try {
        if (x.line != 1) throw "无法进行softmax计算";
    } catch (const char* msg) {
        std::cerr << msg << std::endl;
    }
    matrix y(1, x.col);
    double sum = 0;
    for (size_t i = 0; i < x.col; i++) sum += exp(x.data[0][i]);
    for (size_t i = 0; i < x.col; i++) y.data[0][i] = exp(x.data[0][i]) / sum;
    return y;
}

double sofLayer::forward(const matrix& X, const matrix& T) {
    this->tag = T;
    this->out = std::move(softmax(X));
    this->loss = crossEntropyError(this->out, T);
    return this->loss;
}

matrix sofLayer::backward(const matrix& dL) {
    int size = dL.col;
    matrix dX = (this->out - this->tag) / size;
    return dX;
}
