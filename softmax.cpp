#include "softmax.h"

double sofLayer::crossEntropyError(matrix x, matrix t) {
    try {
        if (x.line != 1) throw "无法进行交叉熵误差计算";
    } catch (const char* msg) {
        std::cerr << msg << std::endl;
    }
    double error = 0;
    for (int i = 0; i < x.col; i++) error -= t.data[0][i] * log(x.data[0][i]);
    return error;
}

matrix sofLayer::softmax(matrix x) {
    try {
        if (x.line != 1) throw "无法进行softmax计算";
    } catch (const char* msg) {
        std::cerr << msg << std::endl;
    }
    matrix* y = new matrix(1, x.col);
    double sum = 0;
    for (int i = 0; i < x.col; i++) sum += exp(x.data[0][i]);
    for (int i = 0; i < x.col; i++) y->data[0][i] = exp(x.data[0][i]) / sum;
    return *y;
}

double sofLayer::forward(matrix X, matrix T) {
    this->tag = T;
    this->loss = crossEntropyError(X, T);
    this->out = softmax(X);
    return this->loss;
}

matrix sofLayer::backward(matrix dL) {
    int size = dL.col;
    matrix dX = (this->out - this->tag) / size;
    return dX;
}
