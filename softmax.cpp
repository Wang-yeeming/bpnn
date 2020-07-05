#include "softmax.h"

double sofLayer::crossEntropyError(const matrix& x, const matrix& t) {
    double sum = 0;
    double delta = 1e-7;
    for (size_t i = 0; i < x.line; i++)
        for (size_t j = 0; j < x.col; j++)
            sum += t.data[i][j] * log(x.data[i][j] + delta);
    double error = (-sum) / x.line;
    return error;
}

matrix sofLayer::softmax(const matrix& x) {
    double max = 0;
    for (size_t i = 0; i < x.line; i++)
        for (size_t j = 0; j < x.col; j++)
            if (max < x.data[i][j]) max = x.data[i][j];
    matrix y(x.line, x.col);
    double* sum = new double[x.line];
    for (size_t i = 0; i < x.line; i++) sum[i] = 0;
    for (size_t i = 0; i < x.line; i++)
        for (size_t j = 0; j < x.col; j++) sum[i] += exp(x.data[i][j] - max);
    for (size_t i = 0; i < x.line; i++)
        for (size_t j = 0; j < x.col; j++)
            y.data[i][j] = exp(x.data[i][j] - max) / sum[i];
    return y;
}

double sofLayer::forward(const matrix& X, const matrix& T) {
    this->tag = T;
    this->out = std::move(softmax(X));
    this->loss = crossEntropyError(this->out, T);
    return this->loss;
}

matrix sofLayer::backward(const matrix& dL) {
    int size = this->tag.col;
    matrix dX = (this->out - this->tag) / (double)size;
    return dX;
}
