#include "affine.h"

matrix affLayer::forward(matrix A) {
    this->x = A;
    matrix L = this->x * this->weight + this->bias;
    return L;
}

matrix affLayer::backward(matrix dL) {
    matrix dX = dL * (~this->weight);
    return dX;
}
