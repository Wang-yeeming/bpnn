#include "affine.h"

affLayer::affLayer(const matrix& w, const matrix& b) {
    this->weight = w;
    this->bias = b;
}

matrix affLayer::forward(const matrix& A) {
    this->x = A;
    matrix L = std::move(this->x * this->weight + this->bias);
    return L;
}

matrix affLayer::backward(const matrix& dL) {
    matrix dX = dL * (~this->weight);
    this->dw = (~this->x) * dL;
    this->db = dL;
    return dX;
}
