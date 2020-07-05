#include "affine.h"

affLayer::affLayer() {}

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
    this->db = matrix(dL.line, dL.col);
    for (size_t i = 0; i < dL.line; i++)
        for (size_t j = 0; j < dL.col; j++)
            for (size_t k = 0; k < dL.line; k++)
                this->db.data[i][j] += dL.data[k][j];
    return dX;
}
