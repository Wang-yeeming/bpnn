#include "mulLayer.h"

double mulLayer::forward(double up, double down) {
    this->x = up;
    this->y = down;
    return up * down;
}

double* mulLayer::backward(double dout) {
    double up = dout * this->y;
    double down = dout * this->x;
    double* arr = new double[2];
    arr[0] = up;
    arr[1] = down;
    return arr;
}
