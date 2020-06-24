#include "addLayer.h"

double addLayer::forward(double up, double down) { return up + down; }

double* addLayer::backward(double dout) {
    double up = dout * 1;
    double down = dout * 1;
    double* arr = new double[2];
    arr[0] = up;
    arr[1] = down;
    return arr;
}
