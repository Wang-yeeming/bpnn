#ifndef SIGMOID_LAYER_INCLUDED
#define SIGMOID_LAYER_INCLUDED

#include <cmath>

#include "matrix.h"

class sigLayer {
   private:
    matrix out;
    double sigmoid(double x);
    double dsigmoid(double x);

   public:
    matrix forward(const matrix& X);
    matrix backward(const matrix& dL);
};

#endif