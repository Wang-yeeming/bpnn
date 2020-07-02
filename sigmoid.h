#ifndef SIGMOID_LAYER_INCLUDED
#define SIGMOID_LAYER_INCLUDED

#include <cmath>

#include "matrix.h"

class sigLayer {
   private:
    double sigmoid(double x);
    double desigmoid(double x);

   public:
    matrix forward(matrix* X);
    matrix backforward(matrix* dL);
};

#endif