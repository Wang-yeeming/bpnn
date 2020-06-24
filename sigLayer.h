#ifndef SIGMOID_LAYER_INCLUDED
#define SIGMOID_LAYER_INCLUDED
#include <cmath>
class sigLayer {
   private:
    double sigmoid(double x);
    double desigmoid(double x);

   public:
    double forward(double x);
    double backforward(double dout);
};

#endif