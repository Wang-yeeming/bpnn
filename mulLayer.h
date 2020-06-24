#ifndef MULTIPLY_LAYER_INCLUDED
#define MULTIPLY_LAYER_INCLUDED

class mulLayer {
   private:
    double x;
    double y;

   public:
    double forward(double x, double y);
    double* backward(double dout);
};

#endif