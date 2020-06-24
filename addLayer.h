#ifndef ADD_LAYER_INCLUDED
#define ADD_LAYER_INCLUDED
class addLayer {
   public:
    double forward(double x, double y);
    double* backward(double dout);
};

#endif  // !ADD_LAYER_INCLUDED
