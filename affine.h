#ifndef AFFINE_LAYER_INCLUDED
#define AFFINE_LAYER_INCLUDED

#include "matrix.h"

class affLayer {
   private:
    matrix x;  // 数据域

   public:
    matrix weight;  // 权重
    matrix bias;    // 偏置
    matrix dw;      // 权值的导数
    matrix db;      // 偏置的导数
    affLayer();
    affLayer(const matrix& weight, const matrix& bias);
    matrix forward(const matrix& A);
    matrix backward(const matrix& dL);
};

#endif