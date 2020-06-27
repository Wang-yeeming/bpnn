/*
 * Affine层
 * 用于计算数据经过一层隐含层得到的结果
 */
#ifndef AFFINE_LAYER_INCLUDED
#define AFFINE_LAYER_INCLUDED

#include "matrix.h"

class affLayer {
   private:
    matrix x;// 数据域
    matrix weight;// 权重
    matrix bias;// 偏置

   public:
    affLayer(matrix weight, matrix bias);
    matrix forward(matrix A);
    matrix backward(matrix dL);
};

#endif