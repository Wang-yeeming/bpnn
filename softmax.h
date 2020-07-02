#ifndef SOFTMAX_LAYER_INCLUDED
#define SOFTMAX_LAYER_INCLUDED

#include <cmath>
#include <iostream>

#include "matrix.h"

class sofLayer {
   private:
    // 损失
    double loss;
    // softmax的输出
    matrix out;
    // 监督数据
    matrix tag;
    // 损失函数：交叉熵误差
    double crossEntropyError(matrix* x, matrix* t);
    // softmax函数：将输出结果转换为概率
    matrix softmax(matrix* x);

   public:
    double forward(matrix* X, matrix* T);
    matrix backward(matrix* dL);
};

#endif  // !SOFTMAX_LAYER_INCLUDED