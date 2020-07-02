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
    double crossEntropyError(const matrix& x, const matrix& t);
    // softmax函数：将输出结果转换为概率
    matrix softmax(const matrix& x);

   public:
    double forward(const matrix& X, const matrix& T);
    matrix backward(const matrix& dL);
};

#endif  // !SOFTMAX_LAYER_INCLUDED