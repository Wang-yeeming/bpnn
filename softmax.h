#ifndef SOFTMAX_LAYER_INCLUDED
#define SOFTMAX_LAYER_INCLUDED

#include <cmath>
#include <iostream>

#include "matrix.h"

class sofLayer {
   private:
    // ��ʧ
    double loss;
    // softmax�����
    matrix out;
    // �ල����
    matrix tag;
    // ��ʧ���������������
    double crossEntropyError(matrix* x, matrix* t);
    // softmax��������������ת��Ϊ����
    matrix softmax(matrix* x);

   public:
    double forward(matrix* X, matrix* T);
    matrix backward(matrix* dL);
};

#endif  // !SOFTMAX_LAYER_INCLUDED