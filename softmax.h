#ifndef SOFTMAX_LAYER_INCLUDED
#define SOFTMAX_LAYER_INCLUDED

#include <cmath>
#include <iostream>

#include "matrix.h"

class sofLayer {
   private:
    // ��ʧ
    double loss;
    // �ල����
    // ��ʧ���������������
    double crossEntropyError(const matrix& x, const matrix& t);
    // softmax��������������ת��Ϊ����
    matrix softmax(const matrix& x);

   public:
    // softmax�����
       matrix tag;
    matrix out;
    // ������ʧ
    double forward(const matrix& X, const matrix& T);
    matrix backward(const matrix& dL);
};

#endif  // !SOFTMAX_LAYER_INCLUDED