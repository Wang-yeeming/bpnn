#ifndef AFFINE_LAYER_INCLUDED
#define AFFINE_LAYER_INCLUDED

#include "matrix.h"

class affLayer {
   private:
    matrix x;  // ������

   public:
    matrix weight;  // Ȩ��
    matrix bias;    // ƫ��
    matrix dw;      // Ȩֵ�ĵ���
    matrix db;      // ƫ�õĵ���
    affLayer();
    affLayer(const matrix& weight, const matrix& bias);
    matrix forward(const matrix& A);
    matrix backward(const matrix& dL);
};

#endif