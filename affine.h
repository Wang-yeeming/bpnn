/*
 * Affine��
 * ���ڼ������ݾ���һ��������õ��Ľ��
 */
#ifndef AFFINE_LAYER_INCLUDED
#define AFFINE_LAYER_INCLUDED

#include "matrix.h"

class affLayer {
   private:
    matrix x;       // ������
    matrix weight;  // Ȩ��
    matrix bias;    // ƫ��

   public:
    affLayer(const matrix& weight, const matrix& bias);
    matrix forward(const matrix& A);
    matrix backward(const matrix& dL);
};

#endif