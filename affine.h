/*
 * Affine��
 * ���ڼ������ݾ���һ��������õ��Ľ��
 */
#ifndef AFFINE_LAYER_INCLUDED
#define AFFINE_LAYER_INCLUDED

#include "matrix.h"

class affLayer {
   private:
    matrix x;// ������
    matrix weight;// Ȩ��
    matrix bias;// ƫ��

   public:
    affLayer(matrix weight, matrix bias);
    matrix forward(matrix A);
    matrix backward(matrix dL);
};

#endif