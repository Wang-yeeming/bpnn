#ifndef BPNN_INCLUDED
#define BPNN_INCLUDED

#include <exception>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "affine.h"   // Affine��
#include "matrix.h"   // ����������
#include "sigmoid.h"  // Sigmoid��
#include "softmax.h"  // Softmax��

class bpnn {
   private:
    // �����ڵ���Ŀ
    size_t input_num = 0;
    // �����㣨1�㣩�ڵ���Ŀ
    size_t hidden_num = 0;
    // �����ڵ���Ŀ
    size_t output_num = 0;
    // �������ݾ�����
    std::vector<matrix> inMatVec;
    // �ල���ݾ�����
    std::vector<matrix> tagMatVec;

   public:
    // �����������ɵ����������BP������
    bpnn(int input_size, int hidden_size, int output_size);
    // ������
    ~bpnn();
    // ��ȡѵ�������ݣ�csv��ʽ��
    void readTrainSet(std::string path);
    // ��ȡ���Լ����ݣ�csv��ʽ��
    void readTestSet(std::string path);
    // ѵ��
    void train();
    // ����Affine��
    affLayer createAffineLayer(matrix* weight, matrix* bias);
    // ����Sigmoid��
    sigLayer createSigmoidLayer();
    // ����Softmax with loss��
    sofLayer createSoftmaxWithLossLayer();
};

#endif