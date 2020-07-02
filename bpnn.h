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
    // �����ڵ���Ŀ
    size_t output_num = 0;

   public:
    // �������ݾ�����
    std::vector<matrix> inMatVec;
    // �ල���ݾ�����
    std::vector<matrix> tagMatVec;
    // ������
    bpnn(int input_size, int output_size);
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