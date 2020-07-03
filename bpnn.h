#ifndef BPNN_INCLUDED
#define BPNN_INCLUDED

#include <exception>
#include <fstream>
#include <iostream>
#include <set>
#include <sstream>
#include <string>
#include <vector>
#include <random>

#include "affine.h"   // Affine��
#include "matrix.h"   // ����������
#include "sigmoid.h"  // Sigmoid��
#include "softmax.h"  // Softmax��

class bpnn {
   private:
    // affine��
    affLayer affineLayer1;
    affLayer affineLayer2;
    // �������
    sigLayer sigmoidLayer;
    // �����
    sofLayer lastLayer;

   public:
    // �����������ݾ�����
    std::vector<matrix> test_inMatVec;
    // ���Լල���ݾ�����
    std::vector<matrix> test_tagMatVec;
    // �������ݾ�����
    std::vector<matrix> inMatVec;
    // �ල���ݾ�����
    std::vector<matrix> tagMatVec;
    // �����ڵ���Ŀ
    size_t input_num = 0;
    // ������ڵ���Ŀ��1�㣩
    size_t hidden_num = 0;
    // �����ڵ���Ŀ
    size_t output_num = 0;
    // ��������
    size_t size;
    // ������
    bpnn(size_t input_size, size_t hidden_size, size_t output_size);
    // ��ȡѵ�������ݣ�csv��ʽ��
    void readTrainSet(std::string path);
    // ��ȡ���Լ����ݣ�csv��ʽ��
    void readTestSet(std::string path);
    // ѵ��
    void train(size_t train_times, size_t batch_size);
    // ����
    double accuracy();
};

#endif