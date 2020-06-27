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

typedef struct node_t node_t;
typedef struct weight_t weight_t;

class bpnn {
   private:
    // �����ڵ���Ŀ
    size_t input_num = 0;
    // �����㣨1�㣩�ڵ���Ŀ
    size_t hidden_num = 0;
    // �����ڵ���Ŀ
    size_t output_num = 0;
    // �����
    std::vector<node_t*> input;
    // �����㣨1�㣩
    std::vector<node_t*> hidden;
    // �����
    std::vector<node_t*> output;
    // ����������
    std::vector<double>* feature_vector;
    // Ŀ��������
    std::vector<std::string>* target_vector;

   public:
    // �޲ι�����
    bpnn();
    // �вι�������ָ�������ڵ㡢�����㣨1�㣩�ڵ㡢�����ڵ���Ŀ
    bpnn(int input_num, int hidden_num, int output_num);
    // ������
    ~bpnn();
    // ָ�������ڵ���Ŀ
    void setInputNum(int num);
    // ָ�������㣨1�㣩�ڵ���Ŀ
    void setHiddenNum(int num);
    // ָ�������ڵ���Ŀ
    void setOutputNum(int num);
    // ��ȡѵ�������ݣ�csv��ʽ��
    void readTrainSet(std::string path);
    // ��ȡ���Լ����ݣ�csv��ʽ��
    void readTestSet(std::string path);
    // ����BP������
    void train();
};

#endif