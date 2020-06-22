#ifndef BPNN_INCLUDED
#define BPNN_INCLUDED

#include <exception>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace std;

typedef struct node_t node_t;

class bpnn {
   private:
    // �����ڵ���Ŀ
    size_t input_num = 0;
    // �����㣨1�㣩�ڵ���Ŀ
    size_t hidden_num = 0;
    // �����ڵ���Ŀ
    size_t output_num = 0;
    // �����
    vector<node_t*> input;
    // �����㣨1�㣩
    vector<node_t*> hidden;
    // �����
    vector<node_t*> output;

   public:
    // �޲ι�����
    bpnn();
    // �вι�������ָ�������ڵ㡢�����㣨1�㣩�ڵ㡢�����ڵ���Ŀ
    bpnn(int input_num, int hidden_num, int output_num);
    // ָ�������ڵ���Ŀ
    void setInputNum(int num);
    // ָ�������㣨1�㣩�ڵ���Ŀ
    void setHiddenNum(int num);
    // ָ�������ڵ���Ŀ
    void setOutputNum(int num);
    // ��ȡѵ�������ݣ�csv��ʽ��
    void readTrainSet(string path);
    // ��ȡ���Լ����ݣ�csv��ʽ��
    void readTestSet(string path);
};

#endif