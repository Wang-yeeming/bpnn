#ifndef BPNN_INCLUDED
#define BPNN_INCLUDED

#include <exception>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "affine.h"   // Affine层
#include "matrix.h"   // 矩阵运算类
#include "sigmoid.h"  // Sigmoid层
#include "softmax.h"  // Softmax层

typedef struct node_t node_t;
typedef struct weight_t weight_t;

class bpnn {
   private:
    // 输入层节点数目
    size_t input_num = 0;
    // 隐含层（1层）节点数目
    size_t hidden_num = 0;
    // 输出层节点数目
    size_t output_num = 0;
    // 输入层
    std::vector<node_t*> input;
    // 隐含层（1层）
    std::vector<node_t*> hidden;
    // 输出层
    std::vector<node_t*> output;
    // 特征向量组
    std::vector<double>* feature_vector;
    // 目标向量组
    std::vector<std::string>* target_vector;

   public:
    // 无参构造器
    bpnn();
    // 有参构造器：指定输入层节点、隐含层（1层）节点、输出层节点数目
    bpnn(int input_num, int hidden_num, int output_num);
    // 析构器
    ~bpnn();
    // 指定输入层节点数目
    void setInputNum(int num);
    // 指定隐含层（1层）节点数目
    void setHiddenNum(int num);
    // 指定输出层节点数目
    void setOutputNum(int num);
    // 读取训练集数据（csv格式）
    void readTrainSet(std::string path);
    // 读取测试集数据（csv格式）
    void readTestSet(std::string path);
    // 生成BP神经网络
    void train();
};

#endif