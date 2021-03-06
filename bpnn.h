#ifndef BPNN_INCLUDED
#define BPNN_INCLUDED

#include <exception>
#include <fstream>
#include <iostream>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "affine.h"   // Affine层
#include "matrix.h"   // 矩阵运算类
#include "sigmoid.h"  // Sigmoid层
#include "softmax.h"  // Softmax层

class bpnn {
   private:
    matrix weight1;
    matrix bias1;
    matrix weight2;
    matrix bias2;

   public:
    // 测试输入数据矩阵组
    std::vector<matrix> test_inMatVec;
    // 测试监督数据矩阵组
    std::vector<matrix> test_tagMatVec;
    // 输入数据矩阵组
    std::vector<matrix> inMatVec;
    // 监督数据矩阵组
    std::vector<matrix> tagMatVec;
    // 损失
    std::vector<double> lossVec;
    // 输入层节点数目
    size_t input_num = 0;
    // 隐含层节点数目（1层）
    size_t hidden_num = 0;
    // 输出层节点数目
    size_t output_num = 0;
    // 数据数量
    size_t size;
    // 构造器
    bpnn(size_t input_size, size_t hidden_size, size_t output_size);
    void setInputSize(size_t);
    void setHiddenSize(size_t);
    void setOutputSize(size_t);
    // 读取训练集数据（csv格式）
    void readTrainSet(std::string path);
    // 读取测试集数据（csv格式）
    void readTestSet(std::string path);
    // 训练
    void train(size_t train_times, size_t batch_size);
    // 精度
    double accuracy();
    // 预测
    std::vector<double> predict(std::vector<double>);
};

#endif