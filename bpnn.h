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

class bpnn {
   private:
    // 输入层节点数目
    size_t input_num = 0;
    // 输出层节点数目
    size_t output_num = 0;

   public:
    // 输入数据矩阵组
    std::vector<matrix> inMatVec;
    // 监督数据矩阵组
    std::vector<matrix> tagMatVec;
    // 构造器
    bpnn(int input_size, int output_size);
    // 析构器
    ~bpnn();
    // 读取训练集数据（csv格式）
    void readTrainSet(std::string path);
    // 读取测试集数据（csv格式）
    void readTestSet(std::string path);
    // 训练
    void train();
    // 生成Affine层
    affLayer createAffineLayer(matrix* weight, matrix* bias);
    // 生成Sigmoid层
    sigLayer createSigmoidLayer();
    // 生成Softmax with loss层
    sofLayer createSoftmaxWithLossLayer();
};

#endif