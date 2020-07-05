/*
 * 这是一个矩阵运算类
 * 简单实现了矩阵的加法、乘法运算
 */
#ifndef MATRIX_INCLUDED
#define MATRIX_INCLUDED

#include <iomanip>
#include <iostream>
#include <random>

class matrix {
    friend class bpnn;
    friend class affLayer;
    friend class sigLayer;
    friend class sofLayer;

   private:
    // 数据
    double** data;

   public:
    // 行数
    size_t line;
    // 列数
    size_t col;
    // 默认构造器
    matrix();
    // 构造m*n矩阵
    matrix(size_t line, size_t col);
    // 拷贝构造函数
    matrix(const matrix& obj);
    // 移动构造函数
    matrix(const matrix&& obj);
    // 构造行向量
    matrix(size_t col);
    // 析构器
    ~matrix();
    // 生成随机数矩阵
    void randomMatrix(double min, double max);
    // 矩阵清零
    void setZero();
    // 通过输入一维数组构造矩阵
    void input(double array[]);
    // 以二维数组的形式输出矩阵
    double** output();
    // 深拷贝
    matrix operator=(const matrix& A);
    matrix operator-=(const matrix& A);
    // 矩阵的转置
    friend matrix operator~(const matrix& A);
    // 矩阵加上矩阵
    friend matrix operator+(const matrix& A, const matrix& B);
    // 矩阵加上常数
    friend matrix operator+(const matrix& A, const double& B);
    // 矩阵减法运算
    friend matrix operator-(const matrix& A, const matrix& B);
    // 常数减去矩阵
    friend matrix operator-(const double& A, const matrix& B);
    // 矩阵乘法运算
    friend matrix operator*(const matrix& A, const matrix& B);
    // 矩阵乘以常数
    friend matrix operator*(const matrix& A, const double& B);
    // 矩阵除以常数
    friend matrix operator/(const matrix& A, const double& C);
    // 格式化输出
    friend std::ostream& operator<<(std::ostream& os, const matrix& A);
};

#endif  // !MATRIX_INCLUDED
