/*
 * 这是一个矩阵运算类
 * 简单实现了矩阵的加法、乘法运算
 */
#ifndef MATRIX_INCLUDED
#define MATRIX_INCLUDED

#include <iomanip>
#include <iostream>

class matrix {
    friend class sigLayer;

   private:
    // 数据
    double** data;

   public:
    // 行数
    int line;
    // 列数
    int col;
    // 默认构造器
    matrix();
    // 构造器
    matrix(int line, int col);
    // 拷贝构造函数
    matrix(const matrix& obj);
    // 析构器
    ~matrix();
    // 通过输入一维数组构造矩阵
    void input(double array[]);
    // 以二维数组的形式输出矩阵
    double** output();
    // 矩阵的转置
    friend matrix operator~(const matrix& A);
    // 矩阵加法运算
    friend matrix operator+(const matrix& A, const matrix& B);
    // 矩阵乘法运算
    friend matrix operator*(const matrix& A, const matrix& B);
    // 格式化输出
    friend std::ostream& operator<<(std::ostream& os, const matrix& A);
};

#endif  // !MATRIX_INCLUDED
