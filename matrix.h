/*
 * ����һ������������
 * ��ʵ���˾���ļӷ����˷�����
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
    // ����
    double** data;

   public:
    // ����
    size_t line;
    // ����
    size_t col;
    // Ĭ�Ϲ�����
    matrix();
    // ����m*n����
    matrix(size_t line, size_t col);
    // �������캯��
    matrix(const matrix& obj);
    // �ƶ����캯��
    matrix(const matrix&& obj);
    // ����������
    matrix(size_t col);
    // ������
    ~matrix();
    // �������������
    void randomMatrix(double min, double max);
    // ��������
    void setZero();
    // ͨ������һά���鹹�����
    void input(double array[]);
    // �Զ�ά�������ʽ�������
    double** output();
    // ���
    matrix operator=(const matrix& A);
    matrix operator-=(const matrix& A);
    // �����ת��
    friend matrix operator~(const matrix& A);
    // ������Ͼ���
    friend matrix operator+(const matrix& A, const matrix& B);
    // ������ϳ���
    friend matrix operator+(const matrix& A, const double& B);
    // �����������
    friend matrix operator-(const matrix& A, const matrix& B);
    // ������ȥ����
    friend matrix operator-(const double& A, const matrix& B);
    // ����˷�����
    friend matrix operator*(const matrix& A, const matrix& B);
    // ������Գ���
    friend matrix operator*(const matrix& A, const double& B);
    // ������Գ���
    friend matrix operator/(const matrix& A, const double& C);
    // ��ʽ�����
    friend std::ostream& operator<<(std::ostream& os, const matrix& A);
};

#endif  // !MATRIX_INCLUDED
