/*
 * ����һ������������
 * ��ʵ���˾���ļӷ����˷�����
 */
#ifndef MATRIX_INCLUDED
#define MATRIX_INCLUDED

#include <iomanip>
#include <iostream>

class matrix {
    friend class sigLayer;

   private:
    // ����
    double** data;

   public:
    // ����
    int line;
    // ����
    int col;
    // Ĭ�Ϲ�����
    matrix();
    // ������
    matrix(int line, int col);
    // �������캯��
    matrix(const matrix& obj);
    // ������
    ~matrix();
    // ͨ������һά���鹹�����
    void input(double array[]);
    // �Զ�ά�������ʽ�������
    double** output();
    // �����ת��
    friend matrix operator~(const matrix& A);
    // ����ӷ�����
    friend matrix operator+(const matrix& A, const matrix& B);
    // ����˷�����
    friend matrix operator*(const matrix& A, const matrix& B);
    // ��ʽ�����
    friend std::ostream& operator<<(std::ostream& os, const matrix& A);
};

#endif  // !MATRIX_INCLUDED
