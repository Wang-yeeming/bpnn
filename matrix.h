/*
 * ����һ������������
 * ��ʵ���˾���ļӷ����˷�����
 */
#ifndef MATRIX_INCLUDED
#define MATRIX_INCLUDED
#include <iomanip>
#include <iostream>
class matrix {
   private:
    int line;       // ����
    int col;        // ����
    double** data;  // ����
   public:
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
    // ����ӷ�����
    friend matrix operator+(const matrix& A, const matrix& B);
    // ����˷�����
    friend matrix operator*(const matrix& A, const matrix& B);
    // ��ʽ�����
    friend std::ostream& operator<<(std::ostream& os, const matrix& A);
};

#endif  // !MATRIX_INCLUDED
