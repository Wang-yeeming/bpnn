#include "matrix.h"

matrix::matrix() {}

matrix::matrix(size_t l, size_t c) {
    this->line = l;
    this->col = c;
    // 分配内存空间
    this->data = new double*[l];
    for (size_t i = 0; i < l; i++) this->data[i] = new double[c];
    // 初始化为零矩阵
    for (size_t i = 0; i < l; i++)
        for (size_t j = 0; j < c; j++) this->data[i][j] = 0;
}

matrix::matrix(const matrix& obj) {
    this->line = obj.line;
    this->col = obj.col;
    // 分配内存空间
    this->data = new double*[obj.line];
    for (size_t i = 0; i < obj.line; i++) this->data[i] = new double[obj.col];
    // 拷贝值
    for (size_t i = 0; i < obj.line; i++)
        for (size_t j = 0; j < obj.col; j++) this->data[i][j] = obj.data[i][j];
}

matrix::matrix(const matrix&& obj) {
    this->line = obj.line;
    this->col = obj.col;
    // 分配内存空间
    this->data = new double*[obj.line];
    for (size_t i = 0; i < obj.line; i++) this->data[i] = new double[obj.col];
    // 拷贝值
    for (size_t i = 0; i < obj.line; i++)
        for (size_t j = 0; j < obj.col; j++) this->data[i][j] = obj.data[i][j];
}

matrix::matrix(size_t c) {
    this->line = 1;
    this->col = c;
    // 分配内存空间
    this->data = new double*[1];
    this->data[0] = new double[c];
    // 初始化为零矩阵
    for (size_t j = 0; j < c; j++) this->data[0][j] = 0;
}

matrix::~matrix() {
    for (size_t i = 0; i < this->line; i++) delete[] this->data[i];

    delete[] this->data;
}

void matrix::randomMatrix(double min, double max) {
    std::default_random_engine eng;
    eng.seed(time(NULL));
    std::normal_distribution<double> n(min, max);
    for (size_t i = 0; i < this->line; i++)
        for (size_t j = 0; j < this->col; j++) this->data[i][j] = n(eng);
}

void matrix::setZero() {
    for (size_t i = 0; i < this->line; i++)
        for (size_t j = 0; j < this->col; j++) this->data[i][j] = 0;
}

void matrix::input(double array[]) {
    size_t i = 0;
    size_t j = 0;
    for (size_t a = 0; a < this->col * this->line; a++) {
        i = a / this->col;
        j = a % this->col;
        this->data[i][j] = array[a];
    }
}

double** matrix::output() {
    double** array = new double*[this->line];
    for (size_t i = 0; i < this->line; i++) array[i] = new double[this->col];
    *array = *this->data;
    return array;
}

matrix matrix::operator=(const matrix& A) {
    this->line = A.line;
    this->col = A.col;
    // 分配内存空间
    this->data = new double*[A.line];
    for (size_t i = 0; i < A.line; i++) this->data[i] = new double[A.col];
    // 拷贝值
    for (size_t i = 0; i < A.line; i++)
        for (size_t j = 0; j < A.col; j++) this->data[i][j] = A.data[i][j];
    return *this;
}

matrix matrix::operator-=(const matrix& A) {
    *this = *this - A;
    return *this;
}

matrix operator~(const matrix& A) {
    matrix B(A.col, A.line);
    for (size_t i = 0; i < B.line; i++)
        for (size_t j = 0; j < B.col; j++) B.data[i][j] = A.data[j][i];
    return B;
}

matrix operator+(const matrix& A, const matrix& B) {
    try {
        if (A.line != B.line) throw "矩阵加法要求行数相同";
        if (A.col != B.col) throw "矩阵加法要求列数相同";
    } catch (const char* msg) {
        std::cout << msg << std::endl;
        system("pause");
        exit(-1);
    }
    matrix C(A.line, A.col);
    for (size_t i = 0; i < C.line; i++)
        for (size_t j = 0; j < C.col; j++)
            C.data[i][j] = A.data[i][j] + B.data[i][j];
    return C;
}

matrix operator+(const matrix& A, const double& B) {
    matrix C(A.line, A.col);
    for (size_t i = 0; i < C.line; i++)
        for (size_t j = 0; j < C.col; j++) C.data[i][j] = A.data[i][j] + B;
    return C;
}

matrix operator-(const matrix& A, const matrix& B) {
    try {
        if (A.line != B.line) throw "矩阵减法要求行数相同";
        if (A.col != B.col) throw "矩阵减法要求列数相同";
    } catch (const char* msg) {
        std::cerr << msg << std::endl;
        system("pause");
        exit(-1);
    }
    matrix C(A.line, A.col);
    for (size_t i = 0; i < C.line; i++)
        for (size_t j = 0; j < C.col; j++)
            C.data[i][j] = A.data[i][j] - B.data[i][j];
    return C;
}

matrix operator-(const double& B, const matrix& A) {
    matrix C(A.line, A.col);
    for (size_t i = 0; i < C.line; i++)
        for (size_t j = 0; j < C.col; j++) C.data[i][j] = B - A.data[i][j];
    return C;
}

matrix operator*(const matrix& A, const matrix& B) {
    try {
        if (A.col != B.line) throw "无法进行矩阵乘法运算";
    } catch (const char* msg) {
        std::cerr << msg << std::endl;
        system("pause");
        exit(-1);
    }
    matrix C(A.line, B.col);
    for (size_t i = 0; i < C.line; i++)
        for (size_t j = 0; j < C.col; j++)
            for (size_t m = 0; m < A.col; m++)
                C.data[i][j] += A.data[i][m] * B.data[m][j];
    return C;
}

matrix operator*(const matrix& A, const double& C) {
    matrix B(A.line, A.col);
    for (size_t i = 0; i < B.line; i++)
        for (size_t j = 0; j < B.col; j++) B.data[i][j] = A.data[i][j] * C;
    return B;
}

matrix operator/(const matrix& A, const double& C) {
    try {
        if (C == 0) throw "无效的除法运算";
    } catch (const char* msg) {
        std::cerr << msg << std::endl;
        system("pause");
        exit(-1);
    }
    matrix B(A.line, A.col);
    for (size_t i = 0; i < B.line; i++)
        for (size_t j = 0; j < B.col; j++) B.data[i][j] = A.data[i][j] / C;
    return B;
}

std::ostream& operator<<(std::ostream& os, const matrix& A) {
    for (size_t i = 0; i < A.line; i++) {
        for (size_t j = 0; j < A.col; j++) {
            if (j == 0) {
                if (i == 0)
                    os << std::left << std::setw(3) << "┏";
                else if (i == A.line - 1)
                    os << std::left << std::setw(3) << "┗";
                else
                    os << std::left << std::setw(3) << "┃";
            }
            os << std::setw(10) << std::fixed << A.data[i][j];
            if (j == A.col - 1) {
                if (i == 0)
                    os << std::right << std::setw(2) << "┓";
                else if (i == A.line - 1)
                    os << std::right << std::setw(2) << "┛";
                else
                    os << std::right << std::setw(2) << "┃";
            }
        }
        os << std::endl;
    }
    return os;
}
