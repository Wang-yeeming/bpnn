#include "bpnn.h"
#define TRY try {
#define CATCH                          \
    }                                  \
    catch (const char* msg) {          \
        std::cerr << msg << std::endl; \
        system("pause");               \
        exit(-1);                      \
    }

static std::vector<std::string> cache;

bpnn::bpnn(int input_size, int output_size) {
    this->input_num = input_size;
    this->output_num = output_size;
}

bpnn::~bpnn() {}

void bpnn::readTrainSet(std::string path) {
    char data[128];  // 缓冲区
    std::string str;
    std::ifstream fin;  // 文件输入流
    fin.open(path, std::ios::in);
    // 检查文件是否有效
    TRY if (!fin) throw "打开文件失败";
    CATCH
    fin >> data;
    std::string::size_type start, end;  // 分割子串用
    cache.clear();
    int count = 0;                         // 读取行数计数器
    size_t feature_num = this->input_num;  // 特征向量数目
    size_t tag_num = this->output_num;     // 监督向量数目
    while (fin) {
        count++;
        // 从第 2 行开始读入数据
        fin >> data;
        str = data;
        start = 0;
        if (str.compare("") != 0) {
            // 分割字符串
            while (true) {
                end = str.find(",");
                if (end != std::string::npos) {
                    cache.push_back(str.substr(start, end - start));
                    start = 0;
                    str = str.substr(end + 1, str.length() - end - 1);
                } else {
                    cache.push_back(str);
                    break;
                }
            }
        }
    }
    // 存储特征向量组
    std::vector<double>* feature_vector = new std::vector<double>[feature_num];
    // 存储监督向量组
    std::vector<int>* tag_vector = new std::vector<int>[tag_num];
    size_t tmp;
    size_t size = cache.size();
    double number;
    int integer;
    std::stringstream ss;
    // 将cache缓存的内容存到向量组内
    for (size_t i = 1; i <= feature_num + tag_num; i++) {
        for (size_t j = 1; j <= size; j++) {
            if (i == feature_num + tag_num)
                tmp = 0;
            else
                tmp = i;
            if (j % (feature_num + tag_num) == tmp) {
                if (i <= feature_num) {
                    // 将离散型数据转化为连续型数据
                    ss.clear();
                    ss << cache[j - 1];
                    ss >> number;
                    // 将特征存入特征向量
                    feature_vector[i - 1].push_back(number);
                } else {
                    // 将监督存入监督向量
                    ss.clear();
                    ss << cache[j - 1];
                    ss >> integer;
                    tag_vector[i - feature_num - 1].push_back(integer);
                }
            }
        }
    }
    size = feature_vector[0].size();
    // 将输入数据转化成矩阵存储
    matrix m(feature_num);
    for (size_t i = 0; i < size; i++) {
        m.setZero();
        for (size_t j = 0; j < feature_num; j++)
            m.data[0][j] = feature_vector[j][i];
        this->inMatVec.push_back(m);
    }
    // 将监督数据转化为矩阵存储
    matrix n(tag_num);
    for (size_t i = 0; i < size; i++) {
        n.setZero();
        for (size_t j = 0; j < tag_num; j++) n.data[0][j] = tag_vector[j][i];
        this->tagMatVec.push_back(n);
    }
}

void bpnn::readTestSet(std::string path) {
    char data[128];  // 缓冲区
    std::string str;
    std::ifstream fin;  // 文件输入流
    fin.open(path, std::ios::in);
    fin >> data;
    while (fin) {
        // 从第 2 行开始读入数据
        fin >> data;
        str = data;
        if (str.compare("") != 0) std::cout << str << std::endl;
    }
}

void bpnn::train() {
    // 初始化
}

affLayer bpnn::createAffineLayer(matrix* weight, matrix* bias) {
    affLayer* affine = new affLayer(*weight, *bias);
    return *affine;
}

sigLayer bpnn::createSigmoidLayer() {
    sigLayer* sigmoid = new sigLayer();
    return *sigmoid;
}

sofLayer bpnn::createSoftmaxWithLossLayer() {
    sofLayer* softmax = new sofLayer();
    return *softmax;
}
