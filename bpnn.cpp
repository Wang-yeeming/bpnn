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

bpnn::bpnn(int input_size, int hidden_size, int output_size) {
    this->input_num = input_size;
    this->hidden_num = hidden_size;
    this->output_num = output_size;
    // 生成均匀分布的随机数来初始化权重
    std::default_random_engine eng;
    std::uniform_real_distribution<double> u(0, 1);
    // 初始化权重
    matrix* w1 = new matrix(this->hidden_num);
    double* arr1 = new double[this->hidden_num];
    for (size_t i = 0; i < this->hidden_num; i++) arr1[i] = u(eng);
    w1->input(arr1);
    // 转为列向量
    *w1 = ~(*w1);
    delete[] arr1;
    this->params["w1"] = *w1;
    // 偏置设为0
    matrix* b1 = new matrix(this->hidden_num);
    this->params["b1"] = *b1;
}

bpnn::~bpnn() {
    delete[] this->feature_vector;
    delete[] this->target_vector;
}

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
    size_t target_num = this->output_num;  // 目标向量数目
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
    this->feature_vector = new std::vector<double>[feature_num];
    // 存储目标向量组
    this->target_vector = new std::vector<std::string>[target_num];
    unsigned tmp;
    unsigned size = cache.size();
    double number;
    std::stringstream ss;
    for (unsigned i = 1; i <= feature_num + target_num; i++) {
        for (unsigned j = 1; j <= size; j++) {
            if (i == feature_num + target_num)
                tmp = 0;
            else
                tmp = i;
            if (j % (feature_num + target_num) == tmp) {
                if (i <= feature_num) {
                    // 将离散型数据转化为连续型数据
                    ss.clear();
                    ss << cache[j - 1];
                    ss >> number;
                    // 将特征存入特征向量
                    feature_vector[i - 1].push_back(number);
                } else {
                    // 将目标存入目标向量
                    target_vector[i - feature_num - 1].push_back(cache[j - 1]);
                }
            }
        }
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

affLayer bpnn::createAffineLayer(matrix weight, matrix bias) {
    affLayer* affine = new affLayer(weight, bias);
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
