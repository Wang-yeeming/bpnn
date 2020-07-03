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

bpnn::bpnn(size_t input_size, size_t hidden_size, size_t output_size) {
    this->input_num = input_size;
    this->hidden_num = hidden_size;
    this->output_num = output_size;
    this->size = 0;
    this->inMatVec.clear();
    this->tagMatVec.clear();
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
    size_t s = cache.size();
    double number;
    int integer;
    std::stringstream ss;
    // 将cache缓存的内容存到向量组内
    for (size_t i = 1; i <= feature_num + tag_num; i++) {
        for (size_t j = 1; j <= s; j++) {
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
    this->size = feature_vector[0].size();
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

void bpnn::train(size_t train_times, size_t batch_size) {
    TRY if (batch_size > this->size) throw "选取数据数目超出样本容量！";
    CATCH
    // 学习率
    double learning_rate = 0.1;
    // 初始化权值
    matrix w1(this->input_num, this->hidden_num);
    w1.randomMatrix(-1, 1);
    // 初始化偏置
    matrix b1(this->hidden_num);
    // 初始化第一个Affine层
    affLayer affine1(w1, b1);
    // 初始化激活函数层
    sigLayer sigmoid;
    // 初始化权值
    matrix w2(this->hidden_num, this->output_num);
    w2.randomMatrix(-1, 1);
    // 初始化偏置
    matrix b2(this->output_num);
    // 初始化第二个Affine层
    affLayer affine2(w2, b2);
    // 初始化输出层
    sofLayer last;
    // 随机抽取数据
    std::default_random_engine eng;
    eng.seed(time(NULL));
    std::uniform_int_distribution<int> uni(0, this->size - 1);
    // 存储随机的索引值并保证索引值不重复
    std::set<size_t> s;
    while (true) {
        s.insert(uni(eng));
        if (s.size() == batch_size) break;
    }
    // 转移到向量存储
    std::vector<size_t> vec;
    for (auto i : s) vec.push_back(i);
    // 用于遍历索引值的随机数生成器
    std::uniform_int_distribution<int> travel(0, batch_size - 1);
    matrix a1 = b1;
    matrix s1 = b1;
    matrix a2 = b2;
    double loss = 0;
    matrix dout(this->output_num);
    size_t index;
    for (size_t i = 0; i < this->output_num; i++) dout.data[0][i] = 1;
    // 开始训练
    for (size_t i = 0; i < train_times; i++) {
        index = vec[travel(eng)];
        // 前向传播
        a1 = affine1.forward(this->inMatVec[index]);
        s1 = sigmoid.forward(a1);
        a2 = affine2.forward(s1);
        loss = last.forward(a2, this->tagMatVec[index]);
        // 后向传播
        a2 = last.backward(dout);
        s1 = affine2.backward(a2);
        a1 = sigmoid.backward(s1);
        affine1.backward(a1);
        // 更新参数
        affine1.weight -= affine1.dw * learning_rate;
        affine1.bias -= affine1.db * learning_rate;
        affine2.weight -= affine2.dw * learning_rate;
        affine2.bias -= affine2.db * learning_rate;
    }
}
