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
}

void bpnn::setInputSize(size_t s) { this->input_num = s; }

void bpnn::setHiddenSize(size_t s) { this->hidden_num = s; }

void bpnn::setOutputSize(size_t s) { this->output_num = s; }

double bpnn::accuracy() {
    size_t test_size = this->test_inMatVec.size();
    affLayer affine1(this->weight1, this->bias1);
    sigLayer sigmoid;
    affLayer affine2(this->weight2, this->bias2);
    sofLayer last;
    matrix a1, a2, s1, out;
    double right = 0;
    size_t max1, max2;
    for (size_t i = 0; i < test_size; i++) {
        max1 = 0;
        max2 = 0;
        a1 = affine1.forward(this->test_inMatVec[i]);
        s1 = sigmoid.forward(a1);
        a2 = affine2.forward(s1);
        last.forward(a2, this->test_tagMatVec[i]);
        out = last.out;
        for (size_t j = 0; j < out.col; j++)
            if (out.data[0][max1] < out.data[0][j]) max1 = j;
        for (size_t j = 0; j < this->test_tagMatVec[i].col; j++)
            if (this->test_tagMatVec[i].data[0][max2] <
                this->test_tagMatVec[i].data[0][j])
                max2 = j;
        if (max1 == max2) right++;
    }
    return right / test_size;
}

std::vector<double> bpnn::predict(std::vector<double> vec) {
    affLayer affine1(this->weight1, this->bias1);
    sigLayer sigmoid;
    affLayer affine2(this->weight2, this->bias2);
    sofLayer last;
    matrix in(vec.size());
    matrix a1, a2, s1, out;
    matrix feak(this->output_num);
    feak.randomMatrix(0, 1);
    for (size_t i = 0; i < vec.size(); i++) in.data[0][i] = vec[i];
    a1 = affine1.forward(in);
    s1 = sigmoid.forward(a1);
    a2 = affine2.forward(s1);
    last.forward(a2, feak);
    out = last.out;
    vec.clear();
    for (size_t i = 0; i < this->output_num; i++) vec.push_back(out.data[0][i]);
    return vec;
}

void bpnn::readTrainSet(std::string path) {
    char data[128];  // 缓冲区
    std::string str;
    std::ifstream fin;  // 文件输入流
    fin.open(path, std::ios::in);
    // 检查文件是否有效
    TRY if (!fin) throw "打开文件失败";
    CATCH
    std::string::size_type start, end;  // 分割子串用
    cache.clear();
    size_t count = 0;                      // 读取行数计数器
    size_t feature_num = this->input_num;  // 特征向量数目
    size_t tag_num = this->output_num;     // 监督向量数目
    while (fin) {
        count++;
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
    std::vector<size_t>* tag_vector = new std::vector<size_t>[tag_num];
    size_t tmp;
    size_t s = cache.size();
    double number;
    size_t integer;
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
    this->inMatVec.clear();
    this->tagMatVec.clear();
    this->test_inMatVec.clear();
    this->test_tagMatVec.clear();
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
    // 测试精度用
    this->test_inMatVec.assign(this->inMatVec.begin(), this->inMatVec.end());
    this->test_tagMatVec.assign(this->tagMatVec.begin(), this->tagMatVec.end());
    delete[] feature_vector;
    delete[] tag_vector;
    cache.clear();
}

void bpnn::readTestSet(std::string path) {
    char data[128];  // 缓冲区
    std::string str;
    std::ifstream fin;  // 文件输入流
    fin.open(path, std::ios::in);
    // 检查文件是否有效
    TRY if (!fin) throw "打开文件失败";
    CATCH
    std::string::size_type start, end;  // 分割子串用
    cache.clear();
    size_t count = 0;                      // 读取行数计数器
    size_t feature_num = this->input_num;  // 特征向量数目
    size_t tag_num = this->output_num;     // 监督向量数目
    while (fin) {
        count++;
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
    std::vector<size_t>* tag_vector = new std::vector<size_t>[tag_num];
    size_t tmp;
    size_t s = cache.size();
    double number;
    size_t integer;
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
    this->test_inMatVec.clear();
    this->test_tagMatVec.clear();
    // 将输入数据转化成矩阵存储
    matrix m(feature_num);
    for (size_t i = 0; i < size; i++) {
        m.setZero();
        for (size_t j = 0; j < feature_num; j++)
            m.data[0][j] = feature_vector[j][i];
        this->test_inMatVec.push_back(m);
    }
    // 将监督数据转化为矩阵存储
    matrix n(tag_num);
    for (size_t i = 0; i < size; i++) {
        n.setZero();
        for (size_t j = 0; j < tag_num; j++) n.data[0][j] = tag_vector[j][i];
        this->test_tagMatVec.push_back(n);
    }
    delete[] feature_vector;
    delete[] tag_vector;
    cache.clear();
}

void bpnn::train(size_t train_times, size_t batch_size) {
    TRY if (batch_size >= this->size) throw "选取数据数目应小于样本容量！";
    CATCH
    // 学习率
    double learning_rate = 0.0005;
    double weight_init_std = 0.01;
    // 初始化权值
    matrix w1(this->input_num, this->hidden_num);
    w1.randomMatrix(0, 1);
    w1 = w1 * weight_init_std;
    // 初始化偏置
    matrix b1(batch_size, this->hidden_num);
    // 初始化第一个Affine层
    affLayer affine1(w1, b1);
    // 初始化激活函数层
    sigLayer sigmoid;
    // 初始化权值
    matrix w2(this->hidden_num, this->output_num);
    w2.randomMatrix(0, 1);
    w2 = w2 * weight_init_std;
    // 初始化偏置
    matrix b2(batch_size, this->output_num);
    // 初始化第二个Affine层
    affLayer affine2(w2, b2);
    // 初始化输出层
    sofLayer last;
    double loss = 0;
    matrix dout(this->output_num);
    for (size_t i = 0; i < this->output_num; i++) dout.data[0][i] = 1;
    matrix xin(batch_size, this->input_num);
    matrix tin(batch_size, this->output_num);
    std::default_random_engine eng;
    eng.seed(time(NULL));
    std::uniform_int_distribution<int> uni(0, this->size - 1);
    std::set<size_t> s;
    std::vector<size_t> vec;
    this->lossVec.clear();
    // 开始训练
    for (size_t k = 0; k < train_times; k++) {
        s.clear();
        vec.clear();
        // 随机抽取数据
        // 存储随机的索引值并保证索引值不重复
        while (true) {
            s.insert(uni(eng));
            if (s.size() == batch_size) break;
        }
        // 转移到向量存储
        for (auto i : s) vec.push_back(i);
        // 拓展输入矩阵
        for (size_t i = 0; i < xin.line; i++)
            for (size_t j = 0; j < xin.col; j++)
                xin.data[i][j] = this->inMatVec[vec[i]].data[0][j];
        for (size_t i = 0; i < tin.line; i++)
            for (size_t j = 0; j < tin.col; j++)
                tin.data[i][j] = this->tagMatVec[vec[i]].data[0][j];
        // 前向传播
        loss = last.forward(affine2.forward(sigmoid.forward(affine1.forward(xin))),
                     tin);
        this->lossVec.push_back(loss);
        // 反向传播
        affine1.backward(
            sigmoid.backward(affine2.backward(last.backward(dout))));
        // 更新参数
        affine1.weight = affine1.weight - (affine1.dw * learning_rate);
        affine1.bias = affine1.bias - (affine1.db * learning_rate);
        affine2.weight = affine2.weight - (affine2.dw * learning_rate);
        affine2.bias = affine2.bias - (affine2.db * learning_rate);
    }
    this->weight1 = affine1.weight;
    this->weight2 = affine2.weight;
    this->bias1 = matrix(this->hidden_num);
    for (size_t i = 0; i < this->hidden_num; i++)
        this->bias1.data[0][i] = affine1.bias.data[0][i];
    this->bias2 = matrix(this->output_num);
    for (size_t i = 0; i < this->output_num; i++)
        this->bias2.data[0][i] = affine2.bias.data[0][i];
}
