#include "bpnn.h"
#define TRY try {
#define CATCH                          \
    }                                  \
    catch (const char* msg) {          \
        std::cerr << msg << std::endl; \
        system("pause");               \
        exit(-1);                      \
    }

// 神经元节点
typedef struct node_t {
    void* content;
} node_t;

// 权重
typedef struct weight_t {
    double param;  // 权值
    node_t* src;   // 源节点
    node_t* dist;  // 目标节点
} weight_t;

static std::vector<std::string> cache;

bpnn::bpnn() {}

bpnn::bpnn(int i, int h, int o) {
    TRY if (i <= 0) throw "输入层节点数目必须大于0";
    if (h <= 0) throw "隐含层节点数目必须大于0";
    if (o <= 0) throw "输出层节点数目必须大于0";
    CATCH
    this->input_num = i;
    this->hidden_num = h;
    this->output_num = o;
}

bpnn::~bpnn() {
    delete[] this->feature_vector;
    delete[] this->target_vector;
    for (auto it : this->input) delete it;
    for (auto it : this->hidden) delete it;
    for (auto it : this->output) delete it;
}

void bpnn::setInputNum(int num) {
    TRY if (num <= 0) throw "输入层节点数目必须大于0";
    CATCH
    this->input_num = num;
}

void bpnn::setHiddenNum(int num) {
    TRY if (num <= 0) throw "隐含层节点数目必须大于0";
    CATCH
    this->hidden_num = num;
}

void bpnn::setOutputNum(int num) {
    TRY if (num <= 0) throw "输出层节点数目必须大于0";
    CATCH
    this->output_num = num;
}

void bpnn::readTrainSet(std::string path) {
    // 必须指定输出节点数目才能运行
    TRY if (this->output_num == 0) throw "请指定输出节点数目";
    CATCH
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
                    // 若没有指定输入节点数目，则自动生成
                    if (count == 1 && this->input_num == 0) {
                        feature_num = cache.size() - target_num;
                        this->input_num = feature_num;
                    }
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
    // 确保所有节点数目已经确定
    TRY if (this->input_num == 0) throw "请指定输入层节点数目";
    if (this->hidden_num == 0) throw "请指定隐含层节点数目";
    if (this->output_num == 0) throw "请指定输出层节点数目";
    CATCH
    // 初始化
    node_t* nin = new node_t[this->input_num];
    for (unsigned i = 0; i < this->input_num; i++)
        this->input.push_back(&nin[i]);
    node_t* nhid = new node_t[this->hidden_num];
    for (unsigned i = 0; i < this->hidden_num; i++)
        this->hidden.push_back(&nhid[i]);
    node_t* nout = new node_t[this->output_num];
    for (unsigned i = 0; i < this->output_num; i++)
        this->output.push_back(&nout[i]);
}
