#include "bpnn.h"

typedef struct node_t {
    void* content;
} node_t;

static vector<string> cache;

bpnn::bpnn() {}

bpnn::bpnn(int i, int h, int o) {
    try {
        if (i <= 0) throw "输入层节点数目必须大于0";
        if (h <= 0) throw "隐含层节点数目必须大于0";
        if (o <= 0) throw "输出层节点数目必须大于0";
    } catch (const char* msg) {
        cerr << msg << endl;
        system("pause");
        exit(-1);
    }
    this->input_num = i;
    this->hidden_num = h;
    this->output_num = o;
}

void bpnn::setInputNum(int num) {
    try {
        if (num <= 0) throw "输入层节点数目必须大于0";
    } catch (const char* msg) {
        cerr << msg << endl;
        system("pause");
        exit(-1);
    }
    this->input_num = num;
}

void bpnn::setHiddenNum(int num) {
    try {
        if (num <= 0) throw "隐含层节点数目必须大于0";
    } catch (const char* msg) {
        cerr << msg << endl;
        system("pause");
        exit(-1);
    }
    this->hidden_num = num;
}

void bpnn::setOutputNum(int num) {
    try {
        if (num <= 0) throw "输出层节点数目必须大于0";
    } catch (const char* msg) {
        cerr << msg << endl;
        system("pause");
        exit(-1);
    }
    this->output_num = num;
}

void bpnn::readTrainSet(string path) {
    try {  // 必须指定输出节点数目才能运行
        if (this->output_num == 0) throw "请指定输出节点数目";
    } catch (const char* msg) {
        cerr << msg << endl;
        system("pause");
        exit(-1);
    }
    char data[128];  // 缓冲区
    string str;
    ifstream fin;  // 文件输入流
    fin.open(path, ios::in);
    fin >> data;
    string::size_type start, end;  // 分割子串用
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
                if (end != string::npos) {
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
    // 存储特征向量
    node_t* nin = new node_t[feature_num];
    // 存储目标向量
    node_t* nout = new node_t[target_num];
    unsigned tmp;
    unsigned size = cache.size();
    double number;
    stringstream ss;
    for (unsigned i = 1; i <= feature_num + target_num; i++) {
        for (unsigned j = 1; j <= size; j++) {
            if (i == feature_num + target_num)
                tmp = 0;
            else
                tmp = i;
            if (j % (feature_num + target_num) == tmp) {
                if (i <= feature_num) {
                    // 将特征存入特征向量
                    // 将离散型数据转化为连续型数据
                    ss << cache[j - 1];
                    ss >> number;
                    nin[i - 1].content = &number;
                } else {
                    // 将目标存入目标向量
                    nout[i - feature_num - 1].content = &(cache[j - 1]);
                }
            }
        }
    }
    // 生成 i 个输入层节点
    for (unsigned i = 0; i < feature_num; i++) this->input.push_back(&nin[i]);
    // 生成 i 个输出层节点
    for (unsigned i = 0; i < target_num; i++) this->output.push_back(&nout[i]);
}

void bpnn::readTestSet(string path) {
    char data[128];  // 缓冲区
    string str;
    ifstream fin;  // 文件输入流
    fin.open(path, ios::in);
    fin >> data;
    while (fin) {
        // 从第 2 行开始读入数据
        fin >> data;
        str = data;
        if (str.compare("") != 0) cout << str << endl;
    }
}
