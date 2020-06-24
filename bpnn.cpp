#include "bpnn.h"
#define TRY try {
#define CATCH                          \
    }                                  \
    catch (const char* msg) {          \
        std::cerr << msg << std::endl; \
        system("pause");               \
        exit(-1);                      \
    }

// ��Ԫ�ڵ�
typedef struct node_t {
    void* content;
} node_t;

// Ȩ��
typedef struct weight_t {
    double param;  // Ȩֵ
    node_t* src;   // Դ�ڵ�
    node_t* dist;  // Ŀ��ڵ�
} weight_t;

static std::vector<std::string> cache;

bpnn::bpnn() {}

bpnn::bpnn(int i, int h, int o) {
    TRY if (i <= 0) throw "�����ڵ���Ŀ�������0";
    if (h <= 0) throw "������ڵ���Ŀ�������0";
    if (o <= 0) throw "�����ڵ���Ŀ�������0";
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
    TRY if (num <= 0) throw "�����ڵ���Ŀ�������0";
    CATCH
    this->input_num = num;
}

void bpnn::setHiddenNum(int num) {
    TRY if (num <= 0) throw "������ڵ���Ŀ�������0";
    CATCH
    this->hidden_num = num;
}

void bpnn::setOutputNum(int num) {
    TRY if (num <= 0) throw "�����ڵ���Ŀ�������0";
    CATCH
    this->output_num = num;
}

void bpnn::readTrainSet(std::string path) {
    // ����ָ������ڵ���Ŀ��������
    TRY if (this->output_num == 0) throw "��ָ������ڵ���Ŀ";
    CATCH
    char data[128];  // ������
    std::string str;
    std::ifstream fin;  // �ļ�������
    fin.open(path, std::ios::in);
    // ����ļ��Ƿ���Ч
    TRY if (!fin) throw "���ļ�ʧ��";
    CATCH
    fin >> data;
    std::string::size_type start, end;  // �ָ��Ӵ���
    cache.clear();
    int count = 0;                         // ��ȡ����������
    size_t feature_num = this->input_num;  // ����������Ŀ
    size_t target_num = this->output_num;  // Ŀ��������Ŀ
    while (fin) {
        count++;
        // �ӵ� 2 �п�ʼ��������
        fin >> data;
        str = data;
        start = 0;
        if (str.compare("") != 0) {
            // �ָ��ַ���
            while (true) {
                end = str.find(",");
                if (end != std::string::npos) {
                    cache.push_back(str.substr(start, end - start));
                    start = 0;
                    str = str.substr(end + 1, str.length() - end - 1);
                } else {
                    cache.push_back(str);
                    // ��û��ָ������ڵ���Ŀ�����Զ�����
                    if (count == 1 && this->input_num == 0) {
                        feature_num = cache.size() - target_num;
                        this->input_num = feature_num;
                    }
                    break;
                }
            }
        }
    }
    // �洢����������
    this->feature_vector = new std::vector<double>[feature_num];
    // �洢Ŀ��������
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
                    // ����ɢ������ת��Ϊ����������
                    ss.clear();
                    ss << cache[j - 1];
                    ss >> number;
                    // ������������������
                    feature_vector[i - 1].push_back(number);
                } else {
                    // ��Ŀ�����Ŀ������
                    target_vector[i - feature_num - 1].push_back(cache[j - 1]);
                }
            }
        }
    }
}

void bpnn::readTestSet(std::string path) {
    char data[128];  // ������
    std::string str;
    std::ifstream fin;  // �ļ�������
    fin.open(path, std::ios::in);
    fin >> data;
    while (fin) {
        // �ӵ� 2 �п�ʼ��������
        fin >> data;
        str = data;
        if (str.compare("") != 0) std::cout << str << std::endl;
    }
}

void bpnn::train() {
    // ȷ�����нڵ���Ŀ�Ѿ�ȷ��
    TRY if (this->input_num == 0) throw "��ָ�������ڵ���Ŀ";
    if (this->hidden_num == 0) throw "��ָ��������ڵ���Ŀ";
    if (this->output_num == 0) throw "��ָ�������ڵ���Ŀ";
    CATCH
    // ��ʼ��
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
