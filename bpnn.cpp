#include "bpnn.h"

typedef struct node_t {
    void* content;
} node_t;

static vector<string> cache;

bpnn::bpnn() {}

bpnn::bpnn(int i, int h, int o) {
    try {
        if (i <= 0) throw "�����ڵ���Ŀ�������0";
        if (h <= 0) throw "������ڵ���Ŀ�������0";
        if (o <= 0) throw "�����ڵ���Ŀ�������0";
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
        if (num <= 0) throw "�����ڵ���Ŀ�������0";
    } catch (const char* msg) {
        cerr << msg << endl;
        system("pause");
        exit(-1);
    }
    this->input_num = num;
}

void bpnn::setHiddenNum(int num) {
    try {
        if (num <= 0) throw "������ڵ���Ŀ�������0";
    } catch (const char* msg) {
        cerr << msg << endl;
        system("pause");
        exit(-1);
    }
    this->hidden_num = num;
}

void bpnn::setOutputNum(int num) {
    try {
        if (num <= 0) throw "�����ڵ���Ŀ�������0";
    } catch (const char* msg) {
        cerr << msg << endl;
        system("pause");
        exit(-1);
    }
    this->output_num = num;
}

void bpnn::readTrainSet(string path) {
    try {  // ����ָ������ڵ���Ŀ��������
        if (this->output_num == 0) throw "��ָ������ڵ���Ŀ";
    } catch (const char* msg) {
        cerr << msg << endl;
        system("pause");
        exit(-1);
    }
    char data[128];  // ������
    string str;
    ifstream fin;  // �ļ�������
    fin.open(path, ios::in);
    fin >> data;
    string::size_type start, end;  // �ָ��Ӵ���
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
                if (end != string::npos) {
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
    // �洢��������
    node_t* nin = new node_t[feature_num];
    // �洢Ŀ������
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
                    // ������������������
                    // ����ɢ������ת��Ϊ����������
                    ss << cache[j - 1];
                    ss >> number;
                    nin[i - 1].content = &number;
                } else {
                    // ��Ŀ�����Ŀ������
                    nout[i - feature_num - 1].content = &(cache[j - 1]);
                }
            }
        }
    }
    // ���� i �������ڵ�
    for (unsigned i = 0; i < feature_num; i++) this->input.push_back(&nin[i]);
    // ���� i �������ڵ�
    for (unsigned i = 0; i < target_num; i++) this->output.push_back(&nout[i]);
}

void bpnn::readTestSet(string path) {
    char data[128];  // ������
    string str;
    ifstream fin;  // �ļ�������
    fin.open(path, ios::in);
    fin >> data;
    while (fin) {
        // �ӵ� 2 �п�ʼ��������
        fin >> data;
        str = data;
        if (str.compare("") != 0) cout << str << endl;
    }
}
