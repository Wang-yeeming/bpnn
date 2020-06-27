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
    // ���ɾ��ȷֲ������������ʼ��Ȩ��
    std::default_random_engine eng;
    std::uniform_real_distribution<double> u(0, 1);
    // ��ʼ��Ȩ��
    matrix* w1 = new matrix(this->hidden_num);
    double* arr1 = new double[this->hidden_num];
    for (size_t i = 0; i < this->hidden_num; i++) arr1[i] = u(eng);
    w1->input(arr1);
    // תΪ������
    *w1 = ~(*w1);
    delete[] arr1;
    this->params["w1"] = *w1;
    // ƫ����Ϊ0
    matrix* b1 = new matrix(this->hidden_num);
    this->params["b1"] = *b1;
}

bpnn::~bpnn() {
    delete[] this->feature_vector;
    delete[] this->target_vector;
}

void bpnn::readTrainSet(std::string path) {
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
    // ��ʼ��
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
