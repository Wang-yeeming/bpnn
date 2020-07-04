#include <Windows.h>

#include "bpnn.h"

using namespace std;

int key1 = 1024;
bpnn nn(0, 0, 0);
size_t s1, s2, s3;
string path;
double tick;
vector<double> v1;
vector<double> v2;

int main(int argc, char* argv[]) {
    while (true) {
        v1.clear();
        cout << "┏━━━━━━━━━━━━━━━━━━━━━━━━━━┓" << endl;
        cout << "┃    简易3层BP神经网络     ┃" << endl;
        cout << "┃                          ┃" << endl;
        cout << "┃       输入指令开始       ┃" << endl;
        cout << "┃    0. 退出程序           ┃" << endl;
        cout << "┃    1. 训练神经网络       ┃" << endl;
        cout << "┗━━━━━━━━━━━━━━━━━━━━━━━━━━┛" << endl;
        cout << "请输入：" << endl;
        cout << "> ";
        cin >> key1;
        if (key1 == 0)
            break;
        else if (key1 == 1) {
            cout << "输入输入层神经元数目（特征数目）" << endl;
            cout << "> ";
            cin >> s1;
            nn.setInputSize(s1);
            cout << "输入隐含层神经元数目" << endl;
            cout << "> ";
            cin >> s2;
            nn.setHiddenSize(s2);
            cout << "输入输出层神经元数目（待分类数目）" << endl;
            cout << "> ";
            cin >> s3;
            nn.setOutputSize(s3);
            cout << "输入csv格式的训练集" << endl;
            cout << "> ";
            cin >> path;
            nn.readTrainSet(path);
            cout << "输入训练次数" << endl;
            cout << "> ";
            cin >> s2;
            cout << "输入选取的样本数目" << endl;
            cout << "> ";
            cin >> s3;
            cout << "训练中......" << endl;
            tick = GetTickCount();
            nn.train(s2, s3);
            cout << "训练完成！用时" << (GetTickCount() - tick) / 1000 << "s"
                 << endl;
            cout << "输入csv格式的测试集" << endl;
            cout << "> ";
            cin >> path;
            nn.readTestSet(path);
            cout << "精度：" << nn.accuracy() << endl;
            cout << "现在开始预测" << endl;
            for (size_t i = 0; i < s1; i++) {
                cout << "输入第" << i + 1 << "个特征" << endl;
                cout << "> ";
                cin >> tick;
                v1.push_back(tick);
            }
            v2 = nn.predict(v1);
            cout << "预测如下：" << endl;
            for (auto i : v2) cout << i << "  " << endl;
        } else {
            cout << "无效的指令" << endl;
        }
        system("pause");
        system("cls");
    }
    return 0;
}