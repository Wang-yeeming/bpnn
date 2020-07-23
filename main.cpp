#include <windows.h>
#ifdef _DEBUG
#define DEBUG_CLIENTBLOCK new(_CLIENT_BLOCK, __FILE__, __LINE__)
#else
#define DEBUG_CLIENTBLOCK
#endif  // _DEBUG
#define _CRTDBG_MAP_ALLOC
#include <crtdbg.h>
#include <stdlib.h>
#ifdef _DEBUG
#define new DEBUG_CLIENTBLOCK
#endif  // _DEBUG

#include "bpnn.h"

using namespace std;

/* ----- simple UI global variables ----- */
int key1 = 1024;
bpnn nn(0, 0, 0);
size_t s1, s2, s3;
string path;
double tick;
vector<double> v1;
vector<double> v2;

int main(int argc, char* argv[]) {
    /* ----- check memory leaks ----- */
    //_CrtSetBreakAlloc(1153);
    _CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
    /* ----- simple UI ----- */
    while (true) {
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
            cout << "输入csv格式的one-hot训练集" << endl;
            cout << "> ";
            cin >> path;
            nn.readTrainSet(path);
            cout << "输入训练次数" << endl;
            cout << "> ";
            cin >> s2;
            cout << "输入mini-batch选取的样本数目" << endl;
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
            while (true) {
                v1.clear();
                for (size_t i = 0; i < s1; i++) {
                    cout << "输入第" << i + 1 << "个特征" << endl;
                    cout << "> ";
                    cin >> tick;
                    v1.push_back(tick);
                }
                v2 = nn.predict(v1);
                cout << "预测：";
                s2 = 0;
                for (size_t i = 0; i < v2.size(); i++)
                    if (v2[s2] < v2[i]) s2 = i;
                cout << "第" << s2 + 1 << "类" << endl;
                cout << "如需退出请输入exit" << endl;
                cout << "> ";
                cin >> path;
                if (path.compare("exit") == 0) break;
            }
            cout << "成功退出" << endl;
        } else {
            cout << "无效的指令" << endl;
        }
        system("pause");
        system("cls");
    }
    return 0;
}