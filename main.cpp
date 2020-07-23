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
        cout << "��������������������������������������������������������" << endl;
        cout << "��    ����3��BP������     ��" << endl;
        cout << "��                          ��" << endl;
        cout << "��       ����ָ�ʼ       ��" << endl;
        cout << "��    0. �˳�����           ��" << endl;
        cout << "��    1. ѵ��������       ��" << endl;
        cout << "��������������������������������������������������������" << endl;
        cout << "�����룺" << endl;
        cout << "> ";
        cin >> key1;
        if (key1 == 0)
            break;
        else if (key1 == 1) {
            cout << "�����������Ԫ��Ŀ��������Ŀ��" << endl;
            cout << "> ";
            cin >> s1;
            nn.setInputSize(s1);
            cout << "������������Ԫ��Ŀ" << endl;
            cout << "> ";
            cin >> s2;
            nn.setHiddenSize(s2);
            cout << "�����������Ԫ��Ŀ����������Ŀ��" << endl;
            cout << "> ";
            cin >> s3;
            nn.setOutputSize(s3);
            cout << "����csv��ʽ��one-hotѵ����" << endl;
            cout << "> ";
            cin >> path;
            nn.readTrainSet(path);
            cout << "����ѵ������" << endl;
            cout << "> ";
            cin >> s2;
            cout << "����mini-batchѡȡ��������Ŀ" << endl;
            cout << "> ";
            cin >> s3;
            cout << "ѵ����......" << endl;
            tick = GetTickCount();
            nn.train(s2, s3);
            cout << "ѵ����ɣ���ʱ" << (GetTickCount() - tick) / 1000 << "s"
                 << endl;
            cout << "����csv��ʽ�Ĳ��Լ�" << endl;
            cout << "> ";
            cin >> path;
            nn.readTestSet(path);
            cout << "���ȣ�" << nn.accuracy() << endl;
            cout << "���ڿ�ʼԤ��" << endl;
            while (true) {
                v1.clear();
                for (size_t i = 0; i < s1; i++) {
                    cout << "�����" << i + 1 << "������" << endl;
                    cout << "> ";
                    cin >> tick;
                    v1.push_back(tick);
                }
                v2 = nn.predict(v1);
                cout << "Ԥ�⣺";
                s2 = 0;
                for (size_t i = 0; i < v2.size(); i++)
                    if (v2[s2] < v2[i]) s2 = i;
                cout << "��" << s2 + 1 << "��" << endl;
                cout << "�����˳�������exit" << endl;
                cout << "> ";
                cin >> path;
                if (path.compare("exit") == 0) break;
            }
            cout << "�ɹ��˳�" << endl;
        } else {
            cout << "��Ч��ָ��" << endl;
        }
        system("pause");
        system("cls");
    }
    return 0;
}