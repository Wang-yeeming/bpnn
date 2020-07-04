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
            cout << "����csv��ʽ��ѵ����" << endl;
            cout << "> ";
            cin >> path;
            nn.readTrainSet(path);
            cout << "����ѵ������" << endl;
            cout << "> ";
            cin >> s2;
            cout << "����ѡȡ��������Ŀ" << endl;
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
            for (size_t i = 0; i < s1; i++) {
                cout << "�����" << i + 1 << "������" << endl;
                cout << "> ";
                cin >> tick;
                v1.push_back(tick);
            }
            v2 = nn.predict(v1);
            cout << "Ԥ�����£�" << endl;
            for (auto i : v2) cout << i << "  " << endl;
        } else {
            cout << "��Ч��ָ��" << endl;
        }
        system("pause");
        system("cls");
    }
    return 0;
}