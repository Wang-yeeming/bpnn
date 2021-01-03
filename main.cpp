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

int main(int argc, char* argv[]) {
    // ��������ڴ�й©
    //_CrtSetBreakAlloc(1153);
    //_CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);

    bpnn* nn = new bpnn(7, 2, 2);
    nn->readTrainSet("data/train.csv");
    cout << "ѵ���У����Ժ�..." << endl;
    nn->train(500000, 16);
    cout << "ѵ�����" << endl;
    // ��ӡ��ʧֵ
    //ofstream fout("loss2.txt");
    //for (auto i : nn->lossVec)
    //    fout << i << endl;
    //fout.close();
    nn->readTestSet("data/test.csv");
    cout << "���ȣ�" << nn->accuracy() << endl;
    system("pause");
    delete nn;
    return 0;
}