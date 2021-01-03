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
    // 检查有无内存泄漏
    //_CrtSetBreakAlloc(1153);
    //_CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);

    bpnn* nn = new bpnn(7, 2, 2);
    nn->readTrainSet("data/train.csv");
    cout << "训练中，请稍后..." << endl;
    nn->train(500000, 16);
    cout << "训练完成" << endl;
    // 打印损失值
    //ofstream fout("loss2.txt");
    //for (auto i : nn->lossVec)
    //    fout << i << endl;
    //fout.close();
    nn->readTestSet("data/test.csv");
    cout << "精度：" << nn->accuracy() << endl;
    system("pause");
    delete nn;
    return 0;
}