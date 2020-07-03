#include "bpnn.h"

using namespace std;

size_t main(size_t argc, char* argv[]) {
    bpnn* bp = new bpnn(22, 10, 1);
    bp->readTrainSet("./mushroom.csv");
    bp->train(1000, 500);
    bp->readTestSet("./test.csv");
    cout << "¾«¶È£º" << bp->accuracy() << endl;
    system("pause");
    return 0;
}