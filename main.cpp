#include "bpnn.h"

using namespace std;

int main(int argc, char* argv[]) {
	bpnn* bp = new bpnn(2, 2, 2);
	bp->readTrainSet("./TrainSet.csv");
	bp->train(1000, 6);
	system("pause");
	return 0;
}