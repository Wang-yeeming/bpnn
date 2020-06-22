#include "bpnn.h"

int main(int argc, char* argv[]) {
	bpnn* bp = new bpnn();
	bp->setOutputNum(1);
	cout << "ÑµÁ·¼¯" << endl;
	bp->readTrainSet("./TestSet.csv");
	cout << "²âÊÔ¼¯" << endl;
	bp->readTestSet("./TestSet.csv");
	system("pause");
	return 0;
}