#include "bpnn.h"

int main(int argc, char* argv[]) {
	bpnn* bp = new bpnn();
	bp->setOutputNum(1);
	cout << "ѵ����" << endl;
	bp->readTrainSet("./TestSet.csv");
	cout << "���Լ�" << endl;
	bp->readTestSet("./TestSet.csv");
	system("pause");
	return 0;
}