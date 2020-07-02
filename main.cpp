#include "bpnn.h"

using namespace std;

int main(int argc, char* argv[]) {
	bpnn* bp = new bpnn(2, 2);
	bp->readTrainSet("./TrainSet.csv");
	matrix w1(3, 1);
	w1.randomMatrix(-4, 4);
	cout << w1 << endl;
	matrix b1(1);
	cout << b1 << endl;
	affLayer affine = bp->createAffineLayer(w1, b1);
	sigLayer sigmoid = bp->createSigmoidLayer();
	sofLayer last = bp->createSoftmaxWithLossLayer();
	matrix a1 = affine.forward(bp->inMatVec[0]);
	cout << a1 << endl;
	matrix s1 = sigmoid.forward(a1);
	cout << s1 << endl;
	cout << last.forward(s1, bp->tagMatVec[0]) << endl;
	bp->train();
	bp->readTestSet("./TestSet.csv");
	system("pause");
	return 0;
}