#include "bpnn.h"

using namespace std;

int main(int argc, char* argv[]) {
	bpnn* bp = new bpnn(2, 2);
	bp->readTrainSet("./TrainSet.csv");
	matrix w1(2, 2);
	w1.randomMatrix(-4, 4);
	matrix b1(2);
	affLayer affine1 = bp->createAffineLayer(w1, b1);
	sigLayer sigmoid1 = bp->createSigmoidLayer();
	sofLayer last = bp->createSoftmaxWithLossLayer();
	matrix a1(2);
	matrix s1(2);
	for (size_t i = 0; i < bp->size; i++) {
		a1 = affine1.forward(bp->inMatVec[i]);
		s1 = sigmoid1.forward(a1);
		cout << "Îó²î£º" << last.forward(s1, bp->tagMatVec[i]) << endl;
	}
	system("pause");
	return 0;
}