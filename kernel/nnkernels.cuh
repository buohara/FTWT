#include <iostream>
#include <vector>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


using namespace std;

void addCUDA(
    vector<int> &c, 
    vector<int> &a,
    vector<int> &b
);

void sigmoids(double *zVec, double *activations, double *sps, int numElements);
void had(double *a, double *b, int numElements);