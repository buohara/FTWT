#include "nnkernels.cuh"

/**
 * [addKernel description]
 * @param c [description]
 * @param a [description]
 * @param b [description]
 */

__global__ 
void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

/**
 * [hadamard description]
 * @param a [description]
 * @param b [description]
 */

__global__
void hadamard(double *a, double *b)
{
    int i = threadIdx.x;
    a[i] = a[i] * b[i];
}

/**
 * [sigmoid description]
 * @param zVec        [description]
 * @param activations [description]
 * @param sps         [description]
 */

__global__
void sigmoid(double *zVec, double *activations, double *sps)
{
    int i = threadIdx.x;

    double sig = 1.0 / (1.0 + exp(-zVec[i]));
    activations[i] = sig;
    sps[i] = sig * (1.0 - sig);
}

/**
 * [had description]
 * @param a           [description]
 * @param b           [description]
 * @param numElements [description]
 */

void had(double *a, double *b, int numElements)
{
    hadamard <<< 1, numElements >>> (a, b);
    return;
}

/**
 * [sigmoids description]
 * @param zVec        [description]
 * @param activations [description]
 * @param sps         [description]
 * @param numElements [description]
 */

void sigmoids(double *zVec, double *activations, double *sps, int numElements)
{
    sigmoid <<< 1, numElements >>> (zVec, activations, sps);
    return;
}