#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <vector>
#include <string>
#include <iostream>

#include <curand.h>
#include <cublas_v2.h>

#include "dataset.h"
#include "settings.h"
#include "utils.h"

struct NNLayerGPU
{
    uint32_t inputSize;
    uint32_t outputSize;

    void Init(
        uint32_t inputSize,
        uint32_t outputSize,
        NNLayerType type
    );

    NNLayerType layerType;

    void EvaluateFull(
        double *in,
        double *out,
        double *aOut,
        double *spOut
    );

    void Evaluate(
        double *in,
        double *out
    );

    double *weights;
    double *biases;
};

struct NNTrainingScratchGPU
{
    vector<double*> activations;
    vector<double*> zVecs;
    vector<double*> sps;
    vector<double*> deltas;

    vector<double*> nablaBs;
    vector<double*> nablaWs;

    double *actual;
};

struct NNFullGPU
{
    vector<NNLayerGPU> layers;

    uint32_t numLayers;
    uint32_t hiddenLayerSize;
    uint32_t inputSize;
    uint32_t outputSize;

    NNTrainingScratchGPU scratch;

    static void main(
        NNSettings &settings,
        MNISTDataSet &trainingSet,
        MNISTDataSet &testSet
    );

    void Init(NNSettings &params);
    
    void Evaluate(VectorXd &in, VectorXd &out);
    void BackProp(double *imgIn, double *actual);

    void SGDStepMiniBatch(
        MNISTDataSet &ds,
        vector<uint32_t> &idcs,
        uint32_t miniBatchSize,
        double learningRate
    );

    void InitTrainingScratch();
    void ZeroGradient();

    void Train(MNISTDataSet &ds, NNSettings &learnParams);
    void Test(MNISTDataSet &testSet);
};