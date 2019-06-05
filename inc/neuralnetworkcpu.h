#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <vector>
#include <map>
#include <string>
#include <iostream>

#include "dataset.h"
#include "settings.h"
#include "Eigen/Dense"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using namespace std;

struct NNLayerCPU
{
    uint32_t inputSize;
    uint32_t outputSize;

    MatrixXd weights;
    VectorXd biases;
    
    NNLayerType layerType;

    void Init(
        uint32_t inSize,
        uint32_t outSize,
        NNLayerType type
    );

    void EvaluateFull(
        VectorXd &in,
        VectorXd &out,
        VectorXd &aOut,
        VectorXd &spOut
    );
    
    void Evaluate(
        VectorXd &in,
        VectorXd &out
    );
};

ostream& operator<<(ostream &os, NNLayerCPU const &m);

struct NNTrainingScratchCPU
{
    vector<VectorXd> activations;
    vector<VectorXd> zVecs;
    vector<VectorXd> sps;
    vector<VectorXd> deltas;

    vector<VectorXd> nablaBs;
    vector<MatrixXd> nablaWs;
};


struct NNFullCPU
{
    vector<NNLayerCPU> layers;
    uint32_t numLayers;
    uint32_t hiddenLayerSize;
    uint32_t inputSize;
    uint32_t outputSize;

    NNTrainingScratchCPU scratch;

    static void main(
        NNSettings &settings, 
        MNISTDataSet &trainingSet,
        MNISTDataSet &testSet
    );

    void Init(NNSettings &params);
    
    void Evaluate(VectorXd &in, VectorXd &out);
    void BackProp(VectorXd &in, VectorXd &actual);

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

ostream& operator<<(ostream &os, NNFullCPU const &m);