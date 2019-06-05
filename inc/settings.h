#pragma once

#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <Windows.h>

using namespace std;

enum NNLayerType
{
    INPUT_LAYER,
    OUTPUT_LAYER,
    HIDDEN_LAYER,
    NUM_LAYER_TYPES
};


struct NNSettings
{
    uint32_t numLayers;
    uint32_t inputSize;
    uint32_t outputSize;
    uint32_t hiddenLayerSize;

    uint32_t miniBatchSize;
    uint32_t numEpochs;
    double learningRate;

    bool useGPU;

    void Load(string &settingsFile);
};