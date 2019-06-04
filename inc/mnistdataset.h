#pragma once

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

using namespace std;

struct MNISTDataSet
{
    void Init(const char* dataFile, const char* labelFile);

    vector<vector<double>> data;
    vector<uint8_t> labels;

    uint32_t numImgs;
    uint32_t imgSize;
};