#pragma once

#include <stdint.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include "cuda_runtime.h"


using namespace std;

void print(float *dev, uint32_t m, uint32_t n, bool truncateOutput = true);
void print(double *dev, uint32_t m, uint32_t n, bool truncateOutput = true);