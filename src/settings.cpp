#include "settings.h"


void NNSettings::Load()
{
    char pwdBuffer[256];
    GetCurrentDirectory(256, pwdBuffer);
    string pwd(pwdBuffer);

    string settingsFile = pwd + "/resource/settings.txt";

    ifstream fin;
    fin.open(settingsFile);

    string line;
    istringstream iss;

    getline(fin, line);
    iss.str(line);
    iss >> numLayers;

    getline(fin, line);
    iss.str(line);
    iss >> inputSize;

    getline(fin, line);
    iss.str(line);
    iss >> outputSize;

    getline(fin, line);
    iss.str(line);
    iss >> hiddenLayerSize;

    getline(fin, line);
    iss.str(line);
    iss >> miniBatchSize;

    getline(fin, line);
    iss.str(line);
    iss >> numEpochs;

    getline(fin, line);
    iss.str(line);
    iss >> learningRate;

    getline(fin, line);
    iss.str(line);
    string gpuString;
    iss >> gpuString;

    if (gpuString == "true")
    {
        useGPU = true;
    }
    else
    {
        useGPU = false;
    }

    fin.close();
}