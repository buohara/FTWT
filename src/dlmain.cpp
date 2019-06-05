#include <string>
#include <stdio.h>
#include <Windows.h>
#include "dataset.h"
#include "neuralnetworkcpu.h"
#include "neuralnetworkgpu.h"
#include "settings.h"


/**
 * [GetDataFilePaths description]
 * @param trainingImgPath [description]
 * @param trainingLblPath [description]
 * @param testImgPath     [description]
 * @param testLblPath     [description]
 */

void GetDataFilePaths(
    string &trainingImgPath,
    string &trainingLblPath,
    string &testImgPath,
    string &testLblPath
)
{
    char pwdBuffer[256];
    GetCurrentDirectory(256, pwdBuffer);
    string pwd(pwdBuffer);

    trainingImgPath = pwd + "/trainimages.txt";
    trainingLblPath = pwd + "/trainlabels.txt";
    testImgPath     = pwd + "/testimages.txt";
    testLblPath     = pwd + "/testlabels.txt";
}

/**
 * [InitData description]
 * @param trainingSet [description]
 * @param testSet     [description]
 */

void InitData(
    MNISTDataSet &trainingSet,
    MNISTDataSet &testSet
)
{
    string trainingImgPath;
    string trainingLblPath;
    string testImgPath;
    string testLblPath;

    GetDataFilePaths(trainingImgPath,
        trainingLblPath,
        testImgPath,
        testLblPath
    );

    trainingSet.Init(
        trainingImgPath.c_str(),
        trainingLblPath.c_str()
    );

    testSet.Init(
        testImgPath.c_str(),
        testLblPath.c_str()
    );
}

/**
 * [main description]
 * @param  argc [description]
 * @param  argv [description]
 * @return      [description]
 */

int main(int argc, char** argv)
{
    NNSettings settings;
    settings.Load(string("/settings.txt"));

    MNISTDataSet trainingSet;
    MNISTDataSet testSet;
    InitData(trainingSet, testSet);

    if (settings.useGPU == false)
    {
        NNFullCPU::main(settings, trainingSet, testSet);
    }
    else
    {
        NNFullGPU::main(settings, trainingSet, testSet);
    }

    return 0;
}