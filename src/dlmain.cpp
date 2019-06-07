#include <string>
#include <stdio.h>
#include <Windows.h>
#include "dataset.h"
#include "neuralnetworkcpu.h"
#include "neuralnetworkgpu.h"
#include "settings.h"


/**
 * GetDataFilePaths - Get training and image data file paths. Expected to be in
 * executable folder /data/mnist subdirectory. Will probably make this a command line
 * or settings file argument in the future.
 * 
 * @param trainingImgPath Path to training images.
 * @param trainingLblPath Path to training image labels
 * @param testImgPath     Path to test images.
 * @param testLblPath     Path to test image labels.
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

    trainingImgPath = pwd + "/data/mnist/trainimages.txt";
    trainingLblPath = pwd + "/data/mnist/trainlabels.txt";
    testImgPath     = pwd + "/data/mnist/testimages.txt";
    testLblPath     = pwd + "/data/mnist/testlabels.txt";
}

/**
 * InitData Load MNIST training and test data from file.
 *
 * @param trainingSet Dataset to fill with training data.
 * @param testSet     Dataset to fill with test data.
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
 * main - Run a NN model on MNIST digit image data. Load settings
 * and data from file. Execute on CPU/GPU depending on settings file.
 *
 * @param  argc Command line argument count.
 * @param  argv List of command line strings.
 * @return      Main success code. Default zero.
 */

int main(int argc, char** argv)
{
    NNSettings settings;
    settings.Load();

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