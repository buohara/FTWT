#include "tests.h"
#include "mnistdataset.h"

static const string trainImageFile  = "data/mnist/trainimages.txt";
static const string trainLabelFile  = "data/mnist/trainlabels.txt";
static const string testImageFile   = "data/mnist/testimages.txt";
static const string testLabelFile   = "data/mnist/testlabels.txt";

static const uint32_t inputSize     = 784;
static const uint32_t outputSize    = 10;
static const uint32_t numIterations = 1;

vector<Triplet<double>> generateSynapses()
{
    vector<Triplet<double>> triples;

    for (uint32_t i = 0; i < inputSize; i++)
    {
        for (uint32_t j = 0; j < outputSize; j++)
        {
            double scale = 0.01;

            double offset = scale * (double)rand() / (double)RAND_MAX - (0.5 * scale);
            triples.push_back({ i, inputSize + j, 0.0 });

            offset = scale * (double)rand() / (double)RAND_MAX - (0.5 * scale);
            triples.push_back({ inputSize + j, i, offset });
        }
    }

    return triples;
}

void getAssocVector(MNISTDataSet &data, uint32_t idx, vector<pair<uint32_t, double>>& assoc)
{
    assert(assoc.size() == (inputSize + 1));

    for (uint32_t i = 0; i < inputSize; i++)
    {
        assoc[i].first  = i;
        assoc[i].second = data.data[idx][i];
    }

    assoc[inputSize].first  = inputSize + data.labels[idx];
    assoc[inputSize].second = 1.0;
}

void MNISTTest()
{
    MNISTDataSet trainData;
    MNISTDataSet testData;

    trainData.Init(trainImageFile.c_str(), trainLabelFile.data());
    testData.Init(testImageFile.c_str(), testLabelFile.data());

    srand((uint32_t)time(NULL));

    vector<Triplet<double>> synapseTriples = generateSynapses();
    NN<double> nn("MNIST Digit Net", inputSize + outputSize, synapseTriples);
    vector<pair<uint32_t, double>> assoc(inputSize + 1);

    for (uint32_t i = 0; i < numIterations; i++)
    {
        for (uint32_t j = 0; j < trainData.numImgs; j++)
        {
            if (j % 100 == 0 && j > 0) printf("Training image %d...\n", j);

            if (trainData.labels[j] != 7) continue;

            getAssocVector(trainData, j, assoc);

            nn.applyAssocs(assoc, 1);
            nn.computePairings();
            nn.updateSynapses();
        }
    }

    vector<double> testVec(inputSize + outputSize, 0.0);

    nn.print(true);

    for (uint32_t i = 0; i < testData.numImgs; i++)
    {
        memcpy(&testVec[0], &testData.data[i][0], inputSize * sizeof(double));

        vector<double> res = nn.applyInput(testVec);
        uint32_t label = testData.labels[i];
    }
}