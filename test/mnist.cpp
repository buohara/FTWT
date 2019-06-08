#include "tests.h"
#include "mnistdataset.h"

static const string trainImageFile  = "data/mnist/trainimages.txt";
static const string trainLabelFile  = "data/mnist/trainlabels.txt";
static const string testImageFile   = "data/mnist/testimages.txt";
static const string testLabelFile   = "data/mnist/testlabels.txt";

static const uint32_t inputSize     = 784;
static const uint32_t outputSize    = 10;
static const uint32_t numIterations = 10;
static const uint32_t batchSize     = 100;
static const uint32_t pulseLength   = 1;
static const double learnRate       = 0.01;
static const double cullThresh      = 1e-8;

/**
 * generateSynapses - Randomly initialize synapse weights for MNIST NN.
 *
 * @return List of synapse (to, from, weight) triples.
 */

vector<Triplet<double>> generateSynapses()
{
    vector<Triplet<double>> triples;

    for (uint32_t i = 0; i < inputSize; i++)
    {
        for (uint32_t j = 0; j < outputSize; j++)
        {
            double scale = 1.0;

            double offset = scale * (double)rand() / (double)RAND_MAX - (0.5 * scale);
            triples.push_back({ i, inputSize + j, offset });

            offset = scale * (double)rand() / (double)RAND_MAX - (0.5 * scale);
            triples.push_back({ inputSize + j, i, offset });
        }
    }

    return triples;
}

/**
 * getAssocBatch - From MNIST data set, generate a set of pre and post synapse activation
 * associations.
 *
 * @param data      MNIST image data set.
 * @param start     Offset from beginning of data set to start building batch.
 * @param batchSize Size of batch to create.
 * @param assocPre  List of presynaptic activations to populate.
 * @param assocPost List of postsynaptic activations to populate.
 */

void getAssocBatch(
    MNISTDataSet &data,
    uint32_t start,
    uint32_t batchSize, 
    vector<vector<pair<uint32_t, double>>>& assocPre,
    vector<vector<pair<uint32_t, double>>>& assocPost)
{
    for (uint32_t i = 0; i < batchSize; i++)
    {
        if (start + i >= data.numImgs) return;

        for (uint32_t j = 0; j < inputSize; j++)
        {
            assocPre[i][j].first    = j;
            assocPre[i][j].second   = data.data[start + i][j];
        }

        assocPost[i][0].first       = inputSize + data.labels[start + i];
        assocPost[i][0].second      = 1.0;
    }
}

/**
 * MNISTTest - MNIST test driver routine. Load data set. Loop over association batches
 * and train NN. Report statistics and training time. Test NN on test set and report
 * accuracy.
 */

void MNISTTest()
{
    MNISTDataSet trainData;
    MNISTDataSet testData;

    trainData.Init(trainImageFile.c_str(), trainLabelFile.data());
    testData.Init(testImageFile.c_str(), testLabelFile.data());

    srand((uint32_t)time(NULL));

    NNCreateParams<double> params;
    params.batchSize    = batchSize;
    params.name         = "MNIST Digit Net";
    params.numNeurons   = inputSize + outputSize;
    params.synapsesIn   = generateSynapses();
    params.learnRate    = learnRate;
    params.cullThresh   = cullThresh;

    NN<double> nn(params);
    
    vector<vector<pair<uint32_t, double>>> assocPre(batchSize);
    vector<vector<pair<uint32_t, double>>> assocPost(batchSize);

    for (uint32_t i = 0; i < batchSize; i++)
    {
        assocPre[i].resize(inputSize);
        assocPost[i].resize(1);
    }

    printf("Training MNIST Digit Images\n");
    printf("Number of training set passes: %d\n", numIterations);
    printf("Training batch size: %d\n", batchSize);

    long long t1 = GetMilliseconds();

    for (uint32_t i = 0; i < numIterations; i++)
    {
        printf("Training iteration %d ...\n", i);

        for (uint32_t j = 0; j < trainData.numImgs; j += batchSize)
        {
            getAssocBatch(trainData, j, batchSize, assocPre, assocPost);
            nn.applyAssocs(assocPre, assocPost, pulseLength);
            nn.computePairings();
            nn.updateSynapses();
        }

        nn.cull();
    }

    long long t2 = GetMilliseconds();

    double trainingTime = ((double)(t2 - t1)) / 1000.0;
    printf("FTWT MNIST Training Time: %g sec\n", trainingTime);

    vector<double> testVec(inputSize + outputSize, 0.0);

    uint32_t correctCnt = 0;

    for (uint32_t i = 0; i < testData.numImgs; i++)
    {
        memcpy(&testVec[0], &testData.data[i][0], inputSize * sizeof(double));
        vector<double> res = nn.applyInput(testVec);

        double max = -50.0;
        uint32_t outIdx = outputSize;

        for (uint32_t i = 0; i < outputSize; i++)
        {
            if (res[inputSize + i] > max)
            {
                max = res[inputSize + i];
                outIdx = i;
            }
        }

        uint32_t label = testData.labels[i];

        if (label == outIdx) correctCnt++;
    }

    double accuracy = 100.0 * (double)correctCnt / (double)testData.numImgs;
    printf("NN test accuracy=%g%%\n", accuracy);
}