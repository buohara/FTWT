#include "tests.h"
#include "mnistdataset.h"

static const string trainImageFile  = "data/mnist/trainimages.txt";
static const string trainLabelFile  = "data/mnist/trainlabels.txt";
static const string testImageFile   = "data/mnist/testimages.txt";
static const string testLabelFile   = "data/mnist/testlabels.txt";

static const uint32_t inputSize     = 784;
static const uint32_t outputSize    = 10;
static const uint32_t numIterations = 1;
static const uint32_t batchSize     = 100;
static const uint32_t pulseLength   = 1;
static const double learnRate       = 0.01;
static const double cullThresh      = 1e-8;

static const uint32_t minVerts      = inputSize + outputSize;
static const uint32_t maxVerts      = inputSize + outputSize + 500; 
static const double minEdge         = 1e-6;
static const double maxEdge         = 100.0;
static const double edgeProb        = 0.3;

/**
 * PickRandomInputsAndOutputs - Take a list of all verts in a random graph.
 * Randomly shuffle the list and grab input and output verts from front of
 * shuffled list.
 */

void PickRandomInputsAndOutputs(
    uint32_t numNodes,
    vector<uint32_t>& inputs,
    vector<uint32_t>& outputs
)
{
    assert(inputs.size() == 0);
    assert(outputs.size() == 0);
    assert(numNodes >= inputSize + outputSize);

    inputs.resize(inputSize);
    outputs.resize(outputSize);

    vector<uint32_t> verts(numNodes);
    for (uint32_t i = 0; i < inputSize + outputSize; i++) verts[i] = i;

    random_shuffle(verts.begin(), verts.end());

    for (uint32_t i = 0; i < inputSize; i++) inputs[i] = verts[i];
    for (uint32_t i = 0; i < outputSize; i++) outputs[i] = verts[inputSize + i];
}

/**
 * getAssocBatch - From MNIST data set, generate a set of preand post synapse activation
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
    vector<uint32_t> &inputs,
    vector<uint32_t> &outputs,
    vector<vector<pair<uint32_t, double>>> &assocPre,
    vector<vector<pair<uint32_t, double>>> &assocPost)
{
    for (uint32_t i = 0; i < batchSize; i++)
    {
        if (start + i >= data.numImgs) return;

        for (uint32_t j = 0; j < inputSize; j++)
        {
            assocPre[i][j].first    = inputs[j];
            assocPre[i][j].second   = data.data[start + i][j];
        }

        assocPost[i][0].first   = outputs[data.labels[start + i]];
        assocPost[i][0].second  = 1.0;
    }
}

/**
 * MNISTRandTest - Train a random graph to learn MNIST hand digits using
 * Hebbian learning. Loop over batches, report training time and NN
 * accuracy.
 */

void MNISTRandTest()
{
    // 1. Initiaze data and random graph.

    MNISTDataSet trainData;
    MNISTDataSet testData;

    trainData.Init(trainImageFile.c_str(), trainLabelFile.data());
    testData.Init(testImageFile.c_str(), testLabelFile.data());

    srand((uint32_t)time(NULL));

    RandomGraph graph(minVerts, maxVerts, edgeProb, minEdge, maxEdge);
    vector<Triplet<double>> synapses = graph.GetEdgeTriplets();

    NNCreateParams<double> params;
    params.batchSize    = batchSize;
    params.name         = "MNIST Digit Net Random";
    params.numNeurons   = graph.numVerts;
    params.synapsesIn   = synapses;
    params.learnRate    = learnRate;
    params.cullThresh   = cullThresh;

    NN<double> nn(params);

    vector<uint32_t> inputs;
    vector<uint32_t> outputs;

    PickRandomInputsAndOutputs(graph.numVerts, inputs, outputs);

    vector<vector<pair<uint32_t, double>>> assocPre(batchSize);
    vector<vector<pair<uint32_t, double>>> assocPost(batchSize);

    for (uint32_t i = 0; i < batchSize; i++)
    {
        assocPre[i].resize(inputSize);
        assocPost[i].resize(1);
    }

    // 2. Train

    printf("Training MNIST Digit Images\n");
    printf("Number of training set passes: %d\n", numIterations);
    printf("Training batch size: %d\n", batchSize);

    long long t1 = GetMilliseconds();

    for (uint32_t i = 0; i < numIterations; i++)
    {
        printf("Training iteration %d ...\n", i);

        for (uint32_t j = 0; j < trainData.numImgs; j += batchSize)
        {
            getAssocBatch(trainData, j, batchSize, inputs, outputs, assocPre, assocPost);
            nn.applyAssocs(assocPre, assocPost, pulseLength);
            nn.computePairings();
            nn.updateSynapses();
        }

        nn.cull();
    }

    long long t2 = GetMilliseconds();

    // 3. Test

    double trainingTime = ((double)(t2 - t1)) / 1000.0;
    printf("FTWT MNIST Training Time: %g sec\n", trainingTime);

    vector<double> testVec(nn.numNeurons);

    uint32_t correctCnt = 0;

    for (uint32_t i = 0; i < testData.numImgs; i++)
    {
        memset(&testVec[0], 0, nn.numNeurons * sizeof(double));

        for (uint32_t j = 0; j < inputSize; j++)
        {
            testVec[inputs[j]] = testData.data[i][j];
        }

        vector<double> res = nn.applyInput(testVec);

        double max = -50.0;
        uint32_t outIdx = outputSize;

        for (uint32_t i = 0; i < outputSize; i++)
        {
            if (res[outputs[i]] > max)
            {
                max = res[outputs[i]];
                outIdx = i;
            }
        }

        uint32_t label = testData.labels[i];

        if (label == outIdx) correctCnt++;
    }

    double accuracy = 100.0 * (double)correctCnt / (double)testData.numImgs;
    printf("NN test accuracy=%g%%\n", accuracy);
}