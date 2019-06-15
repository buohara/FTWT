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

static const uint32_t minVerts      = inputSize + outputSize;
static const uint32_t maxVerts      = inputSize + outputSize + 500; 
static const double minEdge         = 1e-6;
static const double maxEdge         = 100.0;
static const double edgeProb        = 0.7;

void PickRandomInputsAndOutputs(
    uint32_t numNodes,
    vector<uint32_t>& inputs,
    vector<uint32_t>& outputs
)
{
    assert(inputs.size() == 0);
    assert(outputs.size() == 0);
    assert(numNodes >= inputSize + outputSize);

    vector<uint32_t> verts(numNodes);
    for (uint32_t i = 0; i < inputSize + outputSize; i++) verts[i] = i;

    random_shuffle(verts.begin(), verts.end());

    for (uint32_t i = 0; i < inputSize; i++) inputs[i] = verts[i];
    for (uint32_t i = 0; i < outputSize; i++) outputs[i] = verts[inputSize + i];
}

void MNISTRandTest()
{
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
}