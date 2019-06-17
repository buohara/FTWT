#include "tests.h"
#include "mnistdataset.h"

static const string trainImageFile  = "data/mnist/trainimages.txt";
static const string trainLabelFile  = "data/mnist/trainlabels.txt";
static const string testImageFile   = "data/mnist/testimages.txt";
static const string testLabelFile   = "data/mnist/testlabels.txt";

static MNISTDataSet trainData;
static MNISTDataSet testData;

static const uint32_t inputSize     = 784;
static const uint32_t outputSize    = 10;

static const bool bDoSweep          = true;
static const uint32_t numThreads    = 12;

struct TrainParams
{
    uint32_t numIterations;
    uint32_t batchSize;
    uint32_t pulseLength;
    double learnRate;
    double cullThresh;
    uint32_t minVerts;
    uint32_t maxVerts;
    double minEdge;
    double maxEdge;
    double edgeProb;
};

struct TrainResults
{
    double trainTime;
    double accuracy;
};

static vector<TrainParams> ParamQueue;
mutex paramQueueMtx;

vector<pair<TrainParams, TrainResults>> results;
mutex resultMtx;

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
 * InitParamSweepQueue - Build a table for training parameter sweep. Sweep over
 * training parameters like learning rate, edge probability in random graphs, number
 * of iterations of trianing data. Etc. Worker threads will then pull parameters sets
 * from this queue, train a neural net, and report training time and test set
 * accuaracy.
 */

void InitParamSweepQueue()
{
    assert(ParamQueue.size() == 0);

    vector<uint32_t> numIterSweep   = { 1 , 3, 5 };
    vector<uint32_t> batchSizeSweep = { 10 , 100, 500 };
    vector<double> edgeProbSweep    = { 0.1, 0.3, 0.5, 0.7 };
    vector<double> learnRateSweep   = { 0.001, 0.01, 0.1, 1.0 };

    for (auto& n : numIterSweep)
    {
        for (auto& b : batchSizeSweep)
        {
            for (auto& e : edgeProbSweep)
            {
                for (auto& lr : learnRateSweep)
                {
                    TrainParams params = {};

                    params.numIterations    = n;
                    params.batchSize        = b;
                    params.pulseLength      = 1;
                    params.learnRate        = lr;
                    params.cullThresh       = 1e-8;
                    params.minVerts         = inputSize + outputSize;
                    params.maxVerts         = inputSize + outputSize + 500;
                    params.minEdge          = 1e-6;
                    params.maxEdge          = 100.0;
                    params.edgeProb         = e;

                    ParamQueue.push_back(params);
                }
            }
        }
    }
}

/**
 * MNISTRandThreadFunc - Thread routine for MNISTRand training. Grab a set of training parameters
 * from the parameter sweep queue. Train a neural net with these parameters and report training time
 * and test set accuracy. Keep grabbing parameter sets until queue is emtpy.
 */

void MNISTRandThreadFunc()
{
    paramQueueMtx.lock();

    if (ParamQueue.size() > 0)
    {
        TrainParams params = ParamQueue.back();
        ParamQueue.pop_back();
        
        printf("Thread %d training new net. Remaining jobs = %d\n", this_thread::get_id(),
            ParamQueue.size());
        
        paramQueueMtx.unlock();

        RandomGraph graph(
            params.minVerts, 
            params.maxVerts,
            params.edgeProb,
            params.minEdge,
            params.maxEdge
        );

        vector<Triplet<double>> synapses = graph.GetEdgeTriplets();

        NNCreateParams<double> nnParams;
        nnParams.batchSize  = params.batchSize;
        nnParams.name       = "MNIST Digit Net Random";
        nnParams.numNeurons = graph.numVerts;
        nnParams.synapsesIn = synapses;
        nnParams.learnRate  = params.learnRate;
        nnParams.cullThresh = params.cullThresh;

        NN<double> nn(nnParams);

        vector<uint32_t> inputs;
        vector<uint32_t> outputs;

        PickRandomInputsAndOutputs(graph.numVerts, inputs, outputs);

        vector<vector<pair<uint32_t, double>>> assocPre(params.batchSize);
        vector<vector<pair<uint32_t, double>>> assocPost(params.batchSize);

        for (uint32_t i = 0; i < params.batchSize; i++)
        {
            assocPre[i].resize(inputSize);
            assocPost[i].resize(1);
        }

        // 2. Train

        long long t1 = GetMilliseconds();

        for (uint32_t i = 0; i < params.numIterations; i++)
        {
            for (uint32_t j = 0; j < trainData.numImgs; j += params.batchSize)
            {
                getAssocBatch(trainData, j, params.batchSize, inputs, outputs, assocPre, assocPost);
                nn.applyAssocs(assocPre, assocPost, params.pulseLength);
                nn.computePairings();
                nn.updateSynapses();
            }

            nn.cull();
        }

        long long t2 = GetMilliseconds();

        // 3. Test

        double trainingTime = ((double)(t2 - t1)) / 1000.0;
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

        resultMtx.lock();
        results.push_back({ params, {trainingTime, accuracy} });
    }
    else 
    {
        paramQueueMtx.unlock();
    }
}

/**
 * MNISTRandTest - Train a random graph to learn MNIST hand digits using
 * Hebbian learning. Loop over batches, report training time and NN
 * accuracy.
 */

void MNISTRandTest()
{
    trainData.Init(trainImageFile.c_str(), trainLabelFile.data());
    testData.Init(testImageFile.c_str(), testLabelFile.data());

    srand((uint32_t)time(NULL));

    if (bDoSweep)
    {
        InitParamSweepQueue();

        printf("Beginning parameter sweep. Number of jobs = %d\n\n", ParamQueue.size());

        thread threads[numThreads];

        for (uint32_t i = 0; i < numThreads; i++)
        {
            threads[i] = thread(MNISTRandThreadFunc);
        }

        for (uint32_t i = 0; i < numThreads; i++)
        {
            threads[i].join();
        }
    }
    else
    {
        TrainParams params = {};

        params.numIterations    = 5;
        params.batchSize        = 100;
        params.pulseLength      = 1;
        params.learnRate        = 0.01;
        params.cullThresh       = 1e-8;
        params.minVerts         = inputSize + outputSize;
        params.maxVerts         = inputSize + outputSize + 500;
        params.minEdge          = 1e-6;
        params.maxEdge          = 100.0;
        params.edgeProb         = 0.7;

        ParamQueue.push_back(params);
        MNISTRandThreadFunc();
    }

    for (auto& result : results)
    {
        printf("numIterations = %d\n", result.first.numIterations);
        printf("batchSize     = %d\n", result.first.batchSize);
        printf("pulseLength   = %d\n", result.first.pulseLength);
        printf("learnRate     = %g\n", result.first.learnRate);
        printf("cullThresh    = %g\n", result.first.cullThresh);
        printf("minVerts      = %d\n", result.first.minVerts);
        printf("maxVerts      = %d\n", result.first.maxVerts);
        printf("minEdge       = %g\n", result.first.minEdge);
        printf("maxEdge       = %g\n", result.first.maxEdge);
        printf("edgeProb      = %g\n", result.first.edgeProb);
        printf("train time    = %g\n", result.second.trainTime);
        printf("accuarcy      = %g\n\n", result.second.accuracy);
    }
}