#include "tests.h"

void SimpleCrossTest()
{
    const uint32_t numNeurons   = 4;
    const uint32_t numIters     = 10;
    const uint32_t numPulses    = 1;
    const uint32_t batchSize    = 1;
    const double learnRate      = 0.01;
    const double cullThresh     = 1e-8;

    vector<vector<pair<uint32_t, double>>> assocPre1    = { { { 0, 1.0 } } };
    vector<vector<pair<uint32_t, double>>> assocPost1   = { { { 3, 1.0 } } };

    vector<vector<pair<uint32_t, double>>> assocPre2    = { { { 1, 1.0 } } };
    vector<vector<pair<uint32_t, double>>> assocPost2   = { { { 2, 1.0 } } };

    NNCreateParams<double> params;
    params.batchSize    = batchSize;
    params.name         = "Simple 2x2 Net";
    params.numNeurons   = numNeurons;
    params.learnRate    = learnRate;
    params.cullThresh   = cullThresh;
    params.synapsesIn =
    {
        { 0, 2, 52.0 },
        { 2, 0, 45.0 },
        { 0, 3, 57.0 },
        { 3, 0, 47.0 },
        { 1, 2, 45.0 },
        { 2, 1, 58.0 },
        { 1, 3, 49.0 },
        { 3, 1, 56.0 }
    };

    NN<double> network(params);

    for (uint32_t i = 0; i < numIters; i++)
    {
        network.applyAssocs(assocPre1, assocPost1, numPulses);
        network.computePairings();
        network.updateSynapses();

        network.applyAssocs(assocPre2, assocPost2, numPulses);
        network.computePairings();
        network.updateSynapses();
    }

    vector<double> test1 = { 1.0, 0.0, 0.0, 0.0 };
    vector<double> test2 = { 0.0, 1.0, 0.0, 0.0 };

    vector<double> out1 = network.applyInput(test1);
    vector<double> out2 = network.applyInput(test2);

    network.print();

    print(out1);
    print(out2);

    __debugbreak();
}