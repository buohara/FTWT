#include "neuralnetworkcpu.h"

/**
 * [NNFullCPU::ZeroGradient description]
 */

void NNFullCPU::ZeroGradient()
{
    for (auto &nablaB : scratch.nablaBs)
    {
        nablaB.setZero();
    }

    for (auto &nablaW : scratch.nablaWs)
    {
        nablaW.setZero();
    }
}


/**
 * [NNLayerCPU::Init description]
 * @param inSize  [description]
 * @param outSize [description]
 * @param bInput  [description]
 * @param bOutput [description]
 */

void NNLayerCPU::Init(uint32_t inSize, uint32_t outSize, NNLayerType type)
{
    inputSize       = inSize;
    outputSize      = outSize;
    layerType       = type;

    // Input layers are placeholders that just pass input activations to 
    // the next layer. No need to allocate weights and biases.

    if (layerType == INPUT_LAYER)
    {
        return;
    }

    weights = 0.05 * MatrixXd::Random(outSize, inSize);
    biases = 0.05 * VectorXd::Random(outSize);
}


/**
 * [NNLayerCPU::EvaluateFull description]
 * @param in    [description]
 * @param out   [description]
 * @param aOut  [description]
 * @param spOut [description]
 */

void NNLayerCPU::EvaluateFull(VectorXd &in, VectorXd &out, VectorXd &aOut, VectorXd &spOut)
{
    VectorXd ones;
    ones.resize(outputSize);
    ones.setOnes();
    out.setZero();

    out     = weights * in + biases;
    aOut    = -out;
    aOut    = ones.array() + aOut.array().exp();
    aOut    = aOut.cwiseInverse();
    spOut   = aOut.cwiseProduct(ones - aOut);
}

/**
 * [NNLayerCPU::Evaluate description]
 * @param in  [description]
 * @param out [description]
 */

void NNLayerCPU::Evaluate(VectorXd &in, VectorXd &out)
{
    VectorXd ones;
    ones.resize(outputSize);
    ones.setOnes();
    out.setZero();

    out     = weights * in + biases;
    out     = -out;
    out     = ones.array() + out.array().exp();
    out     = out.cwiseInverse();
}

/**
 * [NNLayerCPU::operator<< description]
 * @param os  [description]
 * @param m [description]
 */

ostream& operator<<(ostream &os, NNLayerCPU const &m)
{
    os << "(" << m.inputSize << ", " << m.outputSize << ")\n\n" <<
        "b:\n" << m.biases << "\n\n" << "w:\n" << m.weights << endl;

    return os;
}

/**
 * [NNFullCPU::main description]
 * @param settings    [description]
 * @param trainingSet [description]
 * @param testSet     [description]
 */

void NNFullCPU::main(
    NNSettings &settings,
    MNISTDataSet &trainingSet,
    MNISTDataSet &testSet
)
{
    NNFullCPU NN;
    NN.Init(settings);
    NN.Train(trainingSet, settings);
    NN.Test(testSet);
    return;
}

/**
 * [NNFullCPU::Init description]
 * @param params [description]
 */

void NNFullCPU::Init(NNSettings &params)
{
    hiddenLayerSize = params.hiddenLayerSize;
    numLayers       = params.numLayers;
    inputSize       = params.inputSize;
    outputSize      = params.outputSize;

    layers.resize(numLayers);

    // Input and output layers.

    layers[0].Init(
        inputSize,
        inputSize,
        INPUT_LAYER
    );

    layers[numLayers - 1].Init(
        numLayers > 2 ? hiddenLayerSize : inputSize,
        outputSize,
        OUTPUT_LAYER
    );

    // Hidden layers.

    uint32_t layerIdx = 0;

    for (auto &layer : layers)
    {
        if (layerIdx == 0 || layerIdx == numLayers - 1)
        {
            layerIdx++;
            continue;
        }
      
        layer.Init(
            layerIdx == 1 ? inputSize : hiddenLayerSize,
            hiddenLayerSize,
            HIDDEN_LAYER
        );

        layerIdx++;
    }
}

/**
 * [NNFullCPU::Evaluate description]
 * @param in  [description]
 * @param out [description]
 */

void NNFullCPU::Evaluate(VectorXd &in, VectorXd &out)
{
    // If only an input and output layer, feed input to the output layer
    // and return.

    if (numLayers == 2)
    {
        layers[1].Evaluate(in, out);
        return;
    }

    // Otherwise, evaluate hidden layers and output layer.

    VectorXd h1(hiddenLayerSize);
    VectorXd h2(hiddenLayerSize);
    VectorXd tmp(hiddenLayerSize);

    for (uint32_t l = 1; l < numLayers - 1; l++)
    {
        if (l == 1)
        {
            layers[l].Evaluate(in, h1);
        }
        else
        {
            layers[l].Evaluate(h1, h2);
            
            tmp = h1;
            h1  = h2;
            h2  = tmp;
        }
    }

    layers[numLayers - 1].Evaluate(h1, out);

    return;
}

/**
 * [NNFullCPU::BackProp description]
 * @param in     [description]
 * @param actual [description]
 */

void NNFullCPU::BackProp(VectorXd &in, VectorXd &actual)
{
    // Feedforward pass. Set input activations to input data, then
    // evaluate remaining layers.

    scratch.activations[0] = in;

    for (uint32_t l = 1; l < numLayers; l++)
    {
        layers[l].EvaluateFull(
            scratch.activations[l - 1],
            scratch.zVecs[l],
            scratch.activations[l],
            scratch.sps[l]
        );
    }

    // Compute output layer error: (A^L - y) * sig'(Z^L).

    scratch.deltas[numLayers - 1] = scratch.activations[numLayers - 1] - actual;
    scratch.deltas[numLayers - 1].cwiseProduct(scratch.sps[numLayers - 1]);

    // Backpropagate output error.

    for (uint32_t l = numLayers - 1; l-- > 1;)
    {
        // Compute current layer's error: 
        // delta^L = (W^(L+1))^t * delta^(L+1) * sig'(z^L).

        MatrixXd weightsT = layers[l + 1].weights.transpose();
        scratch.deltas[l].setZero();
        scratch.deltas[l] = weightsT * scratch.deltas[l + 1];
        scratch.deltas[l].cwiseProduct(scratch.sps[l]);
    }

    // Add each layer's weights and bias gradients for current sample to
    // the minibatch gradient estimate.

    for (uint32_t l = numLayers; l-- > 1;)
    {
        scratch.nablaBs[l] += scratch.deltas[l];
        scratch.nablaWs[l] += scratch.deltas[l] * scratch.activations[l - 1].transpose();
    }

    return;
}

/**
 * [NNFullCPU::SGDStepMiniBatch description]
 * @param ds            [description]
 * @param idcs          [description]
 * @param miniBatchSize [description]
 * @param learningRate  [description]
 */

void NNFullCPU::SGDStepMiniBatch(
    MNISTDataSet &ds,
    vector<uint32_t> &idcs,
    uint32_t miniBatchSize,
    double learningRate)
{
    VectorXd actual(outputSize);
    double stepSize = learningRate / ((double)miniBatchSize);

    for (uint32_t i = 0; i < miniBatchSize; i++)
    {
        actual.setZero();
        uint32_t actualHot = ds.labels[idcs[i]];
        actual[actualHot] = 1.0f;

        BackProp(ds.data[idcs[i]], actual);
    }

    for (uint32_t l = 1; l < numLayers; l++)
    {
        layers[l].biases -= stepSize * scratch.nablaBs[l];
        layers[l].weights -= stepSize * scratch.nablaWs[l];
    }
}

/**
 * [NNFullCPU::InitTrainingScratch description]
 */

void NNFullCPU::InitTrainingScratch()
{
    scratch.activations.resize(numLayers);
    scratch.zVecs.resize(numLayers);
    scratch.sps.resize(numLayers);
    scratch.deltas.resize(numLayers);
    scratch.nablaBs.resize(numLayers);
    scratch.nablaWs.resize(numLayers);

    for (uint32_t l = 0; l < numLayers; l++)
    {
        uint32_t cols = layers[l].inputSize;
        uint32_t rows = layers[l].outputSize;

        scratch.activations[l].resize(rows);
        scratch.zVecs[l].resize(rows);
        scratch.sps[l].resize(rows);
        scratch.deltas[l].resize(rows);
        scratch.nablaBs[l].resize(rows);
        scratch.nablaWs[l].resize(rows, cols);
    }
}

/**
 * [NNFullCPU::Train description]
 * @param ds          [description]
 * @param learnParams [description]
 */

void NNFullCPU::Train(MNISTDataSet &ds, NNSettings &learnParams)
{
    cout << "Training neural net...\n" << endl;

    InitTrainingScratch();
    vector<uint32_t> inputIdcs(ds.numImgs);

    for (uint32_t i = 0; i < ds.numImgs; i++)
    {
        inputIdcs[i] = i;
    }

    for (uint32_t i = 0; i < learnParams.numEpochs; i++)
    {
        cout << "Running training epoch " << i + 1 << "..." << endl;
        random_shuffle(inputIdcs.begin(), inputIdcs.end());

        for (uint32_t j = 0; j < ds.numImgs; j += learnParams.miniBatchSize)
        {
            ZeroGradient();

            vector<uint32_t> miniBatchIdcs(inputIdcs.begin() + j, 
                inputIdcs.begin() + j + learnParams.miniBatchSize);

            SGDStepMiniBatch(
                ds,
                miniBatchIdcs,
                learnParams.miniBatchSize,
                learnParams.learningRate
            );
        }
    }
}

/**
 * [NNFullCPU::Test description]
 * @param testSet [description]
 */

void NNFullCPU::Test(MNISTDataSet &testSet)
{
    cout << "Testing neural net...\n" << endl;

    VectorXd out;
    out.resize(outputSize);

    uint32_t matchCnt = 0;

    for (uint32_t i = 0; i < testSet.numImgs; i++)
    {
        out.setOnes();
        Evaluate(testSet.data[i], out);

        double max          = 0.0f;
        double max2         = 0.0f;
        uint32_t maxNum     = 0;
        uint32_t maxNum2    = 0;

        for (uint32_t j = 0; j < outputSize; j++)
        {
            if (out[j] > max)
            {
                max = out[j];
                maxNum = j;
            }
        }

        for (uint32_t j = 0; j < outputSize; j++)
        {
            if (j == maxNum)
            {
                continue;
            }
            if (out[j] > max2)
            {
                max2 = out[j];
                maxNum2 = j;
            }
        }

        if (maxNum == testSet.labels[i])
        {
            matchCnt++;
        }
    }

    double accuracy = 100.0f * ((double)matchCnt / (double)testSet.numImgs);
    cout << "NN test set accuracy: " << accuracy << "\n" << endl;
}

/**
 * [NNFullCPU::operator<< description]
 * @param os  [description]
 * @param m [description]
 */

ostream& operator<<(ostream &os, NNFullCPU const &m)
{
    uint32_t l = 0;

    for (auto &layer : m.layers)
    {
        os << "layer " << l << ":\n" << endl;
        os << layer << endl;
        l++;
    }

    return os;
}