#include "neuralnetworkgpu.h"
#include "nnkernels.cuh"

cublasHandle_t hCublas;

/**
 * [NNLayerGPU::Init description]
 * @param inSize  [description]
 * @param outSize [description]
 * @param type    [description]
 */

void NNLayerGPU::Init(
    uint32_t inSize,
    uint32_t outSize,
    NNLayerType type
)
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

    curandGenerator_t prng;
    curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_XORWOW);
    curandSetPseudoRandomGeneratorSeed(prng, 0);

    cudaMalloc(&weights, inputSize * outputSize * sizeof(double));
    curandGenerateNormalDouble(prng, weights, inputSize * outputSize, 0.0, 0.1);

    cudaMalloc(&biases, outputSize * sizeof(double));
    curandGenerateNormalDouble(prng, biases, outputSize, 0.0, 0.1);
}

/**
 * [NNLayerGPU::EvaluateFull description]
 * @param in    [description]
 * @param out   [description]
 * @param aOut  [description]
 * @param spOut [description]
 */

void NNLayerGPU::EvaluateFull(
    double *in,
    double *out,
    double *aOut,
    double *spOut
)
{
    double alpha = 1.0;
    double beta = 1.0;
    
    cudaMemcpy(out, biases, outputSize * sizeof(double), cudaMemcpyDeviceToDevice);

    cublasDgemv(
        hCublas,
        CUBLAS_OP_N,
        outputSize,
        inputSize,
        &alpha,
        weights,
        outputSize,
        in,
        1,
        &beta,
        out,
        1
    );

    sigmoids(out, aOut, spOut, outputSize);

    return;
}

/**
 * [NNFullGPU::Init description]
 * @param params [description]
 */

void NNFullGPU::Init(NNSettings &params)
{
    hiddenLayerSize = params.hiddenLayerSize;
    numLayers       = params.numLayers;
    inputSize       = params.inputSize;
    outputSize      = params.outputSize;

    layers.resize(numLayers);

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
 * [NNFullGPU::ZeroGradient description]
 */

void NNFullGPU::ZeroGradient()
{
    for (uint32_t l = 0; l < numLayers; l++)
    {
        cudaMemset(
            scratch.nablaBs[l],
            0,
            layers[l].outputSize * sizeof(double)
        );

        cudaMemset(
            scratch.nablaWs[l],
            0,
            layers[l].inputSize * layers[l].outputSize * sizeof(double)
        );
    }
}

/**
 * [NNFullGPU::BackProp description]
 * @param in     [description]
 * @param actual [description]
 */

void NNFullGPU::BackProp(double *imgIn, double *actual)
{
    cudaMemcpy(
        scratch.activations[0],
        imgIn,
        inputSize * sizeof(double),
        cudaMemcpyDeviceToDevice
    );
    
    cudaMemcpy(
        scratch.actual,
        actual,
        outputSize * sizeof(double),
        cudaMemcpyDeviceToDevice
    );

    for (uint32_t l = 1; l < numLayers; l++)
    {
        layers[l].EvaluateFull(
            scratch.activations[l - 1],
            scratch.zVecs[l],
            scratch.activations[l],
            scratch.sps[l]
        );
    }

    double alpha = -1.0;
    double beta = 0.0;

    cudaMemcpy(
        scratch.deltas[numLayers - 1],
        scratch.activations[numLayers - 1],
        outputSize * sizeof(double),
        cudaMemcpyDeviceToDevice
    );
    
    cublasDaxpy(
        hCublas,
        outputSize,
        &alpha,
        scratch.actual,
        1,
        scratch.deltas[numLayers - 1],
        1
    );

    had(
        scratch.deltas[numLayers - 1],
        scratch.sps[numLayers - 1], 
        outputSize
    );

    alpha = 1.0;

    for (uint32_t l = numLayers - 1; l-- > 1;)
    {
        cublasDgemv(
            hCublas,
            CUBLAS_OP_T,
            layers[l + 1].outputSize,
            layers[l + 1].inputSize,
            &alpha,
            layers[l + 1].weights,
            layers[l + 1].outputSize,
            scratch.deltas[l + 1],
            1,
            &beta,
            scratch.deltas[l],
            1
        );

        had(scratch.deltas[l], scratch.sps[l], layers[l].outputSize);
    }

    for (uint32_t l = numLayers; l-- > 1;)
    {
        cublasDaxpy(
            hCublas,
            layers[l].outputSize,
            &alpha,
            scratch.deltas[l],
            1,
            scratch.nablaBs[l],
            1
        );

        cublasDger(
            hCublas,
            layers[l].outputSize,
            layers[l].inputSize,
            &alpha,
            scratch.deltas[l],
            1,
            scratch.activations[l - 1],
            1,
            scratch.nablaWs[l],
            layers[l].outputSize
        );
    }

    return;
}

/**
 * [NNFullGPU::InitTrainingScratch description]
 */

void NNFullGPU::InitTrainingScratch()
{
    scratch.activations.resize(numLayers);
    scratch.zVecs.resize(numLayers);
    scratch.sps.resize(numLayers);
    scratch.deltas.resize(numLayers);
    scratch.nablaBs.resize(numLayers);
    scratch.nablaWs.resize(numLayers);

    for (uint32_t l = 0; l < numLayers; l++)
    {
        size_t cols = (size_t)layers[l].inputSize;
        size_t rows = (size_t)layers[l].outputSize;

        cudaMalloc(&scratch.activations[l], rows * sizeof(double));
        cudaMalloc(&scratch.zVecs[l], rows * sizeof(double));
        cudaMalloc(&scratch.sps[l], rows * sizeof(double));
        cudaMalloc(&scratch.deltas[l], rows * sizeof(double));
        cudaMalloc(&scratch.nablaBs[l], rows * sizeof(double));
        cudaMalloc(&scratch.nablaWs[l], rows * cols * sizeof(double));
    }

    cudaMalloc(&scratch.actual, outputSize * sizeof(double));
}

/**
 * [NNFullGPU::SGDStepMiniBatch description]
 * @param ds            [description]
 * @param idcs          [description]
 * @param miniBatchSize [description]
 * @param learningRate  [description]
 */

void NNFullGPU::SGDStepMiniBatch(
    MNISTDataSet &ds,
    vector<uint32_t> &idcs,
    uint32_t miniBatchSize,
    double learningRate
)
{
    VectorXd actual(outputSize);
    double stepSize = -learningRate / ((double)miniBatchSize);
    double beta = 1.0;

    for (uint32_t i = 0; i < miniBatchSize; i++)
    {
        actual.setZero();
        uint32_t actualHot = ds.labels[idcs[i]];
        actual[actualHot] = 1.0f;

        BackProp(ds.cudaImgs[i], ds.cudaLabels[i]);
    }

    for (uint32_t l = 1; l < numLayers; l++)
    {
        cublasDaxpy(
            hCublas,
            layers[l].outputSize,
            &stepSize,
            scratch.nablaBs[l],
            1,
            layers[l].biases,
            1
        );

        cublasDgeam(
            hCublas,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            layers[l].outputSize,
            layers[l].inputSize,
            &stepSize,
            scratch.nablaWs[l],
            layers[l].outputSize,
            &beta,
            layers[l].weights,
            layers[l].outputSize,
            layers[l].weights,
            layers[l].outputSize
        );
    }
}

/**
 * [NNFullGPU::Train description]
 * @param ds          [description]
 * @param learnParams [description]
 */

void NNFullGPU::Train(MNISTDataSet &ds, NNSettings &learnParams)
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

            if (j > 200)
            {
                exit(0);
            }

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
 * [NNFullGPU::Test description]
 * @param testSet [description]
 */

void NNFullGPU::Test(MNISTDataSet &testSet)
{
    cout << "Testing neural net...\n" << endl;

    VectorXd out;
    out.resize(outputSize);

    uint32_t matchCnt = 0;

    for (uint32_t i = 0; i < testSet.numImgs; i++)
    {
        cudaMemcpy(
            scratch.activations[0],
            testSet.data[i].data(),
            inputSize * sizeof(double),
            cudaMemcpyHostToDevice
        );

        for (uint32_t l = 1; l < numLayers; l++)
        {
            layers[l].EvaluateFull(
                scratch.activations[l - 1],
                scratch.zVecs[l],
                scratch.activations[l],
                scratch.sps[l]
            );
        }

        cudaMemcpy(
            out.data(),
            scratch.activations[numLayers - 1],
            outputSize * sizeof(double),
            cudaMemcpyDeviceToHost
        );

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
 * [CheckCUBLAS description]
 */

void CheckCUBLAS()
{
    MatrixXd A(3, 4);

    A << 1, 2, 3, 4,
        2, 3, 4, 5,
        3, 4, 5, 6;

    VectorXd b1(4);
    VectorXd c1(3);

    VectorXd b2(3);
    VectorXd c2(4);

    b1 << 1, 2, 3, 4;
    c1 << 2, 3, 4;

    b2 << 5, 6, 7;
    c2 << 6, 7, 8, 9;

    cout << A << "\n\n * \n\n" << b1 << "\n\n + \n\n" << c1 << "\n\n = \n\n" << endl;
    c1 = A * b1 + c1;
    cout << c1 << endl << endl;

    cout << A.transpose() << "\n\n * \n\n" << b2 << "\n\n + \n\n" << c2 << "\n\n = \n\n" << endl;
    c2 = A.transpose() * b2 + c2;
    cout << c2 << endl << endl;

    double *devA;
    double *devB1;
    double *devC1;
    double *devB2;
    double *devC2;

    c1 << 2, 3, 4;
    c2 << 6, 7, 8, 9;

    cudaMalloc(&devA, 3 * 4 * sizeof(double));
    cudaMalloc(&devB1, 4 * sizeof(double));
    cudaMalloc(&devC1, 3 * sizeof(double));
    cudaMalloc(&devB2, 3 * sizeof(double));
    cudaMalloc(&devC2, 4 * sizeof(double));

    cublasSetMatrix(3, 4, sizeof(double), A.data(), 3, devA, 3);
    cublasSetVector(4, sizeof(double), b1.data(), 1, devB1, 1);
    cublasSetVector(3, sizeof(double), b2.data(), 1, devB2, 1);
    cublasSetVector(3, sizeof(double), c1.data(), 1, devC1, 1);
    cublasSetVector(4, sizeof(double), c2.data(), 1, devC2, 1);

    double c1Dev[3];
    double c2Dev[4];

    double alpha = 1.0;
    double beta = 1.0;

    cublasDgemv(
        hCublas,
        CUBLAS_OP_N,
        3,
        4,
        &alpha,
        devA,
        3,
        devB1,
        1,
        &beta,
        devC1,
        1
    );

    cublasDgemv(
        hCublas,
        CUBLAS_OP_T,
        3,
        4,
        &alpha,
        devA,
        3,
        devB2,
        1,
        &beta,
        devC2,
        1
    );

    cudaDeviceSynchronize();

    cublasGetVector(3, sizeof(double), devC1, 1, c1Dev, 1);
    cublasGetVector(4, sizeof(double), devC2, 1, c2Dev, 1);

    cout << "\n\ncuda c1 = " << c1Dev[0] << " " << c1Dev[1] << " " << c1Dev[2] << endl;
    cout << "\n\ncuda c2 = " << c2Dev[0] << " " << c2Dev[1] << " " << c2Dev[2] << " " << c2Dev[3] << endl;

    __debugbreak();
}

/**
 * [NNFullGPU::main description]
 * @param settings    [description]
 * @param trainingSet [description]
 * @param testSet     [description]
 */

void NNFullGPU::main(
    NNSettings &settings,
    MNISTDataSet &trainingSet,
    MNISTDataSet &testSet
)
{

    trainingSet.InitCUDAImages();
    cublasCreate_v2(&hCublas);

    NNFullGPU NN;
    NN.Init(settings);
    NN.Train(trainingSet, settings);
    NN.Test(testSet);
    return;
}