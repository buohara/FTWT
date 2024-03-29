#pragma once

#include "matrix.h"

template<class T>
struct NNCreateParams
{
    string name;
    uint32_t numNeurons;
    uint32_t batchSize;
    double learnRate;
    double cullThresh;
    vector<Triplet<T>> synapsesIn;
};

template<class T>
struct NN
{
    uint32_t numNeurons;
    uint32_t batchSize;
    CSCMat<T> synapses;
    vector<vector<T>> activationsPre;
    vector<vector<T>> activationsPost;
    CSCMat<T> pairings;
    double learnRate;
    double cullThresh;

    /**
     * NN::NN - FTWT NN default constructor.
     */

    NN() : numNeurons(0), batchSize(1), learnRate(1.0), cullThresh(0.0) {}

    /**
     * NN::NN - FTWT NN constructor.
     *
     * @param params FTWT NN creation param struct.
     */

    NN(NNCreateParams<T> &params) : numNeurons(params.numNeurons),
        batchSize(params.batchSize),
        learnRate(params.learnRate),
        cullThresh(params.cullThresh)
    {
        activationsPre.resize(batchSize);
        activationsPost.resize(batchSize);

        for (uint32_t i = 0; i < batchSize; i++)
        {
            activationsPre[i].resize(numNeurons);
            activationsPost[i].resize(numNeurons);
        }

        TripletMat<T> synapsesTrip(numNeurons, numNeurons, params.name);
        for (auto& synapse : params.synapsesIn) synapsesTrip.insert(synapse);
        synapses = synapsesTrip.toCSC();
        pairings = synapses;
    }

    /**
     * NN::applyAssocs - Apply desired learning associations and record
     * neuron activations. These activations are used to compute neuron pairings
     * and strengthen/weaken appropriate synapse connections. Neurons pair if a
     * postsynaptic activation follows a presynaptic activation.
     *
     * @param assocPre  Presynaptic neuron activations.
     * @param assocPost Postsynaptic neuron activations.
     * @param numPulses Number of iterations to propagate presynaptic activations through network.
     */
    
    void applyAssocs(
        vector<vector<pair<uint32_t, T>>>& assocPre,
        vector<vector<pair<uint32_t, T>>>& assocPost,
        uint32_t numPulses)
    {
        for (uint32_t i = 0; i < batchSize; i++)
        {
            fill(activationsPre[i].begin(), activationsPre[i].end(), 0.0);
            fill(activationsPost[i].begin(), activationsPost[i].end(), 0.0);
            
            for (auto& assoc : assocPre[i]) activationsPre[i][assoc.first] = assoc.second;
            activationsPost[i][assocPost[i][0].first] = assocPost[i][0].second;
        }
    }

    /**
     * NN::applyInput - Compute response of NN to a given input. Used when testing NN accuracy.
     *
     * @param input Input activations to compute response for.
     *
     * @return Response of NN to input.
     */
    
    vector<T> applyInput(vector<T>& input)
    {
        vector<T> res = synapses * input;
        return res;
    }

    /**
     * NN::computePairings - After applying batch of associations and recording activations, 
     * record average pairing of neurons connected by synapses. Pairing = 
     * preSynapseNeuronActiviation * postSynapseNeuronActiviation. 
     */
    
    void computePairings()
    {
        double* pPairings = &pairings.vals[0];
        memset(pPairings, 0, pairings.vals.size() * sizeof(double));

        for (uint32_t bat = 0; bat < batchSize; bat++)
        {
            double* pBatActivationsPre  = &activationsPre[bat][0];
            double* pBatActivationsPost = &activationsPost[bat][0];
            uint32_t *pOffsets          = &synapses.offsets[0];
            uint32_t *pIdcs             = &synapses.colIdcs[0];

            for (uint32_t row = 0; row < synapses.n; row++)
            {
                for (uint32_t syn = pOffsets[row]; syn < pOffsets[row + 1]; syn++)
                {
                    pPairings[syn] += pBatActivationsPost[row] * pBatActivationsPre[pIdcs[syn]];
                }
            }
        }

        double batchSizeInv = 1.0 / (double)batchSize;

        for (uint32_t i = 0; i < pairings.vals.size(); i++)
        {
            pPairings[i] *= batchSizeInv;
        }
    }

    /**
     * NN::cull - Between synapse updates, remove any weak synapses.
     */
    
    void cull()
    {
        TripletMat<T> triplet;
        triplet.m = synapses.m;
        triplet.n = synapses.n;

        uint32_t r = 0;

        for (uint32_t i = 0; i < synapses.vals.size(); i++)
        {
            while (i == synapses.offsets[r + 1]) r++;
            if (abs(synapses.vals[i]) < cullThresh) continue;
            else
            {
                triplet.insert({ r, synapses.colIdcs[i], synapses.vals[i] });
            }
        }

        synapses = triplet.toCSC();
        pairings = synapses;
    }

    /**
     * NN::updateSynapses - After associations have been applied and average neuron pairings
     * computed, adjust sypanse strength in proportion to product of pre/post synapse activation
     * strength. All synapse weights feeding into a given neuron are normalized to 1 to avoid
     * runaway sypanse weights.
     */
    
    void updateSynapses()
    {
        vector<T> synapseTotal(synapses.n, 0);
        uint32_t r = 0;

        for (uint32_t i = 0; i < pairings.vals.size(); i++)
        {
            while (i == synapses.offsets[r + 1]) r++;
            
            T w = synapses.vals[i] + learnRate * pairings.vals[i];
            
            synapseTotal[r] += w * w;
            synapses.vals[i] = w;
        }

        for (uint32_t i = 0; i < synapseTotal.size(); i++)
        {
            synapseTotal[i] = sqrt(synapseTotal[i]);
        }

        r = 0;

        for (uint32_t i = 0; i < pairings.vals.size(); i++)
        {
            while (i == synapses.offsets[r + 1]) r++;
            synapses.vals[i] /= synapseTotal[r];
        }
    }

    /**
     * NN::print - Print synapse weights for this NN.
     *
     * @param bAll Whether to print full synapse matrix.
     */
    
    void print(bool bAll = false)
    {
        synapses.print(bAll);
    }
};