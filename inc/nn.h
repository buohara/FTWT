#pragma once

#include "matrix.h"

static const double learnRate   = 0.01;
static const double cullThresh  = 1e-8;

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

    NN() : numNeurons(0) {}

    /**
     * NN::NN - FTWT NN constructor.
     *
     * @param name          Name of this net. Used when printing net info.
     * @param numNeurons    Number of neurons in this net.
     * @param synapsesIn    A list of initial synapse weights of form (neuronTo, neuronFrom, weight).
     * @param batchSize     Batch size to use when computing neuron pairings. 
     */
    
    NN(string name, uint32_t numNeurons, vector<Triplet<T>> &synapsesIn, uint32_t batchSize) : numNeurons(numNeurons),
        batchSize(batchSize)
    {
        activationsPre.resize(batchSize);
        activationsPost.resize(batchSize);

        for (uint32_t i = 0; i < batchSize; i++)
        {
            activationsPre[i].resize(numNeurons);
            activationsPost[i].resize(numNeurons);
        }

        TripletMat<T> synapsesTrip(numNeurons, numNeurons, name);
        for (auto& synapse : synapsesIn) synapsesTrip.insert(synapse);
        synapses = synapsesTrip.toCSC();
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
        pairings = synapses;
        fill(pairings.vals.begin(), pairings.vals.end(), 0);

        for (uint32_t r = 0; r < synapses.n; r++)
        {
            for (uint32_t i = synapses.offsets[r]; i < synapses.offsets[r + 1]; i++)
            {
                for (uint32_t j = 0; j < batchSize; j++)
                {
                    pairings.vals[i] += activationsPost[j][r] * activationsPre[j][synapses.colIdcs[i]];
                }
            }
        }

        for (uint32_t i = 0; i < pairings.vals.size(); i++)
        {
            pairings.vals[i] /= (double)batchSize;
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