#pragma once

#include "matrix.h"

static const double maxSynapse  = 10.0;
static const double learnRate   = 0.1;
static const double cullThresh  = 1e-8;

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

    vector<T> applyInput(vector<T>& input)
    {
        vector<T> res = synapses * input;
        return res;
    }

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

    vector<T> getactivationsPre()
    {
        return activationsPre;
    }

    void print(bool bAll = false)
    {
        synapses.print(bAll);
    }
};