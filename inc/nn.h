#pragma once

#include "matrix.h"

static const double maxSynapse  = 10.0;
static const double learnRate   = 5.0;

template<class T>
struct NN
{
    uint32_t numNeurons;
    CSCMat<double> synapses;
    vector<double> activations;
    CSCMat<double> pairings;

    NN() : numNeurons(0) {}

    NN(string name, uint32_t numNeurons, vector<Triplet<T>> &synapsesIn) : numNeurons(numNeurons)
    {
        activations.resize(numNeurons);
        TripletMat<double> synapsesTrip(numNeurons, numNeurons, name);

        for (auto& synapse : synapsesIn) synapsesTrip.insert(synapse);

        synapses = synapsesTrip.toCSC();
        pairings = synapses;
        fill(pairings.vals.begin(), pairings.vals.end(), 0);
    }

    void applyAssocs(vector<pair<uint32_t, T>>& inputs, uint32_t numPulses)
    {
        fill(activations.begin(), activations.end(), 0.0);
        for (auto& input : inputs) activations[input.first] = input.second;

        for (uint32_t i = 0; i < numPulses; i++)
        {
            activations = synapses * activations;
            for (auto& input : inputs) activations[input.first] = input.second;

            for (uint32_t i = 0; i < numNeurons; i++)
            {
                //double x = activations[i];
                //activations[i] = exp(x) / (exp(x) + 1);
            }
        }
    }

    vector<T> applyInput(vector<T>& input)
    {
        vector<T> res = synapses * input;
        for (uint32_t i = 0; i < res.size(); i++)
        {
            //res[i] = exp(res[i]) / (1 + exp(res[i]));
        }

        return res;
    }

    void computePairings()
    {
        fill(pairings.vals.begin(), pairings.vals.end(), 0);

        for (uint32_t r = 0; r < synapses.n; r++)
        {
            for (uint32_t i = synapses.offsets[r]; i < synapses.offsets[r + 1]; i++)
            {
                pairings.vals[i] = activations[r] * activations[synapses.colIdcs[i]];
            }
        }
    }

    void updateSynapses()
    {
        double synapseTotal = 0.0;

        for (uint32_t i = 0; i < pairings.vals.size(); i++)
        {
            double w = synapses.vals[i] + learnRate * pairings.vals[i];
            synapseTotal += w * w;
            synapses.vals[i] = w;
        }

        synapseTotal = sqrt(synapseTotal);

        for (uint32_t i = 0; i < pairings.vals.size(); i++)
        {
            synapses.vals[i] /= synapseTotal;
        }
    }

    vector<T> getActivations()
    {
        return activations;
    }

    void print(bool bAll = false)
    {
        synapses.print(bAll);
    }
};