#pragma once

#include "ftwt.h"
#include "matrix.h"

struct RandomGraph
{
    vector<vector<pair<uint32_t, double>>> adjacencies;

    uint32_t minVerts;
    uint32_t maxVerts;
    uint32_t numVerts;

    double edgeProb;
    double edgeMin;
    double edgeMax;

    /**
     * RandomGraph::RandomGraph - Default constructor. Initialize a random graph with no verts.
     */

    RandomGraph() : minVerts(0), maxVerts(0), edgeProb(0.0), edgeMin(0.0), edgeMax(0.0), numVerts(0) {}


    /**
     * RandomGraph::RandomGraph - Random graph constructor.
     *
     * @param minVerts  Minimum number of verts required for this graph.
     * @param maxVerts  Maximum number of verts for this graph.
     * @param edgeProb  Probability that an edge will be added between two verts.
     * @param edgeMin   Minimum edge weight.
     * @param edgeMax   Maximum edge weight.
     */

    RandomGraph(uint32_t minVerts, uint32_t maxVerts, double edgeProb, double edgeMin, double edgeMax) : 
        minVerts(minVerts), maxVerts(maxVerts), edgeProb(edgeProb), edgeMin(edgeMin), edgeMax(edgeMax) 
    {
        assert(edgeProb >= 0 && edgeProb <= 1.0);
        assert(edgeMax >= edgeMin);
        assert(maxVerts >= minVerts);

        GenerateRandomDirectedGraph();
    }

    /**
     * RandomGraph::GetEdgeTriplets - Return a list of graph edge in triplet form.
     *
     * @return List of graph edges in triplet form.
     */

    vector<Triplet<double>> GetEdgeTriplets()
    {
        vector<Triplet<double>> edges;

        for (uint32_t i = 0; i < adjacencies.size(); i++)
        {
            for (auto &edge : adjacencies[i])
            {
                edges.push_back({ i, edge.first, edge.second });
            }
        }

        return edges;
    }

private:

    /**
     * RandomGraph::GenerateRandomDirectedGraph - Randomly generate a directed graph. 
     * A random number of vertices between min / max values is generated. For each pair of vertices,
     * generate a random value between 0.0 and 1.0. If the value is above edge creation probability,
     * add an edge with random weight between min and max values.
     */

    void GenerateRandomDirectedGraph()
    {
        numVerts = rand() % (maxVerts - minVerts) + minVerts;
        adjacencies.resize(numVerts);

        for (uint32_t i = 0; i < numVerts; i++)
        {
            for (uint32_t j = 0; j < numVerts; j++)
            {
                if (i == j) continue;

                double r = (double)rand() / (double)RAND_MAX;

                if (r <= edgeProb)
                {
                    double weight = (edgeMax - edgeMin) * ((double)rand() / (double)RAND_MAX) + edgeMin;
                    adjacencies[i].push_back({ j, weight });
                }
            }
        }

        vector<vector<uint32_t>> islands;
        GetIslands(islands);

        if (islands.size() > 1)
        {
            ConnectIslands(islands);
        }
    }

    /**
     * RandomGraph::GetNumIslands - Get islands in the graph. Visit each vert. If not visisted so far,
     * create an island, then walk the vert's edges in depth-first order, marking verts as visited and
     * adding them to the current island.
     *
     * @param A list of islands to be populated. Each island is a list of nodes connected with edges.
     * Assumed empty on input.
     */

    void GetIslands(vector<vector<uint32_t>> &islands)
    {
        assert(islands.size() == 0);

        vector<bool> visited(numVerts, false);
        vector<uint32_t> nodeStack;
        uint32_t curIsland = 0;

        for (uint32_t i = 0; i < numVerts; i++)
        {
            if (visited[i] == false)
            {
                uint32_t curNode = i;
                nodeStack.push_back(i);
                visited[i] = true;

                islands.push_back({ i });

                while (!nodeStack.empty())
                {
                    for (uint32_t j = 0; j < adjacencies[curNode].size(); j++)
                    {
                        pair<uint32_t, double> &edge = adjacencies[curNode][j];

                        if (visited[edge.first] == false)
                        {
                            curNode = edge.first;
                            nodeStack.push_back(curNode);
                            visited[curNode] = true;
                            islands[curIsland].push_back(curNode);
                            break;
                        }

                        if (j == adjacencies[curNode].size() - 1)
                        {
                            nodeStack.pop_back();
                            if (!nodeStack.empty()) curNode = nodeStack.back();
                        }
                    }
                }

                curIsland++;
            }
        }
    }

    /**
     * RandomGraph::ConnectIslands - If a graph has multiple islands,
     * randomly pick nodes between islands and add edges.
     *
     * @param islands List of islands to connect.
     */

    void ConnectIslands(vector<vector<uint32_t>> &islands)
    {
        for (uint32_t i = 0; i < islands.size(); i++)
        {
            for (uint32_t j = i + 1; j < islands.size(); j++)
            {
                uint32_t v1 = rand() % islands[i].size();
                uint32_t v2 = rand() % islands[j].size();

                double weight = (edgeMax - edgeMin) * ((double)rand() / (double)RAND_MAX) + edgeMin;

                adjacencies[v1].push_back({ v2, weight });

                v1 = rand() % islands[i].size();
                v2 = rand() % islands[j].size();

                weight = (edgeMax - edgeMin) * ((double)rand() / (double)RAND_MAX) + edgeMin;
                adjacencies[v2].push_back({ v1, weight });
            }
        }
    }
};