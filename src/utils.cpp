#include "utils.h"

/**
 * [print description]
 * @param dev            [description]
 * @param m              [description]
 * @param n              [description]
 * @param truncateOutput [description]
 */

void print(float *dev, uint32_t m, uint32_t n, bool truncateOutput)
{
    cout << "\nCuda mem @" << dev << ": " << endl << endl;

    vector<float> host(m * n);
    cudaMemcpy(&host[0], dev, m * n * sizeof(float), cudaMemcpyDeviceToHost);
    cout << setprecision(2);

    for (uint32_t i = 0; i < m; i++)
    {
        for (uint32_t j = 0; j < n; j++)
        {
            if (truncateOutput && j > 9)
            {
                cout << " ... ";
                break;
            }

            cout << setw(8) << std::right << host[i * n + j];
        }

        cout << endl;
    }

    cout << endl << endl;
}

/**
 * [print description]
 * @param dev            [description]
 * @param m              [description]
 * @param n              [description]
 * @param truncateOutput [description]
 */

void print(double *dev, uint32_t m, uint32_t n, bool truncateOutput)
{
    cout << "Cuda mem @" << dev << ": " << endl << endl;

    vector<double> host(m * n);
    cudaMemcpy(&host[0], dev, m * n * sizeof(double), cudaMemcpyDeviceToHost);
    cout << setprecision(2) << setw(8) << std::right;

    for (uint32_t i = 0; i < m; i++)
    {
        for (uint32_t j = 0; j < n; j++)
        {
            if (truncateOutput && j > 9)
            {
                cout << " ... ";
                break;
            }

            cout << setw(8) << std::right << host[i * n + j];
        }

        cout << endl;
    }

    cout << endl << endl;
}