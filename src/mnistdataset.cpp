#include "mnistdataset.h"

/**
 * MNISTDataSet::Init Load MNIST image data and associated labels from text file. Images
 * are 28x28 and stored as length 784 vectors. Labels are uint8s.
 * 
 * @param dataFile  Path to file containing image data.
 * @param labelFile Path to file containing labels.
 */

void MNISTDataSet::Init(const char* dataFile, const char* labelFile)
{
    printf("Loading MNIST data set...\n");
    printf("Data File: %s\n", dataFile);
    printf("Label File: %s\n\n", labelFile);

    FILE *df = fopen(dataFile, "r");
    char lineBuffer[128];
    fgets(lineBuffer, 128, df);

    // Read data file header (see format here: http://yann.lecun.com/exdb/mnist/)

    char* tok = strtok(lineBuffer, " ");
    tok = strtok(NULL, " ");
    tok = strtok(NULL, " ");

    tok = strtok(NULL, " "); // number of images
    uint32_t numImgsIn = strtol(tok, NULL, 16);
    numImgs = numImgsIn;
    tok = strtok(NULL, " ");

    tok = strtok(NULL, " "); // width
    uint32_t width = strtol(tok, NULL, 16);
    tok = strtok(NULL, " ");

    tok = strtok(NULL, " "); // height

    uint32_t height = strtol(tok, NULL, 16);
    uint32_t imgSizeIn = width * height;
    imgSize = imgSizeIn;
    uint32_t linesPerImage = (imgSize) / 16;

    // Allocate space for images and labels.

    data.resize(numImgs);
    labels.resize(numImgs);

    for (auto &image : data)
    {
        image.resize(imgSize);
    }

    // Load image Data.

    for (uint32_t i = 0; i < numImgs; i++)
    {
        if (i % 1000 == 0)
            printf("Loading image %u...\n", i);

        uint32_t curPixel = 0;

        for (uint32_t j = 0; j < linesPerImage; j++)
        {
            fgets(lineBuffer, 128, df);
            tok = strtok(lineBuffer, " ");
            
            for (uint32_t k = 0; k < 8; k++)
            {
                uint32_t pixels = strtol(tok, NULL, 16);
                data[i][curPixel++] = (double)((pixels >> 8) & 0xff) / 256.0f;
                data[i][curPixel++] = (double)((pixels) & 0xff) / 256.0f;
                tok = strtok(NULL, " ");
            }
        }
    }

    fclose(df);

    // Load labels.

    df = fopen(labelFile, "r");
    fgets(lineBuffer, 128, df);
    uint32_t curLabel = 0;

    tok = strtok(lineBuffer, " ");
    tok = strtok(NULL, " ");
    tok = strtok(NULL, " ");
    tok = strtok(NULL, " ");

    // First four labels are on first line

    for (int i = 0; i < 4; i++)
    {
        tok = strtok(NULL, " ");
        uint32_t label = strtol(tok, NULL, 10);
        labels[curLabel++] = (label / 100);
        labels[curLabel++] = (label % 100);
    }

    // Load remaining labels.

    for (uint32_t i = 0; i < numImgs / 16; i++)
    {
        fgets(lineBuffer, 128, df);
        tok = strtok(lineBuffer, " ");

        for (uint32_t j = 0; j < 8; j++)
        {
            if (curLabel == numImgs)
                break;

            uint32_t label = strtol(tok, NULL, 10);
            labels[curLabel++] = (label / 100);
            labels[curLabel++] = (label % 100);
            tok = strtok(NULL, " ");
        }
    }

    fclose(df);
}