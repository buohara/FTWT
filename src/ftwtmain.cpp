#include "tests.h"

typedef void(*pfnTest)(void);

struct TestCase
{
    pfnTest pfnTest;
    string desc;
};

map<string, TestCase> problems =
{
    { "SimpleCross", { SimpleCrossTest, "SimpleCrossTest - Train a network to cross two inputs to two outputs {0, 1} -> {1, 0}." } },
    { "MNISTTest", { MNISTTest, "MNISTTest - Train a network to idenfity MNIST digit images." } },
    { "MNISTRandTest", {MNISTRandTest, "MNISTRandTest - Train a randombly generated network to identify MNIST digit images."}}
};

/**
 * DisplayTests - Display list of available tests cases that use FTWT models (user runs case by
 * specifying test name on command line).
 */

void DisplayTests()
{
    printf("Available Tests:\n\n");

    for (auto& problem : problems)
    {
        printf("%s: %s\n", problem.first.c_str(), problem.second.desc.c_str());
    }

    printf("\nPress any key to continue ...\n");
    getchar();
}

/**
 * main - Parse command line arguments and kickoff test case specified on command line.
 *
 * @param  argc Number of command line args.
 * @param  argv Array of command args.
 * @return      Program success code (not used).
 */

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        printf("Please specify a test name\n\n");
        DisplayTests();
        exit(0);
    }

    string testStr = string(argv[1]);

    if (problems.count(testStr) == 0)
    {
        printf("Invalid test number specified: %s\n\n", testStr.c_str());
        DisplayTests();
        exit(0);
    }

    problems[testStr].pfnTest();
}