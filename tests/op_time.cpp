//
// Created by Hugo on 15/01/2022.
//

#include "lib/data_structures/matrix/matrix.h"

using namespace cudaNN;

#define x 256
#define y 256

int main(int argc, char *argv[])
{
    auto m1 = matrix(x, y, "1");
    auto m2 = matrix( x, y, "2");
    for (size_t i = 0; i < x; i ++)
    {
        for (size_t j= 0; j < y; j ++)
        {
            m1[i * y + j] = 3;
            m2[i * y + j] = 3;
        }
    }

    cudaEvent_t startEvent,endEvent;
    float timeEvent;

    cudaEventCreate(&startEvent);
    cudaEventCreate(&endEvent);
    cudaEventRecord(startEvent, 0);
    m1 * m2;
    cudaEventRecord(endEvent, 0);
    cudaEventSynchronize(endEvent);
    cudaEventElapsedTime(&timeEvent, startEvent, endEvent);
    printf("Multiplication : %fms\n", timeEvent);
    cudaEventDestroy(startEvent);
    cudaEventDestroy(endEvent);

    cudaEventCreate(&startEvent);
    cudaEventCreate(&endEvent);
    cudaEventRecord(startEvent, 0);
    m1 + m2;
    cudaEventRecord(endEvent, 0);
    cudaEventSynchronize(endEvent);
    cudaEventElapsedTime(&timeEvent, startEvent, endEvent);
    printf("Addition : %fms\n", timeEvent);
    cudaEventDestroy(startEvent);
    cudaEventDestroy(endEvent);

    cudaEventCreate(&startEvent);
    cudaEventCreate(&endEvent);
    cudaEventRecord(startEvent, 0);
    m1 - m2;
    cudaEventRecord(endEvent, 0);
    cudaEventSynchronize(endEvent);
    cudaEventElapsedTime(&timeEvent, startEvent, endEvent);
    printf("Subtraction : %fms\n", timeEvent);
    cudaEventDestroy(startEvent);
    cudaEventDestroy(endEvent);

    cudaEventCreate(&startEvent);
    cudaEventCreate(&endEvent);
    cudaEventRecord(startEvent, 0);
    m1.hadamard_product(m2);
    cudaEventRecord(endEvent, 0);
    cudaEventSynchronize(endEvent);
    cudaEventElapsedTime(&timeEvent, startEvent, endEvent);
    printf("Hadamard product : %fms\n", timeEvent);
    cudaEventDestroy(startEvent);
    cudaEventDestroy(endEvent);
}
