#include <stdio.h>
#include "tnie_model_xor.h"
#include "tnie_nn.h"

int main(void) {
    TNIE_NeuralNetwork *nn = tnie_create_xor_demo_network();
    if (!nn) {
        fprintf(stderr, "[TNIE] Failed to create XOR demo network.\n");
        return 1;
    }

    // XOR truth table:
    // 0 xor 0 = 0
    // 0 xor 1 = 1
    // 1 xor 0 = 1
    // 1 xor 1 = 0
    float inputs[4][2] = {
        {0.0f, 0.0f},
        {0.0f, 1.0f},
        {1.0f, 0.0f},
        {1.0f, 1.0f}
    };
    int expected[4] = {0, 1, 1, 0};

    float output[1];

    printf("TNIE - Tiny Neural Inference Engine\n");
    printf("XOR demo with a hardcoded neural network model\n");
    printf("-------------------------------------------------\n");

    for (int i = 0; i < 4; ++i) {
        int err = tnie_nn_forward(nn, inputs[i], output);
        if (err != 0) {
            fprintf(stderr,
                    "[TNIE] Forward pass failed on sample %d (error code = %d)\n",
                    i, err);
            continue;
        }

        int predicted = (output[0] > 0.5f) ? 1 : 0;

        printf("Input: (%.1f, %.1f) -> raw = %.4f, predicted = %d, expected = %d\n",
               inputs[i][0],
               inputs[i][1],
               output[0],
               predicted,
               expected[i]);
    }

    tnie_nn_free(nn);
    return 0;
}
