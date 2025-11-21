#include <stdlib.h>
#include "tnie_model_xor.h"

/**
 * @brief Helper to allocate a float array and check for failure.
 */
static float *tnie_alloc_floats(size_t count) {
    return (float *)malloc(sizeof(float) * count);
}

TNIE_NeuralNetwork *tnie_create_xor_demo_network(void) {
    TNIE_NeuralNetwork *nn = (TNIE_NeuralNetwork *)malloc(sizeof(TNIE_NeuralNetwork));
    if (!nn) {
        return NULL;
    }

    nn->num_layers  = 2;
    nn->layers      = (TNIE_DenseLayer *)calloc((size_t)nn->num_layers,
                                                sizeof(TNIE_DenseLayer));
    nn->activations = (TNIE_ActivationType *)calloc((size_t)nn->num_layers,
                                                    sizeof(TNIE_ActivationType));

    if (!nn->layers || !nn->activations) {
        tnie_nn_free(nn);
        return NULL;
    }

    // ----- Layer 0: input (2) -> hidden (2), sigmoid -----
    TNIE_DenseLayer *l0 = &nn->layers[0];
    l0->input_size  = 2;
    l0->output_size = 2;
    l0->weights     = tnie_alloc_floats((size_t)l0->input_size *
                                        (size_t)l0->output_size);
    l0->biases      = tnie_alloc_floats((size_t)l0->output_size);

    // ----- Layer 1: hidden (2) -> output (1), sigmoid -----
    TNIE_DenseLayer *l1 = &nn->layers[1];
    l1->input_size  = 2;
    l1->output_size = 1;
    l1->weights     = tnie_alloc_floats((size_t)l1->input_size *
                                        (size_t)l1->output_size);
    l1->biases      = tnie_alloc_floats((size_t)l1->output_size);

    if (!l0->weights || !l0->biases || !l1->weights || !l1->biases) {
        tnie_nn_free(nn);
        return NULL;
    }

    // These weight values were chosen to approximate XOR with sigmoid neurons.
    //
    // Layer 0 weights layout (row-major):
    //   row 0 (hidden neuron h1): w00, w01
    //   row 1 (hidden neuron h2): w10, w11
    //
    // h1 = sigmoid( 20*x1 +  20*x2 - 10)
    // h2 = sigmoid(-20*x1 -  20*x2 + 30)
    //
    l0->weights[0] =  20.0f;  // h1 <- x1
    l0->weights[1] =  20.0f;  // h1 <- x2
    l0->weights[2] = -20.0f;  // h2 <- x1
    l0->weights[3] = -20.0f;  // h2 <- x2

    l0->biases[0] = -10.0f;   // bias for h1
    l0->biases[1] =  30.0f;   // bias for h2

    // Layer 1:
    //   y = sigmoid(20*h1 + 20*h2 - 30)
    l1->weights[0] = 20.0f;   // y <- h1
    l1->weights[1] = 20.0f;   // y <- h2
    l1->biases[0]  = -30.0f;  // bias for y

    nn->activations[0] = TNIE_ACT_SIGMOID;
    nn->activations[1] = TNIE_ACT_SIGMOID;

    return nn;
}
