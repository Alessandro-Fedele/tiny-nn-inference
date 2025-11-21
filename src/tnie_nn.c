#include <stdlib.h>
#include <string.h>
#include "tnie_nn.h"
#include "tnie_activations.h"

void tnie_dense_forward(const TNIE_DenseLayer *layer,
                        const float *input,
                        float *output) {
    if (!layer || !input || !output) {
        return;
    }

    int in_size  = layer->input_size;
    int out_size = layer->output_size;

    for (int o = 0; o < out_size; ++o) {
        // Start from the bias term for this output neuron.
        float sum = layer->biases[o];

        // Pointer to the row of weights associated with this output neuron.
        const float *w_row = &layer->weights[o * in_size];

        // Weighted sum of all input neurons.
        for (int i = 0; i < in_size; ++i) {
            sum += w_row[i] * input[i];
        }

        output[o] = sum;
    }
}

int tnie_nn_forward(const TNIE_NeuralNetwork *nn,
                    const float *input,
                    float *output) {
    if (!nn || nn->num_layers <= 0 || !nn->layers || !nn->activations) {
        return -1; // invalid network
    }

    if (!input || !output) {
        return -2; // invalid buffers
    }

    // Determine the maximum layer size to allocate working buffers.
    int max_size = 0;
    for (int l = 0; l < nn->num_layers; ++l) {
        if (nn->layers[l].output_size > max_size) {
            max_size = nn->layers[l].output_size;
        }
    }

    if (max_size <= 0) {
        return -3; // corrupted network configuration
    }

    // Allocate two working buffers to alternate between layers.
    float *buf_a = (float *)malloc(sizeof(float) * max_size);
    float *buf_b = (float *)malloc(sizeof(float) * max_size);
    if (!buf_a || !buf_b) {
        free(buf_a);
        free(buf_b);
        return -4; // memory allocation failure
    }

    // Copy the input into the first buffer.
    const TNIE_DenseLayer *first_layer = &nn->layers[0];
    if (first_layer->input_size <= 0) {
        free(buf_a);
        free(buf_b);
        return -5;
    }

    // We only copy exactly input_size elements from the input.
    memcpy(buf_a, input, sizeof(float) * first_layer->input_size);

    float *current_in  = buf_a;
    float *current_out = buf_b;

    // Forward pass through each layer.
    for (int l = 0; l < nn->num_layers; ++l) {
        const TNIE_DenseLayer *layer = &nn->layers[l];

        // Compute linear transformation: y = W * x + b
        tnie_dense_forward(layer, current_in, current_out);

        // Apply activation in-place on the output buffer.
        tnie_apply_activation(nn->activations[l],
                              current_out,
                              layer->output_size);

        // If this is not the last layer, swap input/output buffers.
        // Otherwise, copy final result into the user-provided output.
        if (l < nn->num_layers - 1) {
            float *tmp   = current_in;
            current_in   = current_out;
            current_out  = tmp;
        } else {
            memcpy(output, current_out,
                   sizeof(float) * layer->output_size);
        }
    }

    free(buf_a);
    free(buf_b);
    return 0;
}

void tnie_nn_free(TNIE_NeuralNetwork *nn) {
    if (!nn) {
        return;
    }

    if (nn->layers) {
        for (int l = 0; l < nn->num_layers; ++l) {
            free(nn->layers[l].weights);
            free(nn->layers[l].biases);
            nn->layers[l].weights = NULL;
            nn->layers[l].biases  = NULL;
        }
        free(nn->layers);
        nn->layers = NULL;
    }

    if (nn->activations) {
        free(nn->activations);
        nn->activations = NULL;
    }

    free(nn);
}
