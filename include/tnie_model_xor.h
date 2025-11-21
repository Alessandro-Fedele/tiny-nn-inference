#ifndef TNIE_MODEL_XOR_H
#define TNIE_MODEL_XOR_H

/**
 * @file tnie_model_xor.h
 * @brief Factory function for a demo neural network approximating XOR.
 *
 * This model is intentionally hardcoded and only used as a minimal
 * demonstration that the inference engine works correctly.
 */

#include "tnie_nn.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Create a small neural network approximating the XOR function.
 *
 * Architecture:
 *  - Input:  2 neurons
 *  - Hidden: 2 neurons, sigmoid
 *  - Output: 1 neuron, sigmoid
 *
 * The returned pointer must be released with tnie_nn_free().
 *
 * @return Pointer to a TNIE_NeuralNetwork instance, or NULL on failure.
 */
TNIE_NeuralNetwork *tnie_create_xor_demo_network(void);

#ifdef __cplusplus
}
#endif

#endif // TNIE_MODEL_XOR_H
