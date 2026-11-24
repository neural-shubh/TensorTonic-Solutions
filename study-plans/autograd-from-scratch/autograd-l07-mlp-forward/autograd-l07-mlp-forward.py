import torch

def mlp_forward(inputs, weights, biases):
    h = inputs
    layer_outputs = []

    for w, b in zip(weights, biases):
        h = torch.mv(w, h) + b
        h = torch.tanh(h)
        layer_outputs.append(h)

    return h, layer_outputs
