
import torch
import torch.nn.functional as F

def sgd_training_step(inputs, targets, weights, bias, learning_rate):

    # 1. Forward pass
    z = torch.mv(inputs, weights) + bias
    predictions = torch.tanh(z)

    # 2. Loss before update
    loss_before = torch.sum((predictions - targets) ** 2)

    # 3. Calculate gradients using chain rule
    grad = 2 * (predictions - targets) * (1 - predictions ** 2)

    # 4. Weight gradient
    weight_grad = torch.sum(
        grad.unsqueeze(1) * inputs,
        dim=0
    )

    # 5. Bias gradient
    bias_grad = torch.sum(grad)

    # 6. SGD parameter updates
    updated_weights = weights - learning_rate * weight_grad
    updated_bias = bias - learning_rate * bias_grad

    # 7. Loss after update
    new_predictions = torch.tanh(
        torch.mv(inputs, updated_weights) + updated_bias
    )

    loss_after = torch.sum((new_predictions - targets) ** 2)

    # 8. Return required tensors
    return (
        loss_before,
        loss_after,
        updated_weights,
        updated_bias,
        weight_grad,
        bias_grad
    )
