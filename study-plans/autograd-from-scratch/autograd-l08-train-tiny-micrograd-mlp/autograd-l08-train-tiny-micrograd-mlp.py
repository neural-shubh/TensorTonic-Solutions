
import torch

def train_tiny_micrograd_mlp(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    weights: list[torch.Tensor],
    biases: list[torch.Tensor],
    learning_rate: float,
    steps: int
) -> tuple:

    # Clone parameters so original tensors remain unchanged
    trained_weights = [w.detach().clone() for w in weights]
    trained_biases = [b.detach().clone() for b in biases]

    loss_history = []

    def forward(x):
        activations = [x]

        for w, b in zip(trained_weights, trained_biases):
            x = torch.tanh(x @ w.T + b)
            activations.append(x)

        return activations

    for step in range(steps):

        # 1. Forward pass
        activations = forward(inputs)
        predictions = activations[-1].squeeze(-1)

        # 2. Calculate summed squared error
        loss = ((predictions - targets) ** 2).sum()

        # 3. Store pre-update loss
        loss_history.append(loss.clone())

        # 4. Manual backpropagation
        grad_weights = [None] * len(trained_weights)
        grad_biases = [None] * len(trained_biases)

        # Gradient of loss with respect to output activation
        grad = (2 * (predictions - targets)).unsqueeze(-1)

        for i in reversed(range(len(trained_weights))):

            # Derivative of tanh
            grad_z = grad * (1 - activations[i + 1] ** 2)

            # Gradients of weights and biases
            grad_weights[i] = grad_z.T @ activations[i]
            grad_biases[i] = grad_z.sum(dim=0)

            # Propagate gradient to previous layer
            grad = grad_z @ trained_weights[i]

        # 5. Update all parameters
        with torch.no_grad():

            for i in range(len(trained_weights)):
                trained_weights[i] -= learning_rate * grad_weights[i]
                trained_biases[i] -= learning_rate * grad_biases[i]

    # 6. Calculate final predictions and loss
    with torch.no_grad():
        activations = forward(inputs)
        predictions = activations[-1].squeeze(-1)

        final_loss = ((predictions - targets) ** 2).sum()

    return (
        predictions,
        final_loss,
        trained_weights,
        trained_biases,
        loss_history
    )
