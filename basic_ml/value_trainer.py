from value_backprop import MLP

class ValueTrainer:
  """Trains a Value MLP by L2 loss and gradient descent."""

  def train(self, model:MLP, xs, ys, iterations=100, lr=0.01):
    """Trains the input MLP model by inputs `xs` and outputs `ys`.

    Gradient descent's loss is L2.

    Args:
        model: A MLP to be trained on the inputs.
        xs: A list of inputs. Example: [[2.0, 3.0, -1.0], [3.0, -1.0, 0.5], 
            [0.5, 1.0, 1.0], [1.0, 1.0, -1.0]]
        ys: A list of outputs. Example: [1.0, -1.0, -1.0, 1.0]
        iterations: Number of iterations to run gradient descent.
        lr: Learning rate of gradient descent.
    
    Returns:
        The trained mlp model ready for inference.
    """
    for i in range(iterations):
      # Forward pass
      ypred = [model(x) for x in xs]
      # L2 Loss
      loss = sum([(yout - ygt) ** 2 for ygt, yout in zip(ys, ypred)])
      # Backward pass
      # Need to do zero grad!
      for p in model.parameters():
          p.grad = 0.0
      loss.backward()
      print(i, loss)
      # Gradient descent
      for p in model.parameters():
          p.data -= lr * p.grad 
    return model
