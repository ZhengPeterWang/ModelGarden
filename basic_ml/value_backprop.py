"""Implements the Back Propagation Algorithm from scratch with a Python 
   wrapper `Value` class.
"""
import math
import random

class Value:
  """Wraps python values to perform automatic differentiation.
  
  This class is like `BigInteger` class in C++, except that it binds all `Value`
  nodes into a tree.

  Attributes:
      data: A python double wrapped in `Value`.
      grad: A double recording the current gradient of the `Value` node. Need to
        zero the gradients of all `Value`s in the computational graph out at
        every iteration.
      label: A string to identify this `Value` node.
  """

  def __init__(self, data, _children=(), _op = '', label = ''):
    """Initializes a `Value` wrapper.
    
    Args:
      data: The python double wrapped in `Value`
      _children: A tuple of `Value`s that are the children of this node. Point
        of this is to capture dependencies of `self._backward()` so that
        backprop can run correctly.
      _op: A string indicating the operator of this `Value` node, if this value
        node has children. Should be an empty string if this `Value` node is a 
        leaf node.
      label: A string to identify this `Value` node.
    """
    self.data = data
    self.grad = 0.0
    self._prev = set(_children)
    self._op = _op
    self._backward = lambda: None
    self.label = label
  
  def __repr__(self):
    return f"Value(data={self.data}, label={self.label})"
  
  def __add__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data + other.data, (self, other), '+')
    def _backward():
      self.grad += 1.0 * out.grad
      other.grad += 1.0 * out.grad
    out._backward = _backward
    return out
  
  def __mul__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data * other.data, (self, other), '*')
    def _backward():
      self.grad += other.data * out.grad
      other.grad += self.data * out.grad
    out._backward = _backward
    return out
  
  def __rmul__(self, other):
    return self * other
  
  def __radd__(self, other):
    return self + other
  
  def __truediv__(self, other):
    return self * other ** -1

  def __neg__(self):
    return self * -1
  
  def __sub__(self, other):
    return self + (-other)
  
  def __pow__(self, other):
    if not isinstance(other, (int, float)):
      raise ValueError("Only supporting int and float powers for now")
    out = Value(self.data ** other, (self,), f'**{other}')

    def _backward():
      self.grad += other * self.data ** (other - 1) * out.grad
    out._backward = _backward
    return out
  
  def tanh(self):
    n = self.data
    t = (math.exp(2 * n) - 1) / (math.exp(2 * n) + 1)
    out = Value(t, (self, ), "tanh")
    def _backward():
      self.grad += (1 - t ** 2) * out.grad
    out._backward = _backward
    return out
  
  def exp(self):
    x = self.data
    out = Value(math.exp(x), (self,), 'exp')

    def _backward():
      self.grad += out.data * out.grad
    out._backward = _backward
    return out
  
  def backward(self):
    """Topological sort self's children, then backprop from the leaf nodes."""
    topo = []
    visited = set()
    def build_topo(v):
      if v not in visited:
        visited.add(v)
        for child in v._prev:
          build_topo(child)
        topo.append(v)
    build_topo(self)
    self.grad = 1.0
    for node in reversed(topo):
      node._backward()

class Neuron:
  """Linear classifier tanh(w.x + b).

  Attributes:
      w: A list of `Value`s that are the weights of the classifier.
      b: A `Value` that is the bias of the classifier.
  """

  def __init__(self, nin):
    """
    Args:
      nin: Input dimension of the linear classifier.
    """
    self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
    self.b = Value(random.uniform(-1, 1))
  
  def __call__(self, x):
    """Applies the linear classifier on an input `x`.

    Args:
        x: a list of `Value`s to apply the linear classifier to.
    
    Returns:
        A `Value` node that is the output of the linear classifier.
    """
    # tanh(w1x1 + w2x2 + w3x3 + ... + wnxn + b)
    act = sum(w1 * x1 for w1, x1 in zip(self.w, x)) + self.b
    out = act.tanh()
    return out
  
  def parameters(self):
    """A list of the classifier's parameters."""
    return self.w + [self.b]

class Layer:
  """Fully connected (FC) layer tanh(W.x + b).

  Attributes:
      neurons: a list of `Neuron`s forming the `W` matrix.
  """

  def __init__(self, nin, nout):
    """
    Args:
      nin: Input dimension of the FC layer.
      nout: Output dimension of the FC layer.
    """
    self.neurons = [Neuron(nin) for _ in range(nout)]
  
  def __call__(self, x):
    outs = [n(x) for n in self.neurons]
    return outs[0] if len(outs) == 1 else outs
  
  def parameters(self):
    return [p for neuron in self.neurons for p in neuron.parameters()]

class MLP:
  """Multi layer perceptron (MLP) formed by FC layers with different dimensions.

  Attributes:
      layers: A list of `Layer`s storing the weights of this MLP.
  """

  def __init__(self, nin, nouts):
    """
    Args:
      nin: Input dimension of the MLP.
      nouts: Middle and output dimensions of the MLP.
    """
    sz = [nin] + nouts
    self.layers = [Layer(sz[i], sz[i + 1]) for i in range(len(nouts))]
  
  def __call__(self, x):
    for layer in self.layers:
      x = layer(x)
    return x
  
  def parameters(self):
    return [p for layer in self.layers for p in layer.parameters()]