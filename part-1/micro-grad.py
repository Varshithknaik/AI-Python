import math
import numpy as np
import matplotlib.pyplot as plt
from graphviz import Digraph
import random

def trace(root):
  # builds a set of all nodes and edges in a graph
  nodes, edges = set(), set()
  def build(v):
    if v not in nodes:
      nodes.add(v)
      for child in v._prev:
        edges.add((child, v))
        build(child)
  build(root)
  return nodes, edges

def draw_dot(root):
  dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'}) # LR = left to right
  
  nodes, edges = trace(root)
  for n in nodes:
    uid = str(id(n))
    # for any value in the graph, create a rectangular ('record') node for it
    dot.node(name = uid, label = "{ %s | data %.4f | grad %.4f }" % (n.label, n.data, n.grad), shape='record')
    if n._op:
      # if this value is a result of some operation, create an op node for it
      dot.node(name = uid + n._op, label = n._op)
      # and connect this node to it
      dot.edge(uid + n._op, uid)

  for n1, n2 in edges:
    # connect n1 to the op node of n2
    dot.edge(str(id(n1)), str(id(n2)) + n2._op)

  return dot


class Value: 
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0
        self._prev = set(_children)
        self._backward = lambda: None
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"


  
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out

    def __radd__(self, other):
      return self + other
    
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

    def __truediv__(self, other):
      return self * other**-1
    
    def __neg__(self):
      return self * -1

    def __sub__(self , other):
      other = other if isinstance(other, Value) else Value(other)
      return self + (-other)
    
    def __pow__(self, other):
        assert isinstance(other , (int , float)), "only supporting int/float powers for now"

        out = Value(self.data ** other, (self,), '**')

        def _backward():
          self.grad += other * (self.data ** (other - 1)) * out.grad
        out._backward = _backward
        return out
    def tanh(self):
        x = self.data
        out = Value((math.exp(2*x) - 1) / (math.exp(2*x) + 1), (self,), 'tanh')

        def _backward():
          self.grad += (1 - out.data**2) * out.grad
        out._backward = _backward
        return out

    def leakyRelu(self , alpha = 0.01):
        value = self.data if self.data > 0 else alpha*self.data
        out = Value(value , (self,) , 'leakyRelu')

        def _backward():
            local_grad = 1.0 if self.data > 0 else alpha
            self.grad += local_grad * out.grad
        out._backward = _backward
        return out
    def gelu(self):
        x = self.data
        cdf = 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
        pdf = math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)

        out = Value(x * cdf, (self,), "GELU")

        def _backward():
            local_grad = cdf + x * pdf
            self.grad += local_grad * out.grad

        out._backward = _backward
        return out
          
    def exp(self):
      x = self.data
      out = Value( math.exp(x) , (self,), 'exp')
      def _backward():
        self.grad += out.data * out.grad
      out._backward = _backward
      return out
    
    def backward(self):
      todo = []
      visited = set()
      def build_topo(v):
        if v not in visited:
          visited.add(v)
          for child in v._prev:
            build_topo(child)
          todo.append(v)
      build_topo(self)
      self.grad = 1
      for node in reversed(todo):
        node._backward()



class Neuron:
  def __init__(self , nin , activation = "tanh"):
    self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
    self.b = Value(random.uniform(-1, 1))
    self.activation = activation

  def __call__(self, x):
    act = sum((wi*xi for wi, xi in zip(self.w , x)) , self.b)
    if self.activation == "tanh":
      return act.tanh()
    elif self.activation == "leakyRelu":
      return act.leakyRelu()
    elif self.activation == "gelu":
      return act.gelu()
    elif self.activation == "linear":
      return act
    else:
      raise ValueError("Invalid activation function")

  def parameters(self):
    return self.w + [self.b]
class Layer:
  def __init__(self , nin, nout , activation):
    self.neurons = [Neuron(nin , activation=activation) for _ in range(nout)]
  def __call__(self, x):
    outs =  [n(x) for n in self.neurons]
    return outs[0] if len(outs) == 1 else outs

  def parameters(self):
    return [ p for neuron in self.neurons for p in neuron.parameters() ]

class MLP:
  def __init__(self, nin, nouts, hidden_activation="gelu" , output_activation="linear"):
    sz = [nin] + nouts
    activation = ([hidden_activation]* (len(nouts) - 1) ) + [output_activation]
    self.layers = [Layer(sz[i], sz[i+1] , activation[i]) for i in range(len(sz)-1)]

  def __call__(self, x):
    for layer in self.layers:
      x = layer(x)
    return x
  
  def parameters(self):
    return [p for layer in self.layers for p in layer.parameters()]

# x = [2.0 , 3.0 , -1.0]
mlp = MLP(4 , [5 , 5 , 1])

print(len(mlp.parameters()))

# dot = draw_dot(o)
# dot.render('computational_graph', view=False, cleanup=True)

xs = [
  [2.0, 3.0, -1.0],
  [3.0, -1.0, 0.5],
  [0.5, 1.0, 1.0],
  [1.0, 1.0, -1.0],
]
ys = [5., -0.20, -5.0, 1.0] # desired targets

for k in range(200 ):
  
  # forward pass
  ypred = [mlp(x) for x in xs]
  loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))
  
  # backward pass
  for p in mlp.parameters():
    p.grad = 0.0
  loss.backward()
  
  # update
  for p in mlp.parameters():
    p.data += -0.005 * p.grad
  
  print(k, loss.data)

print("===========YOUR PREDICTIONS===========#######################")
print(ypred)


dot = draw_dot(loss)
dot.render('computational_graph', view=False, cleanup=True)

