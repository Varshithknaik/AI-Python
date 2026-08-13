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


# inputs x1,x2
# x1 = Value(2.0, label='x1')
# x2 = Value(0.0, label='x2')
# # weights w1,w2
# w1 = Value(-3.0, label='w1')
# w2 = Value(1.0, label='w2')
# # bias of the neuron
# b = Value(6.8813735870195432, label='b')
# # x1*w1 + x2*w2 + b
# x1w1 = x1*w1; x1w1.label = 'x1*w1'
# x2w2 = x2*w2; x2w2.label = 'x2*w2'
# x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label = 'x1*w1 + x2*w2'
# n = x1w1x2w2 + b; n.label = 'n'
# o = n.tanh(); o.label = 'o'
 
# o.backward()

# dot = draw_dot(o)
# dot.render('computational_graph', view=False, cleanup=True)

# =============***********======================

# inputs x1,x2
x1 = Value(2.0, label='x1')
x2 = Value(0.0, label='x2')
# weights w1,w2
w1 = Value(-3.0, label='w1')
w2 = Value(1.0, label='w2')
# bias of the neuron
b = Value(6.8813735870195432, label='b')
# x1*w1 + x2*w2 + b
x1w1 = x1*w1; x1w1.label = 'x1*w1'
x2w2 = x2*w2; x2w2.label = 'x2*w2'
x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label = 'x1*w1 + x2*w2'
n = x1w1x2w2 + b; n.label = 'n'

# ====
e = (2*n).exp() ;e.label = 'e'  
o = (e - 1) / (e + 1) ;o.label = 'o'
# ====
o.backward()

dot = draw_dot(o)
dot.render('computational_graph', view=False, cleanup=True)


class Neuron:
  def __init__(self , nin):
    self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
    self.b = Value(random.uniform(-1, 1))

  def __call__(self, x):
    act = sum((wi*xi for wi, xi in zip(self.w , x)) , self.b)
    out = act.tanh()
    return out

  def parameters(self):
    return self.w + [self.b]
class Layer:
  def __init__(self , nin, nout):
    self.neurons = [Neuron(nin) for _ in range(nout)]
  def __call__(self, x):
    outs =  [n(x) for n in self.neurons]
    return outs[0] if len(outs) == 1 else outs

  def parameters(self):
    return [ p for neuron in self.neurons for p in neuron.parameters() ]

class MLP:
  def __init__(self, nin, nouts):
    sz = [nin] + nouts
    self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(sz)-1)]

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
ys = [0.5, -0.20, -1.0, 1.0] # desired targets

for k in range(200):
  
  # forward pass
  ypred = [mlp(x) for x in xs]
  loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))
  
  # backward pass
  for p in mlp.parameters():
    p.grad = 0.0
  loss.backward()
  
  # update
  for p in mlp.parameters():
    p.data += -0.01 * p.grad
  
  print(k, loss.data)

print("===========YOUR PREDICTIONS===========#######################")
print(ypred)


dot = draw_dot(loss)
dot.render('computational_graph', view=False, cleanup=True)

