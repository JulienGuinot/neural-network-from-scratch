from cost import CrossEntropy
from layers import Dense 
loss = CrossEntropy()
from activation import Softmax, ReLU


def cost(layers, x, target):
    for layer in layers:
        x = layer.forward(x)
    return loss.forward(x, target)

layers = [Dense(2, 3), ReLU(), Dense(3, 2), Softmax()]
x, target, eps = [1.0, 2.0], 1, 1e-5

# 1. Gradient calculé par backprop
probs = x
for layer in layers:
    probs = layer.forward(probs)
grad = loss.backward(probs, target)
for layer in reversed(layers[:-1]):   # on saute Softmax
    grad = layer.backward(grad)


neuron = layers[0].neurons[0]
backprop = neuron.grad_weights[0]

# 2. Gradient mesuré à la main
w = neuron.weights[0]
neuron.weights[0] = w + eps
c_plus = cost(layers, x, target)
neuron.weights[0] = w - eps
c_minus = cost(layers, x, target)
neuron.weights[0] = w                 # on remet le poids d'origine

numeric = (c_plus - c_minus) / (2 * eps)
print(backprop, numeric)              # doivent être quasi identiques