#| Entrée   | Classe |
#| -------- | ------ |
#| `[0, 0]` | 0      |
#| `[0, 1]` | 1      |
#| `[1, 0]` | 1      |
#| `[1, 1]` | 0      |
#
#- [ ] Réseau : `Dense(2, 8) -> ReLU -> Dense(8, 2) -> Softmax`
#- [ ] Boucle : forward, coût, backward (couches en ordre inverse), `step`, `zero_grad`
#- [ ] Afficher le coût moyen toutes les 100 epochs
#- [ ] Le coût tend vers 0 et les 4 prédictions sont justes
#
# Si le coût stagne, le problème vient presque toujours de l'étape 3.

from layers import Dense
from activation import Softmax, ReLU
from cost import CrossEntropy
from optimize import SGD

sgd = SGD()
loss = CrossEntropy()

layers = [Dense(2, 8), ReLU(),Dense(8, 2) ,Softmax()]
dense_layers = [layers[0], layers[2]]

inputs = [
    [[0, 0], 0],
    [[0, 1], 1],
    [[1, 0], 1],
    [[1, 1], 0]
]




def forward(probs):
    for layer in layers :
        probs = layer.forward(probs)

    return probs


def backward(probs, target): 
    grad = loss.backward(probs, target)
    for layer in reversed(layers[:-1]) : # on skip softmax
        grad = layer.backward(grad)
    return grad

def boucle(epochs=100) : 
    for epoch in range(epochs) : 
        losses=[]
        for pair in inputs : 
            el = pair[0]
            target = pair[1]
            probs = forward(el)
            losses.append(loss.forward(probs, target))
            grad = backward(probs, target)
        sgd.step(dense_layers, len(inputs))
        if epoch % 100 == 0  : print(f"Epoch {epoch} completed, loss : {sum(losses)/len(inputs)}")
        [layer.zero_grad() for layer in dense_layers]



def testing() :
    for pair in inputs :
        predictions = forward(pair[0])
        print(predictions, pair[1])


boucle(10000) 
testing()