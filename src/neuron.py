from utils import CustomRandom
random = CustomRandom()

'''
Neuron class, based on :
Output = ϕ(∑ (xi ⋅ wi) + b)
'''
class Neuron : 
    def __init__(self, input_size, bias=0) :
        self.weights = random.randn(input_size)
        self.bias = bias
        self.zero_grad()
        
    def zero_grad(self) :
        self.grad_weights = [0.0] * len(self.weights)
        self.grad_bias = 0.


    def forward(self, inputs) :
        self.inputs = inputs
        
        self.value = sum(
            x * w
            for x, w in zip(inputs, self.weights)
        )
        self.value += self.bias

        return self.value


    def backward(self, grad):

        # On calcule la responsabilité de chaque weight dans la production de 
        for i, x in enumerate(self.inputs) : 
            self.grad_weights[i] += x * grad

        # on ajuste le biais
        self.grad_bias += grad

        # On retourne les gradients pour la layer précédente
        # en déterminant l'importance qu'a eu chaque poids 
        # dans la production de la réponse
        return [grad * w for w in self.weights]

