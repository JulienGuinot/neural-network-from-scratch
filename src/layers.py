from neuron import Neuron



class Dense :
    def __init__(self, input_size, neuron_count) :
        self.neurons = [
            Neuron(input_size)
            for _ in range(neuron_count)
        ]

    def forward(self, inputs) :
        return [neuron.forward(inputs) for neuron in self.neurons]


    def backward(self, grad_output):

        # on fait un array vide de la taille des weights du premier neurone
        # la taille des weights car un input = 1 weight. 
        grad_input = [0.0] * len(self.neurons[0].weights)

        # pour chaque paire de neurone et grad output (recu de la couche suivante backward())
        # on appelle neuron.backward(grad_output) - renvoie une liste des gradients
        # qui correspondent à l'importance qu'a eu chaque poids dans la production de la reponse, 
        # sur laquelle on itère et on 
        for neuron, g in zip(self.neurons, grad_output):
            for i, c in enumerate(neuron.backward(g)):
                grad_input[i] += c
        return grad_input

    def info(self) :
        print("Neurons :", len(self.neurons))



    def zero_grad(self) :
        for neuron in self.neurons:
            neuron.zero_grad()





#inputs=[1.0, 2.0]
#
#layer = Dense(2, 2)
#
#layer.neurons[0].weights = [0.5, -1.0]
#layer.neurons[1].weights = [0.3, 0.4]
#grad_output = [0.2, -0.1]
#
#print(layer.forward(inputs))
#print(layer.backward(grad_output))

