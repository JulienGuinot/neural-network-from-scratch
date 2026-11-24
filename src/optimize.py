class SGD : 
    def __init__(self, learning_rate=0.1) : 
        self.learning_rate = learning_rate

    def step(self, layers, batch_size) : 
        # layers :
        # batch_size : 

        for layer in layers :
            for neuron in layer.neurons : 
                for i in range(len(neuron.weights)) : 
                    
                    # Nouveau poids = ancien poids − learning rate × gradient du poids ÷ taille du batch.
                    neuron.weights[i] -= ((self.learning_rate * neuron.grad_weights[i]) / batch_size)
                
                neuron.bias -= ((self.learning_rate * neuron.grad_bias) / batch_size)
        return
