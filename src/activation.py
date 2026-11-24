import math

class ReLU :
    def forward(self, inputs):
        self.inputs = inputs
        return [max(x, 0) for x in inputs]

    # On recoit une liste, et pour chaque index si l'input du meme 
    # index etait positif alors on laisse la valeur de la liste, sinon 0
    def backward(self, grad_output):
        copy = grad_output.copy()
        for i, x in enumerate(grad_output):
            if (self.inputs[i]>= 0) :
                copy[i] = x 
            else: copy[i]=0 
        return copy



    def info(self):
        return "RelU activation function"



class Softmax :
    
    # transfome les inputs en probabililites, qui somment à 1
    def forward(self, inputs):
        m = max(inputs)
        exps = [math.exp(x - m) for x in inputs]
        total = sum(exps)
        return [e / total for e in exps]


    def backward(self):
        return 


relu = ReLU()

print(relu.forward([1, 3, 4]))
print(relu.backward([0.2, 0.1, 0.3]))