import random


class CustomRandom : 

    def randn(self, inputs):
        out=[]
        for _ in range(inputs) :
            out.append(random.uniform(-1, 1))

        return out
            
