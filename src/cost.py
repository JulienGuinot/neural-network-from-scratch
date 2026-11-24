import math


# Résume l'erreur du réseau en 1 seul nombre, 
# celui que l'entrainement cherche à faire baisser
class CrossEntropy: 

    def forward(self, probs, target) :
        return -math.log(max(probs[target], 1e-12))

    def backward(self, probs, target) :
        copy = probs.copy()
        copy[target] -=1
        return copy 

        


#ent = CrossEntropy()
#
#
#
#probs = [0.9, 0.1]
#target=1
#
#print(ent.forward(probs, target))
#print(ent.backward(probs, target))