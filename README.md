# Neural Network from Scratch

Réseau de neurones écrit entièrement à la main en Python pur, entraîné sur Fashion MNIST.

## Sommaire

- [Objectif](#objectif)
- [Architecture](#architecture)
- [Étape 0 : Bases](#étape-0--bases)
- [Étape 1 : Activations](#étape-1--activations)
- [Étape 2 : Fonction de coût](#étape-2--fonction-de-coût)
- [Étape 3 : Rétropropagation](#étape-3--rétropropagation)
- [Étape 4 : Optimiseur](#étape-4--optimiseur)
- [Étape 5 : Validation sur XOR](#étape-5--validation-sur-xor)
- [Étape 6 : Chargement des données](#étape-6--chargement-des-données)
- [Étape 7 : Entraînement](#étape-7--entraînement)
- [Étape 8 : Évaluation](#étape-8--évaluation)
- [Bonus](#bonus)
- [Annexe : les 10 classes](#annexe--les-10-classes)

## Objectif

Classer les images de Fashion MNIST (10 catégories de vêtements) avec au moins
**80 % de précision** sur le jeu de test.

**Contraintes :**

- Pas de NumPy, TensorFlow ni PyTorch.
- Seule la bibliothèque standard est autorisée : `math`, `random`, `gzip`, `struct`.

## Architecture

Propagation avant :

```text
image (784) -> Dense(784, 64) -> ReLU -> Dense(64, 10) -> Softmax -> CrossEntropy
```

Rétropropagation :

```text
Dense.backward <- ReLU.backward <- Dense.backward <- CrossEntropy.backward
```

Softmax n'a pas de `backward` propre : son gradient est inclus dans celui de la
cross-entropy.

---

## Étape 0 : Bases

- [x] Classe `Neuron` avec `forward(inputs)`
- [x] Classe `Layer` / `Dense` avec `forward(inputs)`
- [x] Poids initialisés aléatoirement et petits : `random.gauss(0, 1) * math.sqrt(2 / input_size)`
- [x] `Neuron.forward` sauvegarde ses entrées dans `self.inputs`

## Étape 1 : Activations

Sans activation non linéaire, empiler des couches revient à n'avoir qu'une seule
couche linéaire.

### ReLU (couches cachées)

- [x] `forward(inputs)` renvoie `max(x, 0)` pour chaque élément et sauvegarde les entrées
- [x] `backward(grad_output)` renvoie `grad_output[i]` si l'entrée `i` était positive, sinon `0`

### Softmax (couche de sortie)

- [x] `forward(inputs)` renvoie `exp(x_i) / somme(exp(x_j))`
- [x] Soustraire le maximum avant `math.exp` pour éviter un dépassement
- [x] Test : `Softmax().forward([1, 2, 3])` donne environ `[0.09, 0.245, 0.665]`

## Étape 2 : Fonction de coût

La cross-entropy résume l'erreur du réseau en un seul nombre, que l'entraînement
cherche à faire baisser.

- [x] `forward(probs, target)` renvoie `-log(max(probs[target], 1e-12))`
- [x] `backward(probs, target)` renvoie une copie de `probs` avec `1` soustrait à l'indice `target`
- [x] Test : `probs[target] = 0.9` donne un coût d'environ `0.105`
- [x] Test : `probs[target] = 0.1` donne un coût d'environ `2.3`

## Étape 3 : Rétropropagation

Pour un neurone qui reçoit le gradient `grad` de sa sortie :

| Gradient de... | Formule                                  |
| -------------- | ---------------------------------------- |
| poids `w_i`    | `grad * x_i`                             |
| biais `b`      | `grad`                                   |
| entrée `x_i`   | `grad * w_i` (contribution du neurone)   |

La couche additionne les contributions de tous ses neurones pour obtenir le
gradient de chaque entrée.

- [x] `Neuron.zero_grad()` remet `grad_weights` et `grad_bias` à zéro (appelée dans `__init__`)
- [x] `Neuron.backward(grad)` accumule les gradients avec `+=` et renvoie `[grad * w for w in weights]`
- [x] `Layer.backward(grad_output)` additionne les contributions des neurones et renvoie `grad_input`
- [x] `Layer.zero_grad()` appelle `zero_grad` sur chaque neurone
- [x] Test chiffré réussi (voir ci-dessous)
- [x] Gradient check réussi (voir ci-dessous)

**Test chiffré :**

- Entrées : `[1.0, 2.0]`
- Poids du neurone A : `[0.5, -1.0]`
- Poids du neurone B : `[0.3, 0.4]`
- `grad_output` : `[0.2, -0.1]`
- Résultat attendu : `grad_input = [0.07, -0.24]`

**Gradient check :** pour un poids `w` et `ε = 1e-5`, la valeur
`(coût(w + ε) - coût(w - ε)) / (2ε)` doit être presque égale au gradient calculé.

## Étape 4 : Optimiseur

- [x] Classe `SGD(learning_rate)` avec une méthode `step(layers, batch_size)`
- [x] Mise à jour des poids : `w -= learning_rate * grad_w / batch_size`
- [x] Mise à jour des biais de la même façon
- [x] Taux d'apprentissage de départ : `0.1`

## Étape 5 : Validation sur XOR

On vérifie que le réseau apprend sur un problème minuscule avant de passer aux
vraies données.

| Entrée   | Classe |
| -------- | ------ |
| `[0, 0]` | 0      |
| `[0, 1]` | 1      |
| `[1, 0]` | 1      |
| `[1, 1]` | 0      |

- [ ] Réseau : `Dense(2, 8) -> ReLU -> Dense(8, 2) -> Softmax`
- [ ] Boucle : forward, coût, backward (couches en ordre inverse), `step`, `zero_grad`
- [ ] Afficher le coût moyen toutes les 100 epochs
- [ ] Le coût tend vers 0 et les 4 prédictions sont justes

Si le coût stagne, le problème vient presque toujours de l'étape 3.

## Étape 6 : Chargement des données

Fichiers disponibles sur le dépôt GitHub `zalandoresearch/fashion-mnist`, dans
le dossier `data/fashion`.

- [ ] Télécharger les 4 fichiers `.gz` (images et labels, entraînement et test)
- [ ] `load_images(path)` : en-tête de 16 octets lu avec `struct.unpack('>IIII', ...)`, puis 784 octets par image
- [ ] `load_labels(path)` : en-tête de 8 octets lu avec `'>II'`, puis 1 octet par label
- [ ] Normaliser les pixels en divisant par 255
- [ ] Test : 60 000 images d'entraînement et 10 000 de test, chacune de longueur 784

## Étape 7 : Entraînement

- [ ] Réseau : `Dense(784, 64) -> ReLU -> Dense(64, 10) -> Softmax`
- [ ] Mélanger les indices à chaque epoch avec `random.shuffle`
- [ ] Mini-batches de 32 : forward et backward sur chaque exemple, puis un seul `step` et `zero_grad`
- [ ] Afficher le coût moyen à chaque epoch
- [ ] Premier essai sur 5 000 images
- [ ] Entraînement sur le jeu complet

## Étape 8 : Évaluation

- [ ] `accuracy(network, images, labels)` : proportion d'images où `argmax(probs) == label`
- [ ] Afficher la précision sur le jeu de test après chaque epoch
- [ ] **Objectif atteint : précision supérieure ou égale à 80 %**

---

## Bonus

- [ ] Sauvegarder et recharger les poids avec `json`
- [ ] Afficher la matrice de confusion
- [ ] Ajouter une deuxième couche cachée
- [ ] Faire décroître le taux d'apprentissage au fil des epochs
- [ ] Ajouter un optimiseur à momentum
- [ ] Remplacer les objets `Neuron` par des matrices dans `Dense` et mesurer le gain

## Annexe : les 10 classes

| Label | Classe      |
| ----- | ----------- |
| 0     | T-shirt/top |
| 1     | Trouser     |
| 2     | Pullover    |
| 3     | Dress       |
| 4     | Coat        |
| 5     | Sandal      |
| 6     | Shirt       |
| 7     | Sneaker     |
| 8     | Bag         |
| 9     | Ankle boot  |