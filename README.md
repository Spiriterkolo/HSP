# TP HSP SIA G2 DELABARRE GILLET
TP Hardware for signal processing : Implémentation d'un CNN - LeNet-5 sur GPU



**Part 1**
Notre algorithme réalise :

• Création d'une matrice sur CPU                  (0.15ms for a 32x32)

• Affichage d'une matrice sur CPU                 (0.81ms for a 32x32)

• Addition de deux matrices sur CPU            

• Addition de deux matrices sur GPU    

• Multiplication de deux matrices NxN sur CPU     

• Multiplication de deux matrices NxN sur GPU     



**Temps de calcul :**
![HSPseance1screenfinal](https://user-images.githubusercontent.com/93649903/211338506-9e682020-136d-4b5d-ac4a-b1ca6edf020d.JPG)

• Création d'une matrice sur CPU                  (0.15ms for a 32x32)

• Affichage d'une matrice sur CPU                 (0.81ms for a 32x32)

• Addition de deux matrices sur CPU               (0.087ms for two matrix 32x32)  (11ms for two matrix 1000x1000)

• Addition de deux matrices sur GPU               (0.075ms for two matrix 32x32)  (0.083ms for two matrix 1000x1000)

• Multiplication de deux matrices NxN sur CPU     (0.443ms for two matrix 32x32)  (8.441s for two matrix 1000x1000)

• Multiplication de deux matrices NxN sur GPU     (0.078ms for two matrix 32x32)  (0.085ms for two matrix 1000x1000)
 
  
   
Caractéristiques GPU :

Intel Xeon E5-2650 v2

8C/16T @ 2.6GHz-3.4GHz

394.6 GFLOPS = 3.94 ∗ 1011 FLOPS

TDP : 95 W

Price : 1166$

Released date : Q3-2013

**Part  2** Premières couches du réseau de neurone LeNet-5 : Convolution 2D et subsampling

L'architecture du réseau LeNet-5 est composé de plusieurs couches, dont voici les premières :

Layer 1- Couche d'entrée de taille 32x32 correspondant à la taille des images de la base de donnée MNIST

Layer 2- Convolution avec 6 noyaux de convolution de taille 5x5. La taille résultantes est donc de 6x28x28.

Layer 3- Sous-échantillonnage d'un facteur 2. La taille résultantes des données est donc de 6x14x14.

**Part 3** Dernières couches et récupération de vrais poids

Architecture finale de notre réseau Net-5 :

![image](https://user-images.githubusercontent.com/93649903/211815911-69e96ed1-fcc4-40d3-b22e-1d5999d002af.png)

Dont les différentes étapes sont (en Python):

• Conv2D(6, kernel_size=5, strides=1,  activation='tanh', input_shape=train_x[0].shape, padding='same')

• AveragePooling2D()

• Conv2D(16, kernel_size=5, strides=1, activation='tanh', padding='valid')

• AveragePooling2D()

• Flatten() (que l'on ne fera pas ici puisque l'on travaille avec des vecteurs)

• Dense(120, activation='tanh')

• Dense(84, activation='tanh')

• Dense(10, activation='softmax')

Avec des poids et des biais provenant d'un entraînement précédant, il faut les initialiser dans main.cu dans les variables W1, W2, W3, B1, B2 et B3.

On peut ensuite tester le modèle sur n'importe quelle image 28x28x1.
