# HSP
TP Hardware for signal processing

**Part 1**
Notre algorithme réalise :
• Création d'une matrice sur CPU                  (0.15ms for a 32x32)
• Affichage d'une matrice sur CPU                 (0.81ms for a 32x32)
• Addition de deux matrices sur CPU               
• Addition de deux matrices sur GPU               
• Multiplication de deux matrices NxN sur CPU     
• Multiplication de deux matrices NxN sur GPU     

Compléxité et temps de calcul :
![HSPseance1screenfinal](https://user-images.githubusercontent.com/93649903/211338506-9e682020-136d-4b5d-ac4a-b1ca6edf020d.JPG)

• Création d'une matrice sur CPU                  (0.15ms for a 32x32)
• Affichage d'une matrice sur CPU                 (0.81ms for a 32x32)
• Addition de deux matrices sur CPU               (0.087ms for two matrix 32x32)  (11ms for two matrix 1000x1000)
• Addition de deux matrices sur GPU               (0.075ms for two matrix 32x32)  (0.083ms for two matrix 1000x1000)
• Multiplication de deux matrices NxN sur CPU     (0.443ms for two matrix 32x32)  (8.441s for two matrix 1000x1000)
• Multiplication de deux matrices NxN sur GPU     (0.078ms for two matrix 32x32)  (0.085ms for two matrix 1000x1000)
