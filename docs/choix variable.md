# Limite_par_nervures (ou delimite_par_nervures)
**Pourquoi choisir cette caractéristique pour le dataset ?**

* **En phytopathologie** (étude des maladies des plantes), l'interaction entre le pathogène et la structure anatomique de la feuille est un critère de diagnostic biologique fondamental.

La raison biologique : Le champignon de la cercosporiose (Gray Leaf Spot) ne parvient pas à traverser facilement les grandes nervures parallèles du maïs. Il se propage donc longitudinalement, ce qui donne des taches aux bords parfaitement droits et parallèles. À l'inverse, l'agent de l'helminthosporiose (Blight) ou de la rouille possède des mécanismes enzymatiques ou de pression qui lui permettent de franchir ces barrières, créant des formes de fuseaux ou de ronds.

# 2. Avantages de cette caractéristique
**Robustesse aux changements de lumière** : Contrairement à la couleur (lesion_color), qui change selon l'ensoleillement de la photo ou l'appareil utilisé, la géométrie et l'alignement par rapport aux nervures restent constants.

**Indépendance de l'échelle** : Que la photo soit prise de près ou de loin, une forme bloquée par une ligne droite verticale reste détectable par des algorithmes d'analyse de contours.

**Simplicité algorithmique** : Mesurer l'alignement d'un rectangle par rapport à l'axe principal de la feuille est plus simple et moins sujet à l'erreur que d'analyser des textures complexes.

# 3. Méthode d'extraction de la caractéristique:
Voici le résumé détaillé et structuré de chaque étape du traitement d'image effectué par le code OpenCV, avec l'explication concrète de son impact visuel.

---

## a. Prétraitement : Flou Gaussien (`cv2.GaussianBlur`)

* **Ce que fait le code :** Il applique un filtre de flou sur l'image en calculant la moyenne des pixels voisins à l'aide d'une matrice (ici de taille $5 \times 5$).
* **Effet concret sur l'image :** L'image devient légèrement floue. Cela permet de **lisser les imperfections**, d'atténuer les petits grains de poussière, les reflets de lumière ou les détails trop fins des nervures saines qui pourraient être confondus avec des morceaux de maladie.

---

## b. Seuillage Inversé (`cv2.threshold` avec `THRESH_BINARY_INV`)

* **Ce que fait le code :** Il examine chaque pixel en niveau de gris. Si le pixel est plus sombre qu'un certain seuil (ici `100`), il le transforme en **blanc pur** ($255$). S'il est plus clair, il le transforme en **noir pur** ($0$).
* **Effet concret sur l'image :** L'image couleur d'origine disparaît pour laisser place à une image en noir et blanc très contrastée (binaire). Les zones malades (généralement plus sombres ou nécrosées) apparaissent comme des "silhouettes" blanches sur un fond totalement noir.

---

## c. Extraction des contours (`cv2.findContours`)

* **Ce que fait le code :** Il parcourt l'image binaire pour tracer des lignes frontières géométriques autour de chaque groupe ou "îlot" de pixels blancs. Le code utilise ensuite la fonction `max()` combinée à `cv2.contourArea` pour ne conserver que la plus grande zone blanche.
* **Effet concret sur l'image :** L'algorithme isole mathématiquement la lésion principale. Visuellement, c'est comme si on dessinait un **repassage au feutre rouge** uniquement autour des bords extérieurs de la tache de maladie la plus importante, en ignorant les petites taches secondaires.

---

## d. Calcul des boîtes de délimitation (`cv2.boundingRect`)

* **Ce que fait le code :** Il calcule les coordonnées du plus petit rectangle horizontal (non incliné) capable d'englober entièrement le contour de la lésion trouvé à l'étape précédente. Il en extrait quatre variables numériques : la position ($x, y$), la largeur ($w$) et la hauteur ($h$).
* **Effet concret sur l'image :** Cela encadre virtuellement la zone malade dans une **boîte rectangulaire**. Cet encadrement permet de mesurer précisément les dimensions maximales de la maladie (sa hauteur et sa largeur sur la feuille).

---

## e. Analyse géométrique et Logique de décision (Rapport de forme et Solidité)

* **Ce que fait le code :** Il effectue des calculs mathématiques sur la boîte et le contour pour mesurer deux critères :
1. **L'Aspect Ratio ($h / w$)** : Le rapport entre la hauteur et la largeur.
2. **La Solidité** : Le ratio entre la surface réelle de la tache et la surface totale du rectangle protecteur.


* **Effet concret sur l'image et condition de retour ($0$ ou $1$) :**
* **Retourne `1` (Validé) :** Si la tache est très allongée (hauteur $\ge 2,5 \times$ la largeur) **ET** qu'elle remplit presque parfaitement son rectangle de capture (bords très droits, solidité $> 70\%$). C'est le cas typique de la **Cercosporiose (Image 1)** qui est bloquée au pixel près par les nervures.
* **Retourne `0` (Rejeté) :** Si la forme est plutôt ronde/ovale (comme la **Rouille, Image 3**) ou si elle a une forme de fuseau pointu (comme l'**Helminthosporiose, Image 2**). Dans le cas du fuseau, les pointes font que la tache laisse beaucoup de vide dans les coins de son rectangle de délimitation, la solidité chute alors sous le seuil requis.