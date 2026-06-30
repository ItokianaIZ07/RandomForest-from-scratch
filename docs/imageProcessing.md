# 🌿 Pipeline Vision par Ordinateur (Rouille sur feuilles)

---

# 1. 📥 Charger et afficher une image

## 📌 OpenCV : lecture

```python
img = cv2.imread("image.jpg")
```

### 🔧 Fonction : `cv2.imread()`

* **Paramètre** : chemin du fichier
* **Rôle** : charge une image en mémoire
* **Format par défaut** : BGR (pas RGB)

---

## 📌 Affichage (Matplotlib)

```python
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()
```

### 🔧 Fonction : `cv2.cvtColor()`

* **Paramètres** :

  * image source
  * conversion (`BGR2RGB`)
* **Rôle** : corrige l’affichage couleur

---

# 2. 🎨 Conversion en HSV

```python
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
```

### 🔧 Fonction : `cv2.cvtColor()`

* **Paramètres** :

  * image BGR
  * `cv2.COLOR_BGR2HSV`
* **Rôle** :

  * sépare couleur (H), saturation (S), luminosité (V)
  * facilite la détection de couleurs

---

# 3. 🎯 Création d’un masque couleur

```python
mask = cv2.inRange(hsv, lower, upper)
```

### 🔧 Fonction : `cv2.inRange()`

* **Paramètres** :

  * image HSV
  * `lower = [H, S, V]`
  * `upper = [H, S, V]`

### 🧠 Rôle :

* garde pixels dans la plage → 255 (blanc)
* autres pixels → 0 (noir)

👉 résultat : image binaire

---

## 📌 Exemple "rouille"

```python
lower = np.array([10, 50, 50])
upper = np.array([30, 255, 255])
```

### 🧠 Signification :

* H (couleur) → orange / brun
* S → intensité minimale
* V → luminosité minimale

---

# 4. 🟠 Extraction des pixels rouille

```python
rouille = cv2.bitwise_and(img, img, mask=mask)
```

### 🔧 Fonction : `cv2.bitwise_and()`

* **Paramètres** :

  * image 1
  * image 2 (identique ici)
  * mask

### 🧠 Rôle :

* conserve uniquement les pixels où mask = 255
* le reste devient noir

---

# 5. ⚫ Inversion du masque

```python
mask_inv = cv2.bitwise_not(mask)
```

### 🔧 Fonction : `cv2.bitwise_not()`

* inverse :

  * 255 → 0
  * 0 → 255

### 🧠 Rôle :

* sélectionne “tout sauf rouille”

---

## 📌 Extraction non-rouille

```python
non_rouille = cv2.bitwise_and(img, img, mask=mask_inv)
```

---

# 6. 🔢 Comptage des pixels

## 📌 Pixels rouille

```python
np.count_nonzero(mask)
```

### 🧠 Rôle :

* compte les pixels blancs (255)

---

## 📌 Pixels totaux

```python
mask.size
```

* nombre total de pixels

---

## 📌 Pourcentage de rouille

```python
taux = np.count_nonzero(mask) / mask.size * 100
```

### 🧠 Rôle :

> mesure la surface malade

---

# 7. 🧭 Conversion en niveaux de gris

```python
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
```

### 🔧 Fonction : `cv2.cvtColor()`

* BGR → Gray

### 🧠 Rôle :

* simplifie l’image
* garde uniquement intensité lumineuse

---

# 8. ⚡ Filtre de Sobel

## 📌 Gradient horizontal

```python
sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
```

### 🔧 Fonction : `cv2.Sobel()`

* **Paramètres :**

  * image grayscale
  * `CV_64F` → float (permet valeurs négatives)
  * `1,0` → dérivée en X
  * `ksize=3` → taille du filtre

### 🧠 Rôle :

* détecte variations gauche → droite

---

## 📌 Gradient vertical

```python
sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
```

### 🧠 Rôle :

* détecte variations haut → bas

---

## 📌 Magnitude (intensité totale)

```python
sobel = cv2.magnitude(sobel_x, sobel_y)
```

### 🔧 Fonction : `cv2.magnitude()`

* calcule :

```text
√(x² + y²)
```

### 🧠 Rôle :

* combine les deux directions
* donne force totale des contours

---

# 9. 📊 Statistiques sur Sobel

## 📌 Moyenne

```python
np.mean(sobel)
```

### 🧠 Signification :

> quantité moyenne de contours

---

## 📌 Variance

```python
np.var(sobel)
```

### 🧠 Signification :

> irrégularité des textures

---

# 🎯 INTERPRÉTATION GLOBALE

| Étape    | But                          |
| -------- | ---------------------------- |
| HSV      | isoler couleur rouille       |
| masque   | sélectionner zones malades   |
| bitwise  | extraire régions             |
| Sobel    | détecter textures / pustules |
| mean     | quantité de texture          |
| variance | irrégularité maladie         |

---

# 🚀 PIPELINE COMPLET (IDÉE FINALE)

```python
img → HSV → mask → extraction → Sobel → stats → features ML
```

---

# 💡 Résultat final possible

Tu peux obtenir :

```python
features = [
    taux_rouille,
    mean_sobel,
    var_sobel
]
```

👉 utilisable directement pour :

* classification sain / malade
* modèle ML simple
* scoring automatique de maladie
