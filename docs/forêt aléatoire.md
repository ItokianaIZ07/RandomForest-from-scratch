# Random Forest Max-Minority

**Le random forest** consiste à combiner plusieurs arbre de décision personalisé (entrainé avec différente données)

```python
    def __init__(self, nombreArbre: int):
        self.n_trees = nombreArbre
        self.tree_depth = 10
        self.__initializeTree()
    
    def __initializeTree(self):
        self.trees = [DecisionTree(self.tree_depth) for i in range(self.n_trees)]

```

**Bagging**: utilisé pour le sous-echantillonnage des lignes du data

```python
    def __bagging(self, X, y, size):
        index_random = np.random.choice(X.index, size=size, replace=True)
        X_random = X.loc[index_random].reset_index(drop=True)
        y_random = y.loc[index_random].reset_index(drop=True)
        return X_random, y_random

    def fit(self, X, y):
        size = len(X)
        for tree in self.trees:
            X_random, y_random = self.__bagging(X, y, size)
            tree.fit(X_random, y_random)        
```

La **Prédiction** consiste à utiliser toutes les prédictions de tous les arbres et de prendre la prediction majoritaire

```python
    def predict(self, X):
        predictions = []
        for tree in self.trees:
            predictions.append(self.__predict_once(X, tree))

        sain_parmiX = predictions.count(0)
        malade_parmiX = predictions.count(1)

        # return predictions #si on veux retourner directement la prédiction sur la liste des données

        return 0 if sain_parmiX > malade_parmiX else 1

    def __predict_once(self, X, tree: DecisionTree):
        return tree.predict(X)
```
