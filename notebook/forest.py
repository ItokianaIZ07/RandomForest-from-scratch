from tree import DecisionTree
import numpy as np
import pandas as pd

class RandomForest:
    def __init__(self, nombreArbre: int):
        self.n_trees = nombreArbre
        self.tree_depth = 10
        self.__initializeTree()
    
    def fit(self, X, y):
        size = len(X)
        for tree in self.trees:
            X_random, y_random = self.__bagging(X, y, size)
            tree.fit(X_random, y_random)

    def __initializeTree(self):
        self.trees = [DecisionTree(self.tree_depth) for i in range(self.n_trees)]

    def __bagging(self, X, y, size):
            index_random = np.random.choice(X.index, size=size, replace=True)
            X_random = X.loc[index_random].reset_index(drop=True)
            y_random = y.loc[index_random].reset_index(drop=True)
            return X_random, y_random

    def predict(self, X):
        predictions = []
        for tree in self.trees:
            predictions.append(self.__predict_once(X, tree))

        predictions = np.array(predictions)
        final_predictions = []

        num_individus = X.shape[0]

        for i in range(num_individus):
            preidction_i_colonne = predictions[:, i]
            
            sain = np.sum(preidction_i_colonne == 0)
            malade = np.sum(preidction_i_colonne == 1)
            
            vote_final = 0 if sain > malade else 1
            final_predictions.append(vote_final)

        return pd.Series(final_predictions)

        # sain_parmiX = predictions.count(0)
        # malade_parmiX = predictions.count(1)


        # return 0 if sain_parmiX > malade_parmiX else 1

    def __predict_once(self, X, tree: DecisionTree):
        return tree.predict(X)