from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, mean_squared_error

from processing.process import DataProcessing
from model.tree import DecisionTree
from model.forest import RandomForest
import pandas as pd
import joblib

dataProcessing = DataProcessing()


df = pd.read_csv("data/data.csv", sep=";")
X = df.drop(columns=["est_malade"])
y = df["est_malade"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20)



# model = DecisionTree(10)
# model.fit(X_train, y_train)
# joblib.dump(model,"./model/modele_arbre_decision.pkl" )

# model = RandomForest(20)
# model.fit(X_train, y_train)

# joblib.dump(model, "./model/modele_foret.pkl")


# img = "./dataset/Saines/Corn_Health (1122).jpg" # eto no atao le image n'uploadenle user
img = "./dataset/rouille-du-mais-du-mais-ou-du-sorgho-puccinia-sorghi-champignon-maladie-pustules-sur-le-mais-ou-la-feuille-de-mais-illinois-etats-unis-2c4x4fp.jpg" # eto no atao le image n'uploadenle user 
# img = "./dataset/Saines/Corn_Health (92).jpg" 

data = dataProcessing.numeriseDonnee(img)
# model = joblib.load("./model/modele_maladie_mais.pkl")
# model = joblib.load("./model/modele_arbre_decision.pkl")
model = joblib.load("./model/modele_foret.pkl")

prediction = model.predict(data).values
y_prediction = model.predict(X_test)

f1 = f1_score(y_test, y_prediction)

print("Score du modèle", f1)
print("Cette plante est malade" if prediction.sum() == 1 else "Cette plante est saine")