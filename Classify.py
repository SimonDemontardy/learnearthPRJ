"""
Classification supervisée pour prédire la dangerosité 
d'un séisme à partir de ces caractéristiques.
"""

#Importation des modules
import pandas as pd
from sklearn.datasets import load_iris
from sklearn import tree
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

#Importer le fichier
df = pd.read_csv("C:/Users/Lenovo/OneDrive/A.GB5/Introduction de l'IA pour la Biologie/TD/Projet IA/cleaned_processed_data.csv")
data = df[['Magnitude', 'Depth','population_impacted']]
target= df[['Classe']]

#Division du jeu de données en deux ensembles (train et test)
def ensdata(data,label,ratio_traindata):
    #Taille de la matrice de données
    n_samples = len(df.data)
    indices = np.arange(n_samples)
    #Taille du data train 
    taille_data_train=int(n_samples*(ratio_traindata))
    #Mélange les indices
    shuffled_indices = np.random.permutation(n_samples)
    #Indices des listes test et train
    indice_test=shuffled_indices[:taille_data_train]
    indice_train=shuffled_indices[taille_data_train:]
    #Données
    Xtrain = data[:,indice_train] 
    Xtest = data[:,indice_test]   
    ytrain = target[indice_train]  
    ytest = target[indice_test]   
    return Xtrain,Xtest,ytrain,ytest

#Stockage des données dans les variables
Xtrain,Xtest,ytrain,ytest=ensdata(df.data,df.target, 0.7)

#=================== Arbre de décision =====================
clf = tree.DecisionTreeClassifier()
#Entrainement data train
clf = clf.fit(data,target)

#Visualisation de l'arbre de décision
import pydotplus
dot_data = tree.export_graphviz(clf, out_file=None, feature_names=df.feature_names,
class_names=df.target_names, filled=True, rounded=True, special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_pdf("seisme.pdf")

#Prédiction avec le modèle
clf.predict(data[:1, :])
clf.predict_proba(data[:1, :])

#Calcul de l'accuracy
def evalPerf(max_depth, min_samples_leaf,traindata,traintarget,testdata,testtarget):
    """
    Fonction qui construit le modèle et calcule le nombre d'erreur 
    sur l'ensemble de test
    """
    #Création du classificateur d'arbre de décision
    clf = tree.DecisionTreeClassifier( max_depth = max_depth, min_samples_leaf = min_samples_leaf)
    #On fait travailler notre modèle sur nos données d'entrainement 
    clf = clf.fit(traindata,traintarget)
    #On test notre modèle
    ypred = clf.predict(testdata)
    #On calcule le taux d'erreur
    accuracy = accuracy_score(testtarget, ypred)
    return accuracy

accuracy= evalPerf(None, 50, Xtrain,ytrain,Xtest,ytest)
print("Accuracy de l'arbre de décision: ", accuracy)

"""
- Faire varier les parametres max_depth et min_samples_leaf : 
On devrait avoir une meilleure prédiction si max_depth faible et/ou min_samples_leaf élevé
- Est ce pertinent de faire Arbre de décision Régression?
"""
#=================== Random Forest =====================
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Initialiser le classificateur Random Forest
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)  # Vous pouvez ajuster les hyperparamètres

# Entraîner le modèle sur les données d'entraînement
rf_clf.fit(Xtrain, ytrain)

# Prédire sur les données de test
y_pred = rf_clf.predict(Xtest)

# Calculer l'accuracy
accuracy = accuracy_score(ytest, y_pred)
print("Accuracy avec Random Forest: ", accuracy)

#=================== SVM =====================

#Se restreindre à deux caractéristiques (par exemple les deux premiers)
#Et calculer les min et max
"""
Il serait interessant de changer les colonnes choisis
"""
X = data[:, :2]
x_min=min(data[:,0])
x_max=max(data[:,0])
y_min=min(data[:,1])
y_max=max(data[:,1])

#Contruction du classificateur
C=4
gamma=5
h=0.002

from sklearn.svm import LinearSVC
from sklearn.svm import SVC
clf = LinearSVC(C=C,max_iter=20000).fit(X, target)
clf = SVC(kernel='linear', C=C).fit(X, target)
clf = SVC(kernel='rbf', gamma=gamma, max_iter=100000, C=C).fit(X, target)
clf = SVC(kernel='poly', degree=3, max_iter=100000, gamma=gamma,C=C).fit(X, target)
"""
chosir le meilleur modèle
"""

#Figure
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min,y_max, h))
# ou h est le pas du maillage...
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
# ravel permet d’"applatir" le tableau obtenu
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
# Afficher aussi les points d’apprentissage
plt.scatter(X[:, 0], X[:, 1], c=ytrain, cmap=plt.cm.coolwarm)
plt.show()

accuracy = accuracy_score(ytrain, Z) #Z=ypred
print("Accuracy avec SVM:", accuracy)

#=================== Gradient Boosting =====================
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report

# Initialiser le classificateur Gradient Boosting
gb_clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

# Entraîner le modèle sur les données d'entraînement
gb_clf.fit(Xtrain, ytrain)

# Prédire sur les données de test
y_pred = gb_clf.predict(Xtest)

# Calculer l'accuracy
accuracy = accuracy_score(ytest, y_pred)
print("Accuracy avec Gradient Boosting: ", accuracy)

#=================== K-NN =====================
#Entrainement du modèle
def predicteurkNN(Xtrain,Xtest,ytrain,ytest,k):
    #Construction du classifieur
    neigh=KNeighborsClassifier(n_neighbors=k) 
    #Entrainement de notre data train
    neigh.fit(Xtrain.T, ytrain[:, 0]) # .T pour transposer la matrice
    #Prédiction des classes sur le data test
    ypred = neigh.predict(Xtest.T)
    #Calcul du taux de bonnes reponses
    accuracy= accuracy_score(ytest, ypred)
    return accuracy

accuracy=predicteurkNN(Xtrain,Xtest,ytrain,ytest,3)
print("Accuracy de K-NN: ", accuracy)

# #Faire varier le nombre de voisin (n_neighbors)pour ratio different du data train
# liste_ratio_traindata=[0.1,0.3,0.5,0.7,0.9]
# liste_k=[2,3,5,7,8]
# taux_correct=[]

# for ratio in liste_ratio_traindata:
#     liste_temp=[] #liste temporaire qui va contenir les taux de bonne reponse pour un ratio donné mais avec n_neighbors different
#     xtrain,xtest,Ytrain,Ytest=ensdata(df.data,df.target,ratio)
#     for voisin in liste_k:
#         liste_temp.append(predicteurkNN(xtrain,xtest,Ytrain,Ytest,voisin))
#     taux_correct.append(liste_temp)
# print(taux_correct)

# #Visualisation
# plt.figure(figsize=(10, 6))
# for idx, voisin in enumerate(liste_k):
#     plt.plot(liste_ratio_traindata, taux_correct[idx], label=f'n_neighbors = {voisin}', marker='o')

# plt.xlabel("Ratio de données d'entraînement")
# plt.ylabel("Taux de bonne réponse")
# plt.title("Taux de bonne réponse en fonction du ratio d'entraînement pour différents k")
# plt.legend()
# plt.grid(True)
# plt.show()

#=================== Réseaux de neurones =====================
from sklearn.neural_network import MLPClassifier, MLPRegressor

#Construire le perceptron multicouche
#Data train
data_train=Xtrain.copy() #Copie des données
del data_train["target"] #Suppression de l'attribut à prédire
labels_train=Xtrain["target"] #Construction du vecteur de classe

#Data test
data_test=Xtest.copy() #Copie des données
del data_test["target"] #Suppression de l'attribut à prédire
labels_test=Xtest["target"] #Construction du vecteur de classe

#Définir le classifieur et calcul du poids
mlpc=MLPClassifier(hidden_layer_sizes=(100,), max_iter=10000,activation='logistic',learning_rate='adaptive')
mlpc.fit(Xtrain,labels_train) #Entrainement de nos données d'entrainement
#Prédiction sur l'ensemble test
predictions= mlpc.predict(Xtest)
accuracy=accuracy_score(labels_test, predictions)
print("Accuracy pour Réseau de neurones: ", "{:.2f}".format(accuracy))







































