"""
Classification supervisée pour prédire la dangerosité 
d'un séisme à partir de ces caractéristiques.
"""

#Importation des modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.tree import plot_tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.tree import export_graphviz
from sklearn.metrics import accuracy_score, classification_report
import pydotplus
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()


###============================ Etude préliminaire ==========================================###
#Importer le fichier
df = pd.read_csv("./final_clustering_results.csv",delimiter='\t')
data = df[['Magnitude', 'Depth','population_impacted']]
target= df['dangerosite_adjusted']

normalized_df = df.copy()

### Normalisation des colonnes "Depth" et "population_impacted" ###
columns_to_normalize = ["Depth", "population_impacted"]
for col in normalized_df: #Parcours toutes les colonnes 
    if col in columns_to_normalize:  # Vérifie si c'est une colonne a normalisé
        normalized_df[col] = normalized_df[col] / normalized_df[col].max()  # Division par la valeur max de la colonne

# Sauvegarder les données normalisées dans un nouveau fichier CSV
normalize_file = "./final_clustering_results_normalize.csv"
normalized_df.to_csv(normalize_file, index=False)
#Récupérer les données normalisées
df_normalize = pd.read_csv("./final_clustering_results_normalize.csv")
data_normalize=df_normalize[['Magnitude', 'Depth','population_impacted']]
target_normalize= df_normalize['dangerosite_adjusted']

### Division du jeu de données en deux ensembles (train et test) ###
def ensdata(data,label,ratio_traindata):
    #Taille de la matrice de données
    n_samples = len(data)
    indices = np.arange(n_samples)
    #Taille du data train 
    taille_data_train=int(n_samples*(ratio_traindata))
    #Mélange les indices
    shuffled_indices = np.random.permutation(n_samples)
    #Indices des listes test et train
    indice_test=shuffled_indices[:taille_data_train]
    indice_train=shuffled_indices[taille_data_train:]
    #Données
    Xtrain = data.iloc[indice_train, :]  # Sélection des lignes de train
    Xtest = data.iloc[indice_test, :]    # Sélection des lignes de test
    ytrain = label.iloc[indice_train]    # Sélection des étiquettes de train
    ytest = label.iloc[indice_test]      # Sélection des étiquettes de test

    return Xtrain,Xtest,ytrain,ytest

#Stockage des données dans les variables
Xtrain,Xtest,ytrain,ytest=ensdata(data,target, 0.7)
N_Xtrain,N_Xtest,N_ytrain,N_ytest=ensdata(data_normalize,target_normalize, 0.7) #Données normalisées

#========================= Arbre de décision ================================

def evalPerf(max_depth, min_samples_leaf,traindata,traintarget,testdata,testtarget,dataset_type="non normalisé",show_confusion_matrix=True):
    """
    Fonction qui construit le modèle et calcule l'accuracy
    et affiche une matrice de confusion globale
    """
    #Création du classificateur d'arbre de décision
    dt_clf = tree.DecisionTreeClassifier(max_depth = max_depth, min_samples_leaf = min_samples_leaf)
    #On fait travailler notre modèle sur nos données d'entrainement 
    dt_clf = dt_clf.fit(traindata,traintarget)
    #On test notre modèle
    ypred = dt_clf.predict(testdata)
    # # Visualisation de l'arbre de décision
    # plt.figure(figsize=(15, 10))
    # plot_tree(dt_clf, 
    #           feature_names=traindata.columns if hasattr(traindata, "columns") else None,
    #           class_names=[str(cls) for cls in set(traintarget)],
    #           filled=True, 
    #           rounded=True, 
    #           fontsize=10)
    # plt.title("Arbre de décision")
    # plt.show()
    
    # Affichage de la matrice de confusion si demandé
    if show_confusion_matrix:
        cm = confusion_matrix(testtarget, ypred) # Calcul de la matrice de confusion
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=[str(cls) for cls in set(testtarget)],
                    yticklabels=[str(cls) for cls in set(testtarget)])
        plt.title(f"Matrice de confusion pour l'Arbre de décision (sur dataset {dataset_type}) \nparametres: max_depth={max_depth} et min_samples_leaf={min_samples_leaf}")
        plt.xlabel('Prédictions')
        plt.ylabel('Réalités')
        plt.savefig(f"./DT_confusion_matrix_DivEns_{dataset_type}.png") 
        #plt.show()
        plt.close()  # Ferme la figure après sauvegarde
    #On calcule le taux d'erreur
    accuracy = accuracy_score(testtarget, ypred)
    return accuracy

def evalPerf_loo(max_depth, min_samples_leaf, data, target,dataset_type="non normalisé",show_confusion_matrix=True):
    """
    Fonction qui effectue un leave-one-out cross-validation,
    calcule l'accuracy moyenne et affiche une matrice de confusion globale
    """
    # Convertir les données en tableaux NumPy si nécessaire
    if hasattr(data, "values"):
        data = data.values
    if hasattr(target, "values"):
        target = target.values

    # Initialisation
    loo = LeaveOneOut()
    true_labels = []
    predicted_labels = []

    # Boucle sur chaque split
    for train_index, test_index in loo.split(data):
        # Séparation des données d'entraînement et de test
        X_train, X_test = data[train_index], data[test_index]
        y_train, y_test = target[train_index], target[test_index]

        # Création et entraînement du modèle
        dt_clf = tree.DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf)
        dt_clf.fit(X_train, y_train)

        # Prédiction
        y_pred = dt_clf.predict(X_test)

        # Stocker les vraies valeurs et les prédictions
        true_labels.append(y_test[0])
        predicted_labels.append(y_pred[0])

    # Calcul de l'accuracy moyenne
    accuracy = accuracy_score(true_labels, predicted_labels)

    # Affichage de la matrice de confusion si demandé
    if show_confusion_matrix:
        cm = confusion_matrix(true_labels, predicted_labels) # Calcul de la matrice de confusion
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=np.unique(target), 
                    yticklabels=np.unique(target))
        plt.title(f"Matrice de confusion avec l'Arbre de décision \n(sur dataset {dataset_type}) en utilisant LOOCV \nparametres: max_depth={max_depth} et min_samples_leaf={min_samples_leaf}")
        plt.xlabel('Prédictions')
        plt.ylabel('Réalités')
        plt.savefig(f"./DT_confusion_matrix_LOOCV_{dataset_type}.png") 
        #plt.show()
        plt.close()  # Ferme la figure après sauvegarde

    return accuracy

accuracy_dt= evalPerf(None, 50, Xtrain,ytrain,Xtest,ytest, dataset_type="non normalisé",show_confusion_matrix=True) #Division donnée + pas de normalisation
accuracy_dt_loo= evalPerf_loo(None, 50, data, target,dataset_type="non normalisé",show_confusion_matrix=True) #Leave one out + pas de normalisation
accuracy_dt_normalize= evalPerf(None, 50, N_Xtrain,N_ytrain,N_Xtest,N_ytest,dataset_type="normalisé",show_confusion_matrix=True) #Division donnée + normalisation
accuracy_dt_loo_normalize= evalPerf_loo(None, 50, data_normalize, target_normalize, dataset_type="normalisé",show_confusion_matrix=True) #Leave one out + normalisation
print("Paramètre : max_depth=None et min_samples_leaf=50 (paramètres par défaut)")
print(f"Accuracy avec l'Arbre de décision : {accuracy_dt:.2f}")
print(f"Accuracy avec l'Arbre de décision en utilisant la méthode leave-one-out : {accuracy_dt_loo:.2f}")
print(f"Accuracy avec l'Arbre de décision sur données normalisés : {accuracy_dt_normalize:.2f}")
print(f"Accuracy avec l'Arbre de décision sur données normalisés en utilisant la méthode leave-one-out : {accuracy_dt_loo_normalize:.2f} \n")

###============= Variation des hyperparamètres ===============###
#======= Max_depth =======
max_depth_values = [1, 3, 7, 10] # Valeurs de max_depth à tester

# Initialisation des listes pour stocker les résultats
accuracies_div = []
accuracies_loo = []
accuracies_div_normalized = []
accuracies_loo_normalized = []

# Calcul des accuracy pour chaque max_depth
for max_depth in max_depth_values:
    # Cas 1 : Division donnée + pas de normalisation
    accuracy_dt = evalPerf(max_depth, 50, Xtrain, ytrain, Xtest, ytest, dataset_type="non normalisé", show_confusion_matrix=False)
    accuracies_div.append(accuracy_dt)
    
    # Cas 2 : Leave-one-out + pas de normalisation
    accuracy_dt_loo = evalPerf_loo(max_depth, 50, data, target, dataset_type="non normalisé", show_confusion_matrix=False)
    accuracies_loo.append(accuracy_dt_loo)
    
    # Cas 3 : Division donnée + normalisation
    accuracy_dt_normalize = evalPerf(max_depth, 50, N_Xtrain, N_ytrain, N_Xtest, N_ytest, dataset_type="normalisé", show_confusion_matrix=False)
    accuracies_div_normalized.append(accuracy_dt_normalize)
    
    # Cas 4 : Leave-one-out + normalisation
    accuracy_dt_loo_normalize = evalPerf_loo(max_depth, 50, data_normalize, target_normalize, dataset_type="normalisé",show_confusion_matrix=False)
    accuracies_loo_normalized.append(accuracy_dt_loo_normalize)

# Tracer les résultats
plt.figure(figsize=(10, 6))
plt.plot(max_depth_values, accuracies_div, marker='o', label="Division donnée (non normalisé)", color="blue")
plt.plot(max_depth_values, accuracies_loo, marker='o', label="LOOCV (non normalisé)", color="red")
plt.plot(max_depth_values, accuracies_div_normalized, marker='o', label="Division donnée (normalisé)", color="green")
plt.plot(max_depth_values, accuracies_loo_normalized, marker='o', label="LOOCV (normalisé)", color="orange")
plt.title("[Arbre de décision] Accuracy en fonction de max_depth")
plt.xlabel("Valeur de max_depth")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.savefig('./DT_max_depth.png')  # Sauvegarde la figure
#plt.show()
plt.close()  # Ferme la figure après sauvegarde

##======= Min_samples_leaf =======
min_samples_leaf_values = [1, 10, 20, 30, 40, 50, 70, 80] # Valeurs de min_samples_leaf à tester

# Initialisation des listes pour stocker les résultats
accuracies_div = []
accuracies_loo = []
accuracies_div_normalized = []
accuracies_loo_normalized = []

# Calcul des accuracy pour chaque min_samples_leaf
for min_samples_leaf in min_samples_leaf_values:
    # Cas 1 : Division donnée + pas de normalisation
    accuracy_dt = evalPerf(None, min_samples_leaf, Xtrain, ytrain, Xtest, ytest, dataset_type="non normalisé", show_confusion_matrix=False)
    accuracies_div.append(accuracy_dt)
    
    # Cas 2 : Leave-one-out + pas de normalisation
    accuracy_dt_loo = evalPerf_loo(None, min_samples_leaf, data, target, dataset_type="non normalisé", show_confusion_matrix=False)
    accuracies_loo.append(accuracy_dt_loo)
    
    # Cas 3 : Division donnée + normalisation
    accuracy_dt_normalize = evalPerf(None, min_samples_leaf, N_Xtrain, N_ytrain, N_Xtest, N_ytest, dataset_type="normalisé", show_confusion_matrix=False)
    accuracies_div_normalized.append(accuracy_dt_normalize)
    
    # Cas 4 : Leave-one-out + normalisation
    accuracy_dt_loo_normalize = evalPerf_loo(None, min_samples_leaf, data_normalize, target_normalize, dataset_type="normalisé",show_confusion_matrix=False)
    accuracies_loo_normalized.append(accuracy_dt_loo_normalize)

# Tracer les résultats
plt.figure(figsize=(10, 6))
plt.plot(min_samples_leaf_values, accuracies_div, marker='o', label="Division donnée (non normalisé)", color="blue")
plt.plot(min_samples_leaf_values, accuracies_loo, marker='o', label="LOOCV (non normalisé)", color="red")
plt.plot(min_samples_leaf_values, accuracies_div_normalized, marker='o', label="Division donnée (normalisé)", color="green")
plt.plot(min_samples_leaf_values, accuracies_loo_normalized, marker='o', label="LOOCV (normalisé)", color="orange")
plt.title("[Arbre de décision] Accuracy en fonction de min_samples_leaf")
plt.xlabel("Valeur de min_samples_leaf")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.savefig('./DT_min_samples_leaf.png')  # Sauvegarde la figure
#plt.show()
plt.close()  # Ferme la figure après sauvegarde

##========== Paramètres optimales : max_depth=7 et min_samples_leaf= 20 (par exple) =======
accuracy_dt= evalPerf(7, 20, Xtrain,ytrain,Xtest,ytest, dataset_type="non normalisé",show_confusion_matrix=False) #Division donnée + pas de normalisation
accuracy_dt_loo= evalPerf_loo(7, 20, data, target,dataset_type="non normalisé",show_confusion_matrix=False) #Leave one out + pas de normalisation
accuracy_dt_normalize= evalPerf(7, 20, N_Xtrain,N_ytrain,N_Xtest,N_ytest,dataset_type="normalisé",show_confusion_matrix=False) #Division donnée + normalisation
accuracy_dt_loo_normalize= evalPerf_loo(7, 20, data_normalize, target_normalize, dataset_type="normalisé",show_confusion_matrix=False) #Leave one out + normalisation
print("Paramètres 'optimales' : max_depth=7 et min_samples_leaf=20 ")
print(f"Accuracy avec l'Arbre de décision : {accuracy_dt:.2f}")
print(f"Accuracy avec l'Arbre de décision en utilisant la méthode leave-one-out : {accuracy_dt_loo:.2f}")
print(f"Accuracy avec l'Arbre de décision sur données normalisés : {accuracy_dt_normalize:.2f}")
print(f"Accuracy avec l'Arbre de décision sur données normalisés en utilisant la méthode leave-one-out : {accuracy_dt_loo_normalize:.2f} \n")

"""
Avec les paramètres par défaut:
- Div ensemble :0.96
- Div ensemble + normalisation : 0.97
- LOOCV : 0.97
- LOOCV + normalisation : 0.97

Faire varier les parametres max_depth et min_samples_leaf : 
--> max_depth : Très bonne accuracy pour tous les cas (on arrive au plateau dès max_depth=3)
--> min_samples_leaf : Pour les cas où l'on utilise LOOCV , l'accuracy est bonne (tjs entre 0.95 et 1)
Quant au cas "Div ensemble + non normalisation" et "Div ensemble + normalisation", il a une bonne accuracy 
si 1<min_samples_leaf<50. Au delà de 50, l'accuracy se dégrade.

Prenons les paramètres les plus "optimales" : max_depth=7 et min_samples_leaf=20 
- Div ensemble :0.97
- Div ensemble + normalisation : 0.96
- LOOCV : 0.99
- LOOCV + normalisation : 0.99
--> En soi presque les mêmes résultats que les paramètres par défaut 
(Les cas avec LOOCV passe de 0.96 à 0.99)

Modele plutot rapide
"""

###=========================== Random Forest ========================###
#Encoder nos étiquettes par des int car ce sont des str
ytrain_encoded = label_encoder.fit_transform(ytrain)
ytest_encoded = label_encoder.transform(ytest)

def RandomForest(n_estimators, max_depth, min_samples_leaf,traindata,traintarget,testdata,testtarget,dataset_type="non normalisé",show_confusion_matrix=True):
    """
    Fonction qui construit le modèle et calcule l'accuracy
    et affiche une matrice de confusion globale
    """
    # Initialiser le classificateur Random Forest
    rf_clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_leaf=min_samples_leaf)
    # Entraîner le modèle sur les données d'entraînement
    rf_clf.fit(traindata, traintarget)
    # Prédire sur les données de test
    y_pred = rf_clf.predict(testdata)
    # Affichage de la matrice de confusion si demandé
    if show_confusion_matrix:
        conf_matrix = confusion_matrix(testtarget, y_pred) # Calculer la matrice de confusion
        disp = ConfusionMatrixDisplay(conf_matrix, display_labels=label_encoder.classes_)
        disp.plot(cmap="Blues", values_format="d")
        plt.title(f"Matrice de confusion avec Random Forest (sur dataset {dataset_type})  \nParametres: n_estimators={n_estimators}, max_depth={max_depth} \net min_samples_leaf={min_samples_leaf}")
        plt.xlabel('Prédictions')
        plt.ylabel('Réalités')
        plt.savefig(f"./RF_confusion_matrix_DivEns_{dataset_type}.png") 
        #plt.show()
        plt.close()  # Ferme la figure après sauvegarde
    # Calculer l'accuracy
    accuracy = accuracy_score(testtarget, y_pred)
    return accuracy


def RandomForest_loo(n_estimators, max_depth, min_samples_leaf, data, target, dataset_type="non normalisé", show_confusion_matrix=True):
    """
    Fonction qui effectue un Leave-One-Out Cross-Validation (LOO),
    calcule l'accuracy moyenne et affiche une matrice de confusion globale
    """
    # Convertir les données en tableaux NumPy si nécessaire
    if hasattr(data, "values"):
        data = data.values
    if hasattr(target, "values"):
        target = target.values

    # Initialisation de Leave-One-Out
    loo = LeaveOneOut()
    true_labels = []
    predicted_labels = []

    # Boucle sur chaque split
    for train_index, test_index in loo.split(data):
        # Séparation des données d'entraînement et de test
        X_train, X_test = data[train_index], data[test_index]
        y_train, y_test = target[train_index], target[test_index]
        # Création et entraînement du modèle Random Forest
        rf_clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_leaf=min_samples_leaf)
        rf_clf.fit(X_train, y_train)
        # Prédiction
        y_pred = rf_clf.predict(X_test)
        # Stocker les vraies valeurs et les prédictions
        true_labels.append(y_test[0])
        predicted_labels.append(y_pred[0])

    # Affichage de la matrice de confusion si demandé
    if show_confusion_matrix:
        # Calculer la matrice de confusion
        conf_matrix = confusion_matrix(true_labels, predicted_labels) 
        disp = ConfusionMatrixDisplay(conf_matrix, display_labels=label_encoder.classes_)
        disp.plot(cmap="Blues", values_format="d")
        
        # Titre de la matrice de confusion
        plt.title(f"Matrice de confusion avec Random Forest (sur dataset {dataset_type}) en utilisant LOOCV \n"
                  f" Parametres: n_estimators={n_estimators}, max_depth={max_depth} \net min_samples_leaf={min_samples_leaf}")
        plt.xlabel('Prédictions')
        plt.ylabel('Réalités')
        plt.savefig(f"./RF_confusion_matrix_LOOCV_{dataset_type}.png") 
        #plt.show()
        plt.close()  # Ferme la figure après sauvegarde

    # Calcul de l'accuracy moyenne
    accuracy = accuracy_score(true_labels, predicted_labels)
    return accuracy


accuracy_rf= RandomForest(100, None, 1, Xtrain,ytrain,Xtest,ytest, dataset_type="non normalisé",show_confusion_matrix=True) #Division donnée + pas de normalisation
accuracy_rf_loo= RandomForest_loo(100, None, 1, data, target,dataset_type="non normalisé",show_confusion_matrix=True) #Leave one out + pas de normalisation
accuracy_rf_normalize= RandomForest(100, None, 1, N_Xtrain,N_ytrain,N_Xtest,N_ytest,dataset_type="normalisé",show_confusion_matrix=True) #Division donnée + normalisation
accuracy_rf_loo_normalize= RandomForest_loo(100, None, 1, data_normalize, target_normalize, dataset_type="normalisé",show_confusion_matrix=True) #Leave one out + normalisation
print("Paramètres : n_estimators =100, max_depth=None et min_samples_leaf=1 (paramètres par défaut)")
print(f"Accuracy avec Random Forest : {accuracy_rf:.2f}")
print(f"Accuracy avec Random Forest en utilisant la méthode leave-one-out : {accuracy_rf_loo:.2f}")
print(f"Accuracy avec Random Forest sur données normalisées : {accuracy_rf_normalize:.2f}")
print(f"Accuracy avec Random Forest sur données normalisées en utilisant la méthode leave-one-out : {accuracy_rf_loo_normalize:.2f} \n")

###============= Variation des hyperparamètres ===============###
##======= n_estimators =======
n_estimators_values = [20,50,100,200,500] # Valeurs de n_estimators à tester

# Initialisation des listes pour stocker les résultats
accuracies_div = []
accuracies_loo = []
accuracies_div_normalized = []
accuracies_loo_normalized = []

# Calcul des accuracy pour chaque n_estimators
for n_estimators in n_estimators_values:
    # Cas 1 : Division donnée + pas de normalisation
    accuracy_rf = RandomForest(n_estimators,None, 50, Xtrain, ytrain, Xtest, ytest, dataset_type="non normalisé", show_confusion_matrix=False)
    accuracies_div.append(accuracy_rf)
    
    # Cas 2 : Leave-one-out + pas de normalisation
    accuracy_rf_loo = RandomForest_loo(n_estimators, None, 50, data, target, dataset_type="non normalisé", show_confusion_matrix=False)
    accuracies_loo.append(accuracy_rf_loo)
    
    # Cas 3 : Division donnée + normalisation
    accuracy_rf_normalize = RandomForest(n_estimators,None, 50, N_Xtrain, N_ytrain, N_Xtest, N_ytest, dataset_type="normalisé", show_confusion_matrix=False)
    accuracies_div_normalized.append(accuracy_rf_normalize)
    
    # Cas 4 : Leave-one-out + normalisation
    accuracy_rf_loo_normalize = RandomForest_loo(n_estimators,None, 50, data_normalize, target_normalize, dataset_type="normalisé",show_confusion_matrix=False)
    accuracies_loo_normalized.append(accuracy_rf_loo_normalize)

# Tracer les résultats
plt.figure(figsize=(10, 6))
plt.plot(n_estimators_values, accuracies_div, marker='o', label="Division donnée (non normalisé)", color="blue")
plt.plot(n_estimators_values, accuracies_loo, marker='o', label="LOOCV (non normalisé)", color="red")
plt.plot(n_estimators_values, accuracies_div_normalized, marker='o', label="Division donnée (normalisé)", color="green")
plt.plot(n_estimators_values, accuracies_loo_normalized, marker='o', label="LOOCV (normalisé)", color="orange")
plt.title("[Random Forest] Accuracy en fonction de n_estimators")
plt.xlabel("Valeur de n_estimators")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.savefig('./RF_n_estimators.png')  # Sauvegarde la figure
#plt.show()
plt.close()  # Ferme la figure après sauvegarde

##======= Max_depth =======
max_depth_values = [1, 3, 7, 10] # Valeurs de max_depth à tester

# Initialisation des listes pour stocker les résultats
accuracies_div = []
accuracies_loo = []
accuracies_div_normalized = []
accuracies_loo_normalized = []

# Calcul des accuracy pour chaque max_depth
for max_depth in max_depth_values:
    # Cas 1 : Division donnée + pas de normalisation
    accuracy_rf = RandomForest(100,max_depth, 50, Xtrain, ytrain, Xtest, ytest, dataset_type="non normalisé", show_confusion_matrix=False)
    accuracies_div.append(accuracy_rf)
    
    # Cas 2 : Leave-one-out + pas de normalisation
    accuracy_rf_loo = RandomForest_loo(100,max_depth, 50, data, target, dataset_type="non normalisé", show_confusion_matrix=False)
    accuracies_loo.append(accuracy_rf_loo)
    
    # Cas 3 : Division donnée + normalisation
    accuracy_rf_normalize = RandomForest(100,max_depth, 50, N_Xtrain, N_ytrain, N_Xtest, N_ytest, dataset_type="normalisé", show_confusion_matrix=False)
    accuracies_div_normalized.append(accuracy_rf_normalize)
    
    # Cas 4 : Leave-one-out + normalisation
    accuracy_rf_loo_normalize = RandomForest_loo(100,max_depth, 50, data_normalize, target_normalize, dataset_type="normalisé",show_confusion_matrix=False)
    accuracies_loo_normalized.append(accuracy_rf_loo_normalize)

# Tracer les résultats
plt.figure(figsize=(10, 6))
plt.plot(max_depth_values, accuracies_div, marker='o', label="Division donnée (non normalisé)", color="blue")
plt.plot(max_depth_values, accuracies_loo, marker='o', label="LOOCV (non normalisé)", color="red")
plt.plot(max_depth_values, accuracies_div_normalized, marker='o', label="Division donnée (normalisé)", color="green")
plt.plot(max_depth_values, accuracies_loo_normalized, marker='o', label="LOOCV (normalisé)", color="orange")
plt.title("[Random Forest] Accuracy en fonction de max_depth")
plt.xlabel("Valeur de max_depth")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.savefig('./RF_max_depth.png')  # Sauvegarde la figure
#plt.show()
plt.close()  # Ferme la figure après sauvegarde

##======= Min_samples_leaf =======
min_samples_leaf_values = [1, 10, 20, 30, 40, 50, 70, 80] # Valeurs de min_samples_leaf à tester

# Initialisation des listes pour stocker les résultats
accuracies_div = []
accuracies_loo = []
accuracies_div_normalized = []
accuracies_loo_normalized = []

# Calcul des accuracy pour chaque min_samples_leaf
for min_samples_leaf in min_samples_leaf_values:
    # Cas 1 : Division donnée + pas de normalisation
    accuracy_rf = RandomForest(100,None, min_samples_leaf, Xtrain, ytrain, Xtest, ytest, dataset_type="non normalisé", show_confusion_matrix=False)
    accuracies_div.append(accuracy_rf)
    
    # Cas 2 : Leave-one-out + pas de normalisation
    accuracy_rf_loo = RandomForest_loo(100,None, min_samples_leaf, data, target, dataset_type="non normalisé", show_confusion_matrix=False)
    accuracies_loo.append(accuracy_rf_loo)
    
    # Cas 3 : Division donnée + normalisation
    accuracy_rf_normalize = RandomForest(100,None, min_samples_leaf, N_Xtrain, N_ytrain, N_Xtest, N_ytest, dataset_type="normalisé", show_confusion_matrix=False)
    accuracies_div_normalized.append(accuracy_rf_normalize)
    
    # Cas 4 : Leave-one-out + normalisation
    accuracy_rf_loo_normalize = RandomForest_loo(100,None, min_samples_leaf, data_normalize, target_normalize, dataset_type="normalisé",show_confusion_matrix=False)
    accuracies_loo_normalized.append(accuracy_rf_loo_normalize)

# Tracer les résultats
plt.figure(figsize=(10, 6))
plt.plot(min_samples_leaf_values, accuracies_div, marker='o', label="Division donnée (non normalisé)", color="blue")
plt.plot(min_samples_leaf_values, accuracies_loo, marker='o', label="LOOCV (non normalisé)", color="red")
plt.plot(min_samples_leaf_values, accuracies_div_normalized, marker='o', label="Division donnée (normalisé)", color="green")
plt.plot(min_samples_leaf_values, accuracies_loo_normalized, marker='o', label="LOOCV (normalisé)", color="orange")
plt.title("[Random Forest] Accuracy en fonction de min_samples_leaf")
plt.xlabel("Valeur de min_samples_leaf")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.savefig('./RF_min_samples_leaf.png')  # Sauvegarde la figure
#plt.show()
plt.close()  # Ferme la figure après sauvegarde

##============== Paramètres optimales: n_estimators=100, max_depth=7, min_samples_leaf=1  =============##
accuracy_rf= RandomForest(100, 7, 1, Xtrain,ytrain,Xtest,ytest, dataset_type="non normalisé",show_confusion_matrix=True) #Division donnée + pas de normalisation
accuracy_rf_loo= RandomForest_loo(100, 7, 1, data, target,dataset_type="non normalisé",show_confusion_matrix=True) #Leave one out + pas de normalisation
accuracy_rf_normalize= RandomForest(100, 7, 1, N_Xtrain,N_ytrain,N_Xtest,N_ytest,dataset_type="normalisé",show_confusion_matrix=True) #Division donnée + normalisation
accuracy_rf_loo_normalize= RandomForest_loo(100, 7, 1, data_normalize, target_normalize, dataset_type="normalisé",show_confusion_matrix=True) #Leave one out + normalisation
print("Paramètres : n_estimators=100, max_depth=7,min_samples_leaf=1 (paramètres 'optimales')")
print(f"Accuracy avec Random Forest : {accuracy_rf:.2f}")
print(f"Accuracy avec Random Forest en utilisant la méthode leave-one-out : {accuracy_rf_loo:.2f}")
print(f"Accuracy avec Random Forest sur données normalisées : {accuracy_rf_normalize:.2f}")
print(f"Accuracy avec Random Forest sur données normalisées en utilisant la méthode leave-one-out : {accuracy_rf_loo_normalize:.2f} \n")
  
"""
Avec les paramètres par défaut:
- Div ensemble : 0.98
- Div ensemble + normalisation : 0.99 
- LOOCV : 0.97
- LOOCV + normalisation : 0.97

Faire varier les hyperparamètres:
--> n_estimators: Accuracy pas terrible pour "Div ensemble + pas de normalisation" 
et "Div ensemble + normalisation" (varie entre 0.65 et 0.75)
Quant à ce fais avec LOOCV, ils ont une très bonne accuracy
--> max_depth: la performance du modèle augemente considérablement lorsque max_depth >=3 
Mais les modèles fais par "Div ensemble" (normalisé ou non) ont une accuracy limitée (entre 0.65 et 0.75)
Quant aux modèles fais avec LOOCR restent les meilleurs
--> min_samples_leaf: Diminution globale de l'accuracy lorsque min_samples_leaf>20
Les modèles fais avec "Div ensemble" passent de 0.96 (min_samples_leaf=1) à moins de à 0.70 (min_samples_leaf=80)

Avec les paramètres 'optimales':
- Div ensemble : 0.99
- Div ensemble + normalisation : 1.00
- LOOCV : 1.00
- LOOCV + normalisation : 1.00
--> 100% de réussite

"""


###============================== Gradient Boosting ===================================

def GradBoosting(learning_rate, n_estimators, max_depth, min_samples_leaf,traindata,traintarget,testdata,testtarget,dataset_type="non normalisé",show_confusion_matrix=True):
    """
    Fonction qui construit le modèle et calcule l'accuracy
    et affiche une matrice de confusion globale
    """
    # Initialiser le classificateur Gradient Boosting
    gb_clf = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, min_samples_leaf=min_samples_leaf)
    # Entraîner le modèle sur les données d'entraînement
    gb_clf.fit(traindata, traintarget)
    # Prédire sur les données de test
    y_pred = gb_clf.predict(testdata)

    # Affichage de la matrice de confusion si demandé
    if show_confusion_matrix:
        #Visualisation de la matrice de confusion
        cm = confusion_matrix(ytest, y_pred) #Calcul matrice de confusion
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(ytest), yticklabels=np.unique(ytest))
        plt.title(f"Matrice de confusion avec Gradient Boosting (sur dataset {dataset_type}) \n"
                  f"Parametres: learning_rate={learning_rate}, n_estimators={n_estimators}, \nmax_depth={max_depth} et min_samples_leaf={min_samples_leaf}")
        plt.xlabel('Prédictions')
        plt.ylabel('Réalités')
        plt.savefig(f"./GB_confusion_matrix_DivEns_{dataset_type}.png") 
        #plt.show()
        plt.close()  # Ferme la figure après sauvegarde
    # Calculer l'accuracy
    accuracy = accuracy_score(testtarget, y_pred)
    return accuracy

def GradBoosting_loo(learning_rate, n_estimators, max_depth, min_samples_leaf, data, target, dataset_type="non normalisé", show_confusion_matrix=True):
    """
    Fonction qui effectue un leave-one-out cross-validation,
    calcule l'accuracy moyenne et affiche une matrice de confusion globale.
    """
    # Convertir les données en tableaux NumPy si nécessaire
    if hasattr(data, "values"):
        data = data.values
    if hasattr(target, "values"):
        target = target.values

    # Initialisation
    loo = LeaveOneOut()
    true_labels = []
    predicted_labels = []

    # Boucle sur chaque split
    for train_index, test_index in loo.split(data):
        # Séparation des données d'entraînement et de test
        X_train, X_test = data[train_index], data[test_index]
        y_train, y_test = target[train_index], target[test_index]

        # Création et entraînement du modèle Gradient Boosting
        gb_clf = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, min_samples_leaf=min_samples_leaf)
        gb_clf.fit(X_train, y_train)

        # Prédiction
        y_pred = gb_clf.predict(X_test)

        # Stocker les vraies valeurs et les prédictions
        true_labels.append(y_test[0])
        predicted_labels.append(y_pred[0])

    # Calcul de la matrice de confusion et affichage si demandé
    if show_confusion_matrix:
        cm = confusion_matrix(true_labels, predicted_labels)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(target), yticklabels=np.unique(target))
        plt.title(f"Matrice de confusion avec Gradient Boosting (sur dataset {dataset_type}) en utilisant LOOCV \n"
                  f"Parametres: learning_rate={learning_rate}, n_estimators={n_estimators}, \nmax_depth={max_depth} et min_samples_leaf={min_samples_leaf}")
        plt.xlabel('Prédictions')
        plt.ylabel('Réalités')
        plt.savefig(f"./GB_confusion_matrix_LOOCV_{dataset_type}.png") 
        #plt.show()
        plt.close()  # Ferme la figure après sauvegarde

    # Calcul de l'accuracy moyenne
    accuracy = accuracy_score(true_labels, predicted_labels)
    return accuracy


accuracy_gb= GradBoosting(0.1, 100, None, 1, Xtrain,ytrain,Xtest,ytest, dataset_type="non normalisé",show_confusion_matrix=True) #Division donnée + pas de normalisation
accuracy_gb_loo= GradBoosting_loo(0.1,100, None, 1, data, target,dataset_type="non normalisé",show_confusion_matrix=True) #Leave one out + pas de normalisation
accuracy_gb_normalize= GradBoosting(0.1, 100, None, 1, N_Xtrain,N_ytrain,N_Xtest,N_ytest,dataset_type="normalisé",show_confusion_matrix=True) #Division donnée + normalisation
accuracy_gb_loo_normalize= GradBoosting_loo(0.1, 100, None, 1, data_normalize, target_normalize, dataset_type="normalisé",show_confusion_matrix=True) #Leave one out + normalisation
print("Paramètres : learning_rate=0.1, n_estimators =100, max_depth=None et min_samples_leaf=1 (paramètres par défaut)")
print(f"Accuracy avec Gradient Boosting : {accuracy_gb:.2f}")
print(f"Accuracy avec Gradient Boosting en utilisant la méthode leave-one-out : {accuracy_gb_loo:.2f}")
print(f"Accuracy avec Gradient Boosting sur données normalisées : {accuracy_gb_normalize:.2f}")
print(f"Accuracy avec Gradient Boosting sur données normalisées en utilisant la méthode leave-one-out : {accuracy_gb_loo_normalize:.2f} \n")

###============= Variation des hyperparamètres ===============###
##======= learning_rate =======
learning_rate_values = [0.01, 0.05, 0.1, 0.2, 0.3] # Valeurs de learning_rate à tester

# Initialisation des listes pour stocker les résultats
accuracies_div = []
accuracies_loo = []
accuracies_div_normalized = []
accuracies_loo_normalized = []

# Calcul des accuracy pour chaque learning_rate
for learning_rate in learning_rate_values:
    # Cas 1 : Division donnée + pas de normalisation
    accuracy_gb = GradBoosting(learning_rate, 100 ,3, 1, Xtrain, ytrain, Xtest, ytest, dataset_type="non normalisé", show_confusion_matrix=False)
    accuracies_div.append(accuracy_gb)
    
    # Cas 2 : Leave-one-out + pas de normalisation
    accuracy_gb_loo = GradBoosting_loo(learning_rate, 100 ,3, 1, data, target, dataset_type="non normalisé", show_confusion_matrix=False)
    accuracies_loo.append(accuracy_gb_loo)
    
    # Cas 3 : Division donnée + normalisation
    accuracy_gb_normalize = GradBoosting(learning_rate, 100 ,3, 1, N_Xtrain, N_ytrain, N_Xtest, N_ytest, dataset_type="normalisé", show_confusion_matrix=False)
    accuracies_div_normalized.append(accuracy_gb_normalize)
    
    # Cas 4 : Leave-one-out + normalisation
    accuracy_gb_loo_normalize = GradBoosting_loo(learning_rate, 100 ,3, 1, data_normalize, target_normalize, dataset_type="normalisé",show_confusion_matrix=False)
    accuracies_loo_normalized.append(accuracy_gb_loo_normalize)

# Tracer les résultats
plt.figure(figsize=(10, 6))
plt.plot(learning_rate_values, accuracies_div, marker='o', label="Division donnée (non normalisé)", color="blue")
plt.plot(learning_rate_values, accuracies_loo, marker='o', label="LOOCV (non normalisé)", color="red")
plt.plot(learning_rate_values, accuracies_div_normalized, marker='o', label="Division donnée (normalisé)", color="green")
plt.plot(learning_rate_values, accuracies_loo_normalized, marker='o', label="LOOCV (normalisé)", color="orange")
plt.title("[Gradient Boosting] Accuracy en fonction de learning_rate")
plt.xlabel("Valeur de learning_rate")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.savefig('./GB_learning_rate.png')  # Sauvegarde la figure
#plt.show()
plt.close()  # Ferme la figure après sauvegarde

##======= n_estimators =======
n_estimators_values = [20,50,100,200,500] # Valeurs de n_estimators à tester

# Initialisation des listes pour stocker les résultats
accuracies_div = []
accuracies_loo = []
accuracies_div_normalized = []
accuracies_loo_normalized = []

# Calcul des accuracy pour chaque n_estimators
for n_estimators in n_estimators_values:
    # Cas 1 : Division donnée + pas de normalisation
    accuracy_gb = GradBoosting(0.1, n_estimators ,3, 1, Xtrain, ytrain, Xtest, ytest, dataset_type="non normalisé", show_confusion_matrix=False)
    accuracies_div.append(accuracy_gb)
    
    # Cas 2 : Leave-one-out + pas de normalisation
    accuracy_gb_loo = GradBoosting_loo(0.1, n_estimators,3, 1, data, target, dataset_type="non normalisé", show_confusion_matrix=False)
    accuracies_loo.append(accuracy_gb_loo)
    
    # Cas 3 : Division donnée + normalisation
    accuracy_gb_normalize = GradBoosting(0.1, n_estimators ,3, 1, N_Xtrain, N_ytrain, N_Xtest, N_ytest, dataset_type="normalisé", show_confusion_matrix=False)
    accuracies_div_normalized.append(accuracy_gb_normalize)
    
    # Cas 4 : Leave-one-out + normalisation
    accuracy_gb_loo_normalize = GradBoosting_loo(0.1, n_estimators ,3, 1, data_normalize, target_normalize, dataset_type="normalisé",show_confusion_matrix=False)
    accuracies_loo_normalized.append(accuracy_gb_loo_normalize)

# Tracer les résultats
plt.figure(figsize=(10, 6))
plt.plot(n_estimators_values, accuracies_div, marker='o', label="Division donnée (non normalisé)", color="blue")
plt.plot(n_estimators_values, accuracies_loo, marker='o', label="LOOCV (non normalisé)", color="red")
plt.plot(n_estimators_values, accuracies_div_normalized, marker='o', label="Division donnée (normalisé)", color="green")
plt.plot(n_estimators_values, accuracies_loo_normalized, marker='o', label="LOOCV (normalisé)", color="orange")
plt.title("[Gradient Boosting] Accuracy en fonction de n_estimators")
plt.xlabel("Valeur de n_estimators")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.savefig('./GB_n_estimators.png')  # Sauvegarde la figure
#plt.show()
plt.close()  # Ferme la figure après sauvegarde

##======= Max_depth =======
max_depth_values = [1, 3, 7, 10] # Valeurs de max_depth à tester

# Initialisation des listes pour stocker les résultats
accuracies_div = []
accuracies_loo = []
accuracies_div_normalized = []
accuracies_loo_normalized = []

# Calcul des accuracy pour chaque max_depth
for max_depth in max_depth_values:
    # Cas 1 : Division donnée + pas de normalisation
    accuracy_gb = GradBoosting(0.1, 100 ,max_depth, 1, Xtrain, ytrain, Xtest, ytest, dataset_type="non normalisé", show_confusion_matrix=False)
    accuracies_div.append(accuracy_gb)
    
    # Cas 2 : Leave-one-out + pas de normalisation
    accuracy_gb_loo = GradBoosting_loo(0.1, 100 ,max_depth, 1, data, target, dataset_type="non normalisé", show_confusion_matrix=False)
    accuracies_loo.append(accuracy_gb_loo)
    
    # Cas 3 : Division donnée + normalisation
    accuracy_gb_normalize = GradBoosting(0.1, 100 ,max_depth, 1, N_Xtrain, N_ytrain, N_Xtest, N_ytest, dataset_type="normalisé", show_confusion_matrix=False)
    accuracies_div_normalized.append(accuracy_gb_normalize)
    
    # Cas 4 : Leave-one-out + normalisation
    accuracy_gb_loo_normalize = GradBoosting_loo(0.1, 100 ,max_depth, 1, data_normalize, target_normalize, dataset_type="normalisé",show_confusion_matrix=False)
    accuracies_loo_normalized.append(accuracy_gb_loo_normalize)


# Tracer les résultats
plt.figure(figsize=(10, 6))
plt.plot(max_depth_values, accuracies_div, marker='o', label="Division donnée (non normalisé)", color="blue")
plt.plot(max_depth_values, accuracies_loo, marker='o', label="LOOCV (non normalisé)", color="red")
plt.plot(max_depth_values, accuracies_div_normalized, marker='o', label="Division donnée (normalisé)", color="green")
plt.plot(max_depth_values, accuracies_loo_normalized, marker='o', label="LOOCV (normalisé)", color="orange")
plt.title("[Gradient Boosting] Accuracy en fonction de max_depth")
plt.xlabel("Valeur de max_depth")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.savefig('./GB_max_depth.png')  # Sauvegarde la figure
#plt.show()
plt.close()  # Ferme la figure après sauvegarde

##======= Min_samples_leaf =======
min_samples_leaf_values = [1, 10, 20, 30, 40, 50, 70, 80] # Valeurs de min_samples_leaf à tester

# Initialisation des listes pour stocker les résultats
accuracies_div = []
accuracies_loo = []
accuracies_div_normalized = []
accuracies_loo_normalized = []

# Calcul des accuracy pour chaque min_samples_leaf
for min_samples_leaf in min_samples_leaf_values:
    # Cas 1 : Division donnée + pas de normalisation
    accuracy_gb = GradBoosting(0.1, 100 ,3, min_samples_leaf, Xtrain, ytrain, Xtest, ytest, dataset_type="non normalisé", show_confusion_matrix=False)
    accuracies_div.append(accuracy_gb)
    
    # Cas 2 : Leave-one-out + pas de normalisation
    accuracy_gb_loo = GradBoosting_loo(0.1, 100 ,3, min_samples_leaf, data, target, dataset_type="non normalisé", show_confusion_matrix=False)
    accuracies_loo.append(accuracy_gb_loo)
    
    # Cas 3 : Division donnée + normalisation
    accuracy_gb_normalize = GradBoosting(0.1, 100 ,3, min_samples_leaf, N_Xtrain, N_ytrain, N_Xtest, N_ytest, dataset_type="normalisé", show_confusion_matrix=False)
    accuracies_div_normalized.append(accuracy_gb_normalize)
    
    # Cas 4 : Leave-one-out + normalisation
    accuracy_gb_loo_normalize = GradBoosting_loo(0.1, 100 ,3, min_samples_leaf, data_normalize, target_normalize, dataset_type="normalisé",show_confusion_matrix=False)
    accuracies_loo_normalized.append(accuracy_gb_loo_normalize)

# Tracer les résultats
plt.figure(figsize=(10, 6))
plt.plot(min_samples_leaf_values, accuracies_div, marker='o', label="Division donnée (non normalisé)", color="blue")
plt.plot(min_samples_leaf_values, accuracies_loo, marker='o', label="LOOCV (non normalisé)", color="red")
plt.plot(min_samples_leaf_values, accuracies_div_normalized, marker='o', label="Division donnée (normalisé)", color="green")
plt.plot(min_samples_leaf_values, accuracies_loo_normalized, marker='o', label="LOOCV (normalisé)", color="orange")
plt.title("[Gradient Boosting] Accuracy en fonction de min_samples_leaf")
plt.xlabel("Valeur de min_samples_leaf")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.savefig('./GB_min_samples_leaf.png')  # Sauvegarde la figure
#plt.show()
plt.close()  # Ferme la figure après sauvegarde

##============== Paramètres optimales: learning_rate= 0.1, n_estimators= 100, max_depth=3, min_samples_leaf=10 =============##
accuracy_gb= GradBoosting(0.1, 100, 3, 10, Xtrain,ytrain,Xtest,ytest, dataset_type="non normalisé",show_confusion_matrix=True) #Division donnée + pas de normalisation
accuracy_gb_loo= GradBoosting_loo(0.1, 100, 3, 10, data, target,dataset_type="non normalisé",show_confusion_matrix=True) #Leave one out + pas de normalisation
accuracy_gb_normalize= GradBoosting(0.1, 100, 3, 10,  1, N_Xtrain,N_ytrain,N_Xtest,N_ytest,dataset_type="normalisé",show_confusion_matrix=True) #Division donnée + normalisation
accuracy_gb_loo_normalize= GradBoosting_loo(0.1, 100, 3, 10, data_normalize, target_normalize, dataset_type="normalisé",show_confusion_matrix=True) #Leave one out + normalisation
print("Paramètres : learning_rate= 0.1, n_estimators= 100, max_depth=3, min_samples_leaf=10 (paramètres 'optimales')")
print(f"Accuracy avec Gradient Boosting : {accuracy_gb:.2f}")
print(f"Accuracy avec Gradient Boosting en utilisant la méthode leave-one-out : {accuracy_gb_loo:.2f}")
print(f"Accuracy avec Gradient Boosting sur données normalisées : {accuracy_gb_normalize:.2f}")
print(f"Accuracy avec Gradient Boosting sur données normalisées en utilisant la méthode leave-one-out : {accuracy_gb_loo_normalize:.2f} \n")

"""
Avec les paramètres par défaut:
- Div ensemble :0.98
- Div ensemble + normalisation : 0.99
- LOOCV : 1.00
- LOOCV + normalisation : 1.00

Faire varier les hyperparamètres:
--> learning_rate: Très bon accuracy pour les 4 cas (sup à 0.98)
Les meilleures modèles sont ceux faites avec LOOCV. 
L'accuracy est le meilleur lorsque learning_rate
--> n_estimators : Très bonnes accuracy pour les 4 cas (sup à 0.99)
Les meilleurs restent ceux fais avec LOOCV.
Meilleur performance lorsque n_estimators=100
--> max_depth: Bonne accuracy (sup à 0.99) pour les 4 cas
Meilleur accuracy lorsque la variable vaut 3
--> min_samples_leaf: Diminution de l'accuracy por les 4 cas lorsque min_samples_leaf >50
Mais l'accuracy reste correcte (sup à 0.9)
Les meilleurs modèes restent ceux faite avec LOOCV et lorsque la variable= 10

Avec les paramètres 'optimales':
- Div ensemble :0.99
- Div ensemble + normalisation : 1.00
- LOOCV : 1.00
- LOOCV + normalisation : 1.00
"""

#=================== K-NN =====================

def KNN(n_neighbors,traindata,traintarget,testdata,testtarget,dataset_type="non normalisé",show_confusion_matrix=True):
    """
    Fonction qui construit le modèle et calcule l'accuracy
    et affiche une matrice de confusion globale
    """
    # Encodage des étiquettes si elles sont de type string
    if isinstance(traintarget.iloc[0], str) or isinstance(testtarget.iloc[0], str):
        label_encoder = LabelEncoder()
        traintarget = label_encoder.fit_transform(traintarget)
        testtarget = label_encoder.transform(testtarget)

    # Initialiser le classificateur kNN
    knn_clf = KNeighborsClassifier(n_neighbors=n_neighbors) 
    # Entraîner le modèle sur les données d'entraînement
    knn_clf.fit(traindata, traintarget)
    # Prédire sur les données de test
    y_pred = knn_clf.predict(testdata)

    # Affichage de la matrice de confusion si demandé
    if show_confusion_matrix:
        #Visualisation de la matrice de confusion
        cm = confusion_matrix(ytest_encoded, y_pred) #Calcul matrice de confusion
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(testtarget), yticklabels=np.unique(testtarget))
        plt.title(f"Matrice de confusion avec k-NN (sur dataset {dataset_type}) \n"
                  f"Parametre: n_neighbors={n_neighbors}")
        plt.xlabel('Prédictions')
        plt.ylabel('Réalités')
        plt.savefig(f"./kNN_confusion_matrix_DivEns_{dataset_type}.png") 
        #plt.show()
        plt.close()  # Ferme la figure après sauvegarde
    # Calculer l'accuracy
    accuracy = accuracy_score(testtarget, y_pred)
    return accuracy

def KNN_loo(n_neighbors, data, target, dataset_type="non normalisé", show_confusion_matrix=True):
    """
    Fonction qui effectue une validation croisée leave-one-out (LOO) avec k-NN,
    calcule l'accuracy moyenne et affiche une matrice de confusion globale.
    """
    # Encodage des étiquettes si elles sont de type string
    if isinstance(target.iloc[0], str):
        label_encoder = LabelEncoder()
        target = label_encoder.fit_transform(target)
    else:
        label_encoder = None  # Pas d'encodage nécessaire
    
    # Convertir les données en tableaux NumPy si nécessaire
    if hasattr(data, "values"):
        data = data.values
    if hasattr(target, "values"):
        target = target.values

    # Initialisation pour Leave-One-Out
    loo = LeaveOneOut()
    true_labels = []
    predicted_labels = []

    # Boucle sur chaque split
    for train_index, test_index in loo.split(data):
        # Séparer les données d'entraînement et de test
        X_train, X_test = data[train_index], data[test_index]
        y_train, y_test = target[train_index], target[test_index]

        # Initialiser et entraîner le classificateur kNN
        knn_clf = KNeighborsClassifier(n_neighbors=n_neighbors)
        knn_clf.fit(X_train, y_train)

        # Prédire la classe pour le test
        y_pred = knn_clf.predict(X_test)

        # Stocker les vraies valeurs et les prédictions
        true_labels.append(y_test[0])
        predicted_labels.append(y_pred[0])

    # Calcul et affichage de la matrice de confusion si demandé
    if show_confusion_matrix:
        cm = confusion_matrix(true_labels, predicted_labels)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=label_encoder.classes_ if label_encoder else np.unique(target), 
                    yticklabels=label_encoder.classes_ if label_encoder else np.unique(target))
        plt.title(f"Matrice de confusion avec k-NN (sur dataset {dataset_type}) en utilisant LOOCV \n"
                  f"Paramètre: n_neighbors={n_neighbors}")
        plt.xlabel('Prédictions')
        plt.ylabel('Réalités')
        plt.savefig(f"./kNN_confusion_matrix_LOOCV_{dataset_type}.png") 
        #plt.show()
        plt.close()  # Ferme la figure après sauvegarde

    # Calcul de l'accuracy moyenne
    accuracy = accuracy_score(true_labels, predicted_labels)
    return accuracy

accuracy_knn= KNN(5, Xtrain,ytrain,Xtest,ytest, dataset_type="non normalisé",show_confusion_matrix=True) #Division donnée + pas de normalisation
accuracy_knn_loo= KNN_loo(5, data, target,dataset_type="non normalisé",show_confusion_matrix=True) #Leave one out + pas de normalisation
accuracy_knn_normalize= KNN(5, N_Xtrain,N_ytrain,N_Xtest,N_ytest,dataset_type="normalisé",show_confusion_matrix=True) #Division donnée + normalisation
accuracy_knn_loo_normalize= KNN_loo(5, data_normalize, target_normalize, dataset_type="normalisé",show_confusion_matrix=True) #Leave one out + normalisation
print("Paramètre : n_neighbors=5 (paramètre par défaut)")
print(f"Accuracy avec k-NN : {accuracy_knn:.2f}")
print(f"Accuracy avec k-NN en utilisant la méthode leave-one-out : {accuracy_knn_loo:.2f}")
print(f"Accuracy avec k-NN sur données normalisées : {accuracy_knn_normalize:.2f}")
print(f"Accuracy avec k-NN sur données normalisées en utilisant la méthode leave-one-out : {accuracy_knn_loo_normalize:.2f} \n")



##======= Variation de l'hyperparamètre : n_neighbors =======
n_neighbors_values = [2,3,5,7,8] # Valeurs de n_neighbors à tester

# Initialisation des listes pour stocker les résultats
accuracies_div = []
accuracies_loo = []
accuracies_div_normalized = []
accuracies_loo_normalized = []

# Calcul des accuracy pour chaque min_samples_leaf
for n_neighbors in n_neighbors_values:
    # Cas 1 : Division donnée + pas de normalisation
    accuracy_knn = KNN(n_neighbors, Xtrain, ytrain, Xtest, ytest, dataset_type="non normalisé", show_confusion_matrix=False)
    accuracies_div.append(accuracy_knn)
    
    # Cas 2 : Leave-one-out + pas de normalisation
    accuracy_knn_loo = KNN_loo(n_neighbors, data, target, dataset_type="non normalisé", show_confusion_matrix=False)
    accuracies_loo.append(accuracy_knn_loo)
    
    # Cas 3 : Division donnée + normalisation
    accuracy_knn_normalize = KNN(n_neighbors, N_Xtrain, N_ytrain, N_Xtest, N_ytest, dataset_type="normalisé", show_confusion_matrix=False)
    accuracies_div_normalized.append(accuracy_knn_normalize)
    
    # Cas 4 : Leave-one-out + normalisation
    accuracy_knn_loo_normalize = KNN_loo(n_neighbors, data_normalize, target_normalize, dataset_type="normalisé",show_confusion_matrix=False)
    accuracies_loo_normalized.append(accuracy_knn_loo_normalize)

# Tracer les résultats
plt.figure(figsize=(10, 6))
plt.plot(n_neighbors_values, accuracies_div, marker='o', label="Division donnée (non normalisé)", color="blue")
plt.plot(n_neighbors_values, accuracies_loo, marker='o', label="LOOCV (non normalisé)", color="red")
plt.plot(n_neighbors_values, accuracies_div_normalized, marker='o', label="Division donnée (normalisé)", color="green")
plt.plot(n_neighbors_values, accuracies_loo_normalized, marker='o', label="LOOCV (normalisé)", color="orange")
plt.title("[k-NN] Accuracy en fonction de n_neighbors")
plt.xlabel("Valeur de n_neighbors")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.savefig('./KNN_n_neighbors.png')  # Sauvegarde la figure
#plt.show()
plt.close()  # Ferme la figure après sauvegarde

##============== Paramètres optimales: n_neighors=7  =============##  
accuracy_knn= KNN(7, Xtrain,ytrain,Xtest,ytest, dataset_type="non normalisé",show_confusion_matrix=True) #Division donnée + pas de normalisation
accuracy_knn_loo= KNN_loo(7, data, target,dataset_type="non normalisé",show_confusion_matrix=True) #Leave one out + pas de normalisation
accuracy_knn_normalize= KNN(7, N_Xtrain,N_ytrain,N_Xtest,N_ytest,dataset_type="normalisé",show_confusion_matrix=True) #Division donnée + normalisation
accuracy_knn_loo_normalize= KNN_loo(7, data_normalize, target_normalize, dataset_type="normalisé",show_confusion_matrix=True) #Leave one out + normalisation
print("Paramètre : n_neighbors=7 (paramètre 'optimale'')")
print(f"Accuracy avec k-NN : {accuracy_knn:.2f}")
print(f"Accuracy avec k-NN en utilisant la méthode leave-one-out : {accuracy_knn_loo:.2f}")
print(f"Accuracy avec k-NN sur données normalisées : {accuracy_knn_normalize:.2f}")
print(f"Accuracy avec k-NN sur données normalisées en utilisant la méthode leave-one-out : {accuracy_knn_loo_normalize:.2f} \n")

"""
Avec les paramètres par défaut:
- Div ensemble :0.95
- Div ensemble + normalisation : 0.54
- LOOCV : 0.97
- LOOCV + normalisation : 0.54

Faire varier les hyperparamètres:
--> n_neighbors : Cas "Div ensemble" et "LOOCV" ont les meilleurs accuracy
Cas "Div ensemble + normalisation" et "LOOCV + normalisation" ont des accuracy
faible (autour de 0.5), même si on augemente la valeur de n_neighbors
On va prendre n_neighbors=7 (plus stables)

Avec les paramètres 'optimales':
- Div ensemble :0.95
- Div ensemble + normalisation : 0.54
- LOOCV : 0.97
- LOOCV + normalisation : 0.54
"""


# #=================== Réseaux de neurones =====================
# from sklearn.neural_network import MLPClassifier, MLPRegressor

# #Définir le classifieur et calcul du poids
# mlpc=MLPClassifier(hidden_layer_sizes=(100,), max_iter=10000,activation='logistic',learning_rate='adaptive')
# mlpc.fit(Xtrain,ytrain) #Entrainement de nos données d'entrainement
# #Prédiction sur l'ensemble test
# ypred= mlpc.predict(Xtest)
# accuracy=accuracy_score(ytest, ypred)
# print(f"Accuracy avec Réseau de neurones : {accuracy:.2f}")

# # Calcul de la matrice de confusion
# cm = confusion_matrix(ytest, ypred)

# # Visualisation de la matrice de confusion
# plt.figure(figsize=(8, 6))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['faible', 'moderee', 'elevee'], yticklabels=['faible', 'moderee', 'elevee'])
# plt.title('Matrice de confusion pour Réseau de neurones')
# plt.xlabel('Prédictions')
# plt.ylabel('Réalités')
# plt.show()




# #=================== SVM =====================

# #Se restreindre à deux caractéristiques (par exemple les deux premiers)
# #Et calculer les min et max
# """
# Il serait interessant de changer les colonnes choisis
# """
# # Extraire les deux premières colonnes en tant que tableau NumPy
# X = data.iloc[:, :2].to_numpy()  # Convertir en tableau NumPy
# #Transformer les classes en nb entier
# from sklearn.preprocessing import LabelEncoder
# label_encoder = LabelEncoder()
# target_np = label_encoder.fit_transform(target.values.ravel())
# ytrain_encoded = label_encoder.fit_transform(ytrain.values.ravel())
# ytest_encoded = label_encoder.transform(ytest.values.ravel())


# # Calcul des limites pour la première colonne (x) et la deuxième colonne (y)
# x_min = X[:, 0].min()  # Minimum de la première colonne
# x_max = X[:, 0].max()  # Maximum de la première colonne
# y_min = X[:, 1].min()  # Minimum de la deuxième colonne
# y_max = X[:, 1].max()  # Maximum de la deuxième colonne

# print(f"x_min = {x_min}, x_max = {x_max}, y_min = {y_min}, y_max = {y_max}")

# #Contruction du classificateur
# C=4
# gamma=5
# h=0.1

# from sklearn.svm import LinearSVC
# from sklearn.svm import SVC
# clf = LinearSVC(C=C,max_iter=20000).fit(X, target_np)
# clf = SVC(kernel='linear', C=C).fit(X, target_np)
# clf.fit(Xtrain, ytrain)
# #clf = SVC(kernel='rbf', gamma=gamma, max_iter=100000, C=C).fit(X, target_np)
# #clf = SVC(kernel='poly', degree=3, max_iter=100000, gamma=gamma,C=C).fit(X, target_np)

# """
# choisir le meilleur modèle
# """
# # Prédictions et évaluation
# ypred = clf.predict(Xtest)
# accuracy = accuracy_score(ytest_encoded, ypred) 
# print(f"Accuracy avec SVM : {accuracy:.2f}")

# # # Visualisation des frontières de décision
# # import matplotlib.pyplot as plt

# # xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
# # Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
# # Z = Z.reshape(xx.shape)

# # plt.figure(figsize=(10, 6))
# # plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
# # plt.scatter(X[:, 0], X[:, 1], c=target_np, cmap=plt.cm.coolwarm, edgecolors="k")
# # plt.title("SVM - Frontières de décision")
# # plt.xlabel("Feature 1")
# # plt.ylabel("Feature 2")
# # plt.show()











