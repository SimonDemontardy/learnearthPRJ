import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np

# ====== 1. Chargement des données ======
data = pd.read_csv('processed_data_all_cleaned.csv')
data['Magnitude_log'] = np.log1p(data['Magnitude'])
data['Depth_log'] = np.log1p(data['Depth'])
data['population_impacted_log'] = np.log1p(data['population_impacted'])
data['Depth_inverse'] = 1 / (data['Depth'] + 1)
data['Depth_inverse_log'] = np.log1p(data['Depth_inverse'])


# ====== 2. Préparation des données ======
# Normalisation des caractéristiques
#features = ['Magnitude', 'Depth', 'population_impacted']
features = [ 'Magnitude', 'Depth_inverse', 'population_impacted_log']
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data[features])
# Pondération des variables
data_scaled[:, 0] *= 2  # Poids pour la Magnitude
data_scaled[:, 1] *= 1  # Poids pour la Profondeur Inverse
data_scaled[:, 2] *= 15  # Poids élevé pour la Population Impactée

# ====== 3. Clustering avec K-Means ======
kmeans = KMeans(n_clusters=3, random_state=42)
data['cluster_kmeans'] = kmeans.fit_predict(data_scaled)

# Résumé des clusters pour K-Means
kmeans_summary = data.groupby('cluster_kmeans')[features].mean()
print("Résumé des clusters (K-Means) :")
print(kmeans_summary)

# ====== 4. Clustering avec DBSCAN ======
dbscan = DBSCAN(eps=1.5, min_samples=2)  # Ajustez eps et min_samples selon vos données
data['cluster_dbscan'] = dbscan.fit_predict(data_scaled)

# Vérification des clusters DBSCAN
print("Clusters DBSCAN détectés :", np.unique(data['cluster_dbscan']))

# ====== 5. Attribution des niveaux de dangerosité ======

# K-Means : Mapping basé sur les moyennes des clusters
dangerosite_mapping_kmeans = {
    0: 'faible',  # Exemple
    1: 'moderee', # Exemple
    2: 'elevee'   # Exemple
}
data['dangerosite_kmeans'] = data['cluster_kmeans'].map(dangerosite_mapping_kmeans)

# DBSCAN : Pour DBSCAN, les -1 sont considérés comme anomalies (pas de cluster)
dangerosite_mapping_dbscan = {
    -1: 'anomalie',  # Points non assignés à un cluster
    0: 'faible',     # Exemple
    1: 'elevee',     # Exemple
    2: 'moderee'     # Exemple
}
data['dangerosite_dbscan'] = data['cluster_dbscan'].map(dangerosite_mapping_dbscan)

# ====== 6. Visualisation des clusters ======
# Réduction des dimensions avec PCA pour visualisation
pca = PCA(n_components=2)
X_pca = pca.fit_transform(data_scaled)
print(data[(X_pca[:, 1] > 20)])  # Vérifiez les points avec PCA2 > 20

# Nettoyage des données
#data_cleaned = data[(X_pca[:, 1] <= 20)]  # Garder uniquement les points sous un seuil
#data_scaled_cleaned = scaler.fit_transform(data_cleaned[features])

# Recalcule des clusters
#kmeans_cleaned = KMeans(n_clusters=3, random_state=42)
#data['cluster_kmeans'] = kmeans_cleaned.fit_predict(data_scaled)

#dbscan_cleaned = DBSCAN(eps=1.5, min_samples=2)
#data['cluster_dbscan'] = dbscan_cleaned.fit_predict(data_scaled)

#X_pca_cleaned = PCA(n_components=2).fit_transform(data_scaled)

# Visualisation K-Means
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=data['cluster_kmeans'], cmap='viridis')
plt.title('K-Means Clustering')
plt.xlabel('PCA1')
plt.ylabel('PCA2')

# Visualisation DBSCAN
plt.subplot(1, 2, 2)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=data['cluster_dbscan'], cmap='viridis')
plt.title('DBSCAN Clustering')
plt.xlabel('PCA1')
plt.ylabel('PCA2')

plt.show()

# ====== 7. Résultats finaux ======
print("\nDataset final avec dangerosité attribuée :")
print(data)
data.to_csv('clustering_results_all.txt', sep='\t', index=False)
print("\nLes résultats ont été sauvegardés dans 'clustering_results_all.txt'.")


#data['dangerosite_adjusted'] = data.apply(adjust_dangerosity, axis=1)

import seaborn as sns
print(data['Magnitude'].describe())  # Inspecte les statistiques de la colonne Magnitude
print(data[data['Magnitude'] > 10])  # Identifie les magnitudes anormales
# Comparer les caractéristiques entre clusters


import matplotlib.pyplot as plt
import seaborn as sns

# Création des subplots
fig, axes = plt.subplots(1, 3, figsize=(15, 5))  # 1 ligne, 3 colonnes

# Boxplot pour Magnitude
sns.boxplot(x='cluster_kmeans', y='Magnitude', data=data, ax=axes[0])
axes[0].set_title('Magnitude par Cluster')
axes[0].set_ylabel('Magnitude')

# Boxplot pour Depth
sns.boxplot(x='cluster_kmeans', y='Depth', data=data, ax=axes[1])
axes[1].set_title('Depth par Cluster')
axes[1].set_ylabel('Depth')

# Boxplot pour population_impacted
sns.boxplot(x='cluster_kmeans', y='population_impacted', data=data, ax=axes[2])
axes[2].set_title('Population impactée par Cluster')
axes[2].set_ylabel('Population Impactée')

# Ajuster les espacements
plt.tight_layout()
plt.show()


def adjust_dangerosity(row):
    # Ajuster la dangerosité en fonction des critères
    if row['population_impacted'] < 2000:  # Si la population impactée est faible
        return 'faible' if row['Magnitude'] < 7.0 else 'moderee'
    if row['Depth'] > 500: 
        if row['population_impacted']<5000: # Si la profondeur est très élevée
            return 'faible'
        else:
            return 'moderee'
    return row['dangerosite_kmeans']  # Sinon, utiliser la classification initiale

# Appliquer les ajustements au dataset
data['dangerosite_adjusted'] = data.apply(adjust_dangerosity, axis=1)

# ====== Export du CSV final ======
# Sélection des colonnes finales
final_columns = ['Date', 'Time', 'Magnitude', 'Depth', 'population_impacted', 'dangerosite_adjusted']

# Création du dataset final avec les colonnes sélectionnées
final_dataset = data[final_columns]

# Export du dataset final vers un fichier CSV
output_file_path = 'final_clustering_results_all.csv'  # Nom du fichier
final_dataset.to_csv(output_file_path, sep='\t', index=False)

print(f"Les résultats finaux ont été sauvegardés dans : {output_file_path}")


# Identification des incohérences dans le dataframe principal `data`
incoherences = data[
    ((data['Depth'] > 500) & (data['cluster_kmeans'] == 2)) |  # Profondeur élevée mais forte dangerosité
    ((data['population_impacted'] < 1000) & (data['cluster_kmeans'] == 2)) |  # Population faible mais forte dangerosité
    ((data['Magnitude'] < 6.0) & (data['cluster_kmeans'] == 2)) |  # Magnitude faible mais forte dangerosité
    ((data['population_impacted'] == 0) & (data['cluster_kmeans'] == 2))  # Population nulle mais forte dangerosité
]


# Affichage des incohérences
print("Incohérences identifiées :")
print(incoherences[['Date', 'Depth', 'Magnitude', 'population_impacted', 'cluster_kmeans', 'dangerosite_kmeans']])