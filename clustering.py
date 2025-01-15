########## IMPORTS #########

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
import seaborn as sns


######### loading data #########
data = pd.read_csv('datasets/processed_data_all_cleaned.csv')

######### modifiying values #########
# test with logs
#data['Magnitude_log'] = np.log1p(data['Magnitude'])
#data['Depth_log'] = np.log1p(data['Depth'])
#data['Depth_inverse_log'] = np.log1p(data['Depth_inverse'])

# to visualise the data we use the log of population_impacted which varies too much
data['population_impacted_log'] = np.log1p(data['population_impacted'])
# we are going to use the inverse of the depth since it should impact inversely proportional the dangerosity
data['Depth_inverse'] = 1 / (data['Depth'] + 1)



######### data preparation #########
# we are using the following features
features = [ 'Magnitude', 'Depth_inverse', 'population_impacted_log']
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data[features])
# adding weight
data_scaled[:, 0] *= 2  # magnitude weight
data_scaled[:, 1] *= 1  # depth inverse weight
data_scaled[:, 2] *= 15  # highest weight since the dangerosity highly depend on the population affected

######### Clustering with K-Means #########
kmeans = KMeans(n_clusters=3, random_state=42)
data['cluster_kmeans'] = kmeans.fit_predict(data_scaled)

# k_means clusters summary
kmeans_summary = data.groupby('cluster_kmeans')[features].mean()
print("Résumé des clusters (K-Means) :")
print(kmeans_summary)

######### Clustering with DBSCAN #########
dbscan = DBSCAN(eps=2, min_samples=3)  
data['cluster_dbscan'] = dbscan.fit_predict(data_scaled)

# clusters DBSCANS
print("Clusters DBSCAN détectés :", np.unique(data['cluster_dbscan']))

######### dangerosity levels attributions #########

# K-Means : 
dangerosite_mapping_kmeans = {
    0: 'faible',  
    1: 'moderee', 
    2: 'elevee'   
}
# write the danger level in the data
data['dangerosite_kmeans'] = data['cluster_kmeans'].map(dangerosite_mapping_kmeans)

# DBSCAN :   -1= anomalies (values without cluster)
dangerosite_mapping_dbscan = {
    -1: 'anomalie',  # values without cluster
    0: 'faible',     
    1: 'elevee',     
    2: 'moderee'     
}
# write the danger level in the data
data['dangerosite_dbscan'] = data['cluster_dbscan'].map(dangerosite_mapping_dbscan)

######### clusters visualisation #########
# Réduction des dimensions avec PCA pour visualisation
pca = PCA(n_components=2)
X_pca = pca.fit_transform(data_scaled)
print(data[(X_pca[:, 1] > 20)])  # Vérifiez les points avec PCA2 > 20

# this code part was used to avoid the extreme outlayers, but using the log of population this problem was solved
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

# Sauvegarder le graphique
plt.savefig('clustering_results/PCAclustering.png')
plt.show()

######### Résultats finaux #########
# cela nous permet de visualiser les données pour vérifier la cohérence des attributions et ajouter ou non un ajustement. 
print("\nDataset final avec dangerosité attribuée :")
print(data)
data.to_csv('clustering_results_all.txt', sep='\t', index=False)
print("\nLes résultats ont été sauvegardés dans 'clustering_results_all.txt'.")

######### Cluster visualisation per variable #########

# subplots
fig, axes = plt.subplots(1, 3, figsize=(15, 5))  # 1 ligne, 3 colonnes

# Boxplot Magnitude
sns.boxplot(x='cluster_kmeans', y='Magnitude', data=data, ax=axes[0])
axes[0].set_title('Magnitude par Cluster')
axes[0].set_ylabel('Magnitude')

# Boxplot Depth
sns.boxplot(x='cluster_kmeans', y='Depth', data=data, ax=axes[1])
axes[1].set_title('Depth par Cluster')
axes[1].set_ylabel('Depth')

# Boxplot population_impacted
sns.boxplot(x='cluster_kmeans', y='population_impacted', data=data, ax=axes[2])
axes[2].set_title('Population impactée par Cluster')
axes[2].set_ylabel('Population Impactée')

plt.tight_layout()
plt.savefig('clustering_results/boxplots_by_cluster.png')
plt.show()


# Identification of incoherences in `data`
incoherences = data[
    ((data['Depth'] > 500) & (data['cluster_kmeans'] == 2)) |  # Profondeur élevée mais forte dangerosité
    ((data['population_impacted'] < 1000) & (data['cluster_kmeans'] == 2)) |  # Population faible mais forte dangerosité
    ((data['Magnitude'] < 6.0) & (data['cluster_kmeans'] == 2)) |  # Magnitude faible mais forte dangerosité
    ((data['population_impacted'] == 0) & (data['cluster_kmeans'] == 2))  # Population nulle mais forte dangerosité
]


# Affichage des incohérences
print("Incohérences identifiées :")
print(incoherences[['Date', 'Depth', 'Magnitude', 'population_impacted', 'cluster_kmeans', 'dangerosite_kmeans']])

#this function allow to correct incoherences within the clusters danger level attribution
def adjust_dangerosity(row):
    if row['population_impacted'] < 2000:  # Si la population impactée est faible
        return 'faible' if row['Magnitude'] < 7.0 else 'moderee'
    if row['Depth'] > 500: 
        if row['population_impacted']<5000: # Si la profondeur est très élevée
            return 'faible'
        else:
            return 'moderee'
    # we are applying the adjustements only on the working clustering using the kmeans method
    return row['dangerosite_kmeans']

# apply the adjusted dangerosity
data['dangerosite_adjusted'] = data.apply(adjust_dangerosity, axis=1)

######### Export final CSV #########
# Sélection des colonnes finales
final_columns = ['Date', 'Time', 'Magnitude', 'Depth', 'population_impacted', 'dangerosite_adjusted']

# Création du dataset final avec les colonnes sélectionnées
final_dataset = data[final_columns]

# Export du dataset final vers un fichier CSV
output_file_path = 'datasets/final_clustering_results_all.csv'  # file name for the OUTPUT
final_dataset.to_csv(output_file_path, sep='\t', index=False)

print(f"Les résultats finaux ont été sauvegardés dans : {output_file_path}")


