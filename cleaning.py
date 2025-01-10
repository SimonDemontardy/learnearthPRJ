import pandas as pd

# Charger le CSV
file_path = 'processed_data1000.csv' 
columns_to_keep = ['Date','Time', 'Magnitude', 'Depth', 'Longitude', 'Latitude','population_impacted']

# Lire le fichier CSV en spécifiant les colonnes à conserver
data = pd.read_csv(file_path, usecols=columns_to_keep)

# Afficher les premières lignes du DataFrame
print(data.head())

# Sauvegarder le fichier nettoyé si nécessaire
data.to_csv('cleaned_processed_data1000.csv', index=False)
