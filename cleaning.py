import pandas as pd

# Charger le CSV
file_path = 'database.csv' 
columns_to_keep = ['Date','Time', 'Magnitude', 'Depth', 'Longitude', 'Latitude']

# Lire le fichier CSV en spécifiant les colonnes à conserver
data = pd.read_csv(file_path, usecols=columns_to_keep)

# Afficher les premières lignes du DataFrame
print(data.head())

# Sauvegarder le fichier nettoyé si nécessaire
data.to_csv('cleaned_earth_data.csv', index=False)
