import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
import requests
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt

# Load the data
earth = pd.read_csv('Earth/database.csv').head(10)
print(earth.head())

# dropt the unnecessary columns :
# ID,Source,Location Source,Magnitude Source,Status
earth.drop(['ID', 'Source', 'Location Source', 'Magnitude Source', 'Status'], axis=1, inplace=True)
print(earth.head())
#earth.dropna(inplace=True)

def get_population(lat, lon, radius=100):
    """
    Function to estimate the population within a given radius around specific coordinates.
    Uses Overpass API.
    """
    overpass_url = "http://overpass-api.de/api/interpreter"
    query = f"""
    [out:json];
    (
      node["population"](around:{radius * 1000}, {lat}, {lon});
    );
    out body;
    """
    response = requests.get(overpass_url, params={'data': query})
    if response.status_code == 200:
        data = response.json()
        print(f"Response data for lat: {lat}, lon: {lon} -> {data}")  # Debugging line
        return sum(int(elm.get("tags", {}).get("population", 0)) for elm in data["elements"])
    else:
        print(f"Failed to get data for lat: {lat}, lon: {lon}, status code: {response.status_code}")  # Debugging line
    return 0


# Ajouter une colonne pour la population impactée
earth['population_impacted'] = earth.apply(lambda x: get_population(x['Latitude'], x['Longitude']), axis=1)

print(earth.head())

# Créer une géométrie pour les données
geometry = [Point(xy) for xy in zip(earth['Longitude'], earth['Latitude'])]
geo_data = gpd.GeoDataFrame(earth, geometry=geometry)

# Afficher un aperçu
print(geo_data.head())


