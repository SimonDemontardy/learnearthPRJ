import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import requests


# Load the data
earth = pd.read_csv('database.csv').head(1000)
print(earth.head())

# dropt the unnecessary columns :
# ID,Source,Location Source,Magnitude Source,Status
earth.drop(['ID', 'Source', 'Location Source', 'Magnitude Source', 'Status'], axis=1, inplace=True)
print(earth.head())
#earth.dropna(inplace=True)

# cette fonction permet de récupérer les données de population grâce à l'API overpass. 
# la population correspond à une coordonnées donnée + le rayon de 100 km autour
def get_population(lat, lon, radius=100):
    overpass_url = "http://overpass-api.de/api/interpreter"
    query = f"""
    [out:json];
    (
      node["population"](around:{radius * 1000}, {lat}, {lon});
    );
    out body;
    """
    try:
        response = requests.get(overpass_url, params={'data': query}, timeout=60)
        if response.status_code == 200:
            data = response.json()
            #print(f"Response data for lat: {lat}, lon: {lon} -> {data}")  # Debugging line
            population_sum = 0
            for elm in data["elements"]:
                population_str = elm.get("tags", {}).get("population", "0")
                try:
                    population = int(population_str.replace(' habitantes', '').replace('.', '').strip())
                    population_sum += population
                except ValueError:
                    print(f"Invalid population value: {population_str}")
            return population_sum
        else:
            print(f"Failed to get data for lat: {lat}, lon: {lon}, status code: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"Request failed for lat: {lat}, lon: {lon}, error: {e}")
    return 0

# Ajouter une colonne pour la population impactée
earth['population_impacted'] = earth.apply(lambda x: get_population(x['Latitude'], x['Longitude']), axis=1)

print(earth.head())

# Créer une géométrie pour les données (BONUS) afin de créer une manière de visualiser
geometry = [Point(xy) for xy in zip(earth['Longitude'], earth['Latitude'])]
geo_data = gpd.GeoDataFrame(earth, geometry=geometry)

# Afficher un aperçu
print(geo_data.head())

# enregistrement des resultats dans un nouveau csv
geo_data.to_csv('processed_data1000.csv', index=False)


