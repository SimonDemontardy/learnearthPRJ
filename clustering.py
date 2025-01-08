
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def perform_clustering(data):
    features = ['Magnitude', 'Depth', 'population_impacted']
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data[features])
    kmeans = KMeans(n_clusters=3, random_state=42)
    data['cluster'] = kmeans.fit_predict(data_scaled)
    return data

if __name__ == "__main__":
    data = pd.read_csv('cleaned_processed_data.csv')
    data = perform_clustering(data)
    plt.scatter(data['Magnitude'], data['Depth'], c=data['cluster'], cmap='viridis')
    plt.xlabel('Magnitude')
    plt.ylabel('Depth')
    plt.title('Clusters des séismes')
    plt.show()