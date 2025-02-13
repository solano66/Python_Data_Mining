#%%
from pyclustering.cluster.kmedoids import kmedoids # pip install pyclustering
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from pyclustering.cluster import cluster_visualizer
from pyclustering.utils import read_sample
from pyclustering.samples.definitions import FCPS_SAMPLES
#%% 
# Load list of points for cluster analysis.
sample = read_sample(FCPS_SAMPLES.SAMPLE_TWO_DIAMONDS)
#%%
# Initialize initial medoids using K-Means++ algorithm
# Index of initial medoids
initial_medoids = kmeans_plusplus_initializer(sample, amount_centers=2).initialize(return_index=True)
#%%
# Create instance of K-Medoids (PAM) algorithm.
kmedoids_instance = kmedoids(sample, initial_medoids)
#%%
# Run cluster analysis and obtain results.
kmedoids_instance.process()
clusters = kmedoids_instance.get_clusters()
medoids = kmedoids_instance.get_medoids()
#%% 
# Print allocated clusters.
print("Clusters:", clusters)
#%% 
# Display clustering results.
visualizer = cluster_visualizer()
visualizer.append_clusters(clusters, sample)
visualizer.append_cluster(initial_medoids, sample, markersize=12, marker='*', color='gray')
visualizer.append_cluster(medoids, sample, markersize=14, marker='*', color='black')
visualizer.show()

#### Reference:
# https://pyclustering.github.io/docs/0.10.1/html/de/dfd/namespacepyclustering_1_1cluster_1_1kmedoids.html