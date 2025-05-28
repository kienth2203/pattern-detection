from autoencoder import train_autoencoder, extract_latent_features
from gnn_extractor import extract_graph_embeddings
from cluster_visualize import cluster_and_crop
import glob
import numpy as np

image_paths = glob.glob('pattern_samples/*.jpg')
model = train_autoencoder(image_paths, epochs=20)
latent_features = extract_latent_features(model, image_paths)
graph_embeddings = extract_graph_embeddings(image_paths)
combined_features = np.hstack([latent_features, graph_embeddings])
cluster_and_crop(combined_features, image_paths)
