from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

def cluster_and_crop(features, image_paths, output_dir='unit_cells'):
    os.makedirs(output_dir, exist_ok=True)
    clustering = DBSCAN(eps=3, min_samples=2).fit(features)
    labels = clustering.labels_
    tsne = TSNE(n_components=2, perplexity=min(len(features)-1, 30)).fit_transform(features)
    plt.scatter(tsne[:, 0], tsne[:, 1], c=labels, cmap='tab10')
    plt.title("t-SNE Clustering")
    plt.show()

    for label in np.unique(labels):
        if label == -1: continue
        idxs = np.where(labels == label)[0]
        for i, idx in enumerate(idxs):
            img = cv2.imread(image_paths[idx])
            h, w = img.shape[:2]
            crop = img[h//4: 3*h//4, w//4: 3*w//4]
            out_path = os.path.join(output_dir, f'cluster_{label}_{i}.jpg')
            cv2.imwrite(out_path, crop)
