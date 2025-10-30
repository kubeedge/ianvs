import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Any

def text_to_vector(texts: List[str]) -> np.ndarray:
    """Convert texts into vector representations (TF-IDF)."""
    vec = TfidfVectorizer(max_features=64)
    matrix = vec.fit_transform(texts)
    return matrix.toarray()

def apply_dp_noise(vectors: np.ndarray, epsilon: float = 1.0) -> np.ndarray:
    """Optionally add Laplacian DP noise to each feature vector."""
    noise = np.random.laplace(0, 1/epsilon, vectors.shape)
    return vectors + noise

# Example usage
if __name__ == "__main__":
    txts = ["My number is 12345678901.", "Hello from demo@example.com"]
    vecs = text_to_vector(txts)
    print("Vectors: ", vecs)
    noisy_vecs = apply_dp_noise(vecs)
    print("DP-noised vectors:", noisy_vecs)
