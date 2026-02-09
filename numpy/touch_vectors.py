import numpy as np

v1 = np.array([1, 2, 3])
v2 = np.array([2, 4, 6])
v3 = np.array([-1, -2, -3])

# Length (magnitude)
print(np.linalg.norm(v1))

# Dot product
print(np.dot(v1, v2))
print(np.dot(v1, v3))

# 1 2 3
# 2 4 6
#-1 -2 -3

def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

print(cosine_sim(v1, v2))  # ~1
print(cosine_sim(v1, v3))  # ~-1

# Q: What a vector is?
# A: A vector is a mathematical object that has both magnitude and direction.

# Q: Why distance matters more than exact values?
# A: Distance helps us understand similarity and relationships between data points, which is crucial for many machine learning algorithms.

# Q: Why cosine similarity is often better than euclidean distance?
# A: Cosine similarity is often better than euclidean distance because it is scale-invariant and focuses on the angle between vectors, which is more meaningful for high-dimensional data.
