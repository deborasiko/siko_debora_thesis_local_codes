from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

sentences = ["Hello, world", "Hi, world"]

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
embeddings = model.encode(sentences)
print(embeddings)
# Compute cosine similarity
similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

print("Cosine similarity:", similarity)
