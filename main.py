from typing import List
import numpy as np
from utils import get_embed

# TODO:
# 1. dataset download
#   1-1. dataset load
#   1-2. dataset > ndarr conversion
# 2. faiss index type
# 3. index train
# 4. index add
# 5. retrieve
# 6. CLI arg parse


# def retrieve(query_texts: List[str], k: int, index, dataset):
#     query_embeddings = get_embed(query_texts)
#
#     # Search
#     distances, indices = index.search(query_embeddings.shape[0], query_embeddings, k)
#
#     results = []
#     for q_idx, query in enumerate(query_texts):
#         query_results = []
#         for i, doc_idx in enumerate(indices[q_idx]):
#             retrieved_doc = dataset[doc_idx]
#             query_results.append(
#                 {
#                     "query": query,
#                     "retrieved_title": retrieved_doc["title"],
#                     "retrieved_text": retrieved_doc["text"],
#                     "distance": distances[q_idx][i],
#                 }
#             )
#         results.append(query_results)
#
#     return results


if __name__ == "__main__":
    print()
