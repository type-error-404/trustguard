"""
faiss_index.py
--------------
Optional FAISS feature index.

Used for future similarity-based
forensic retrieval.
"""

import os
import numpy as np


INDEX_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "outputs",
    "trustguard.index"
)


def add_embedding(
    embedding
):

    try:

        import faiss

        vector = np.array(
            [embedding],
            dtype="float32"
        )

        dimension = vector.shape[1]

        if os.path.exists(
            INDEX_PATH
        ):

            index = faiss.read_index(
                INDEX_PATH
            )

        else:

            index = faiss.IndexFlatL2(
                dimension
            )

        index.add(
            vector
        )

        os.makedirs(
            os.path.dirname(
                INDEX_PATH
            ),
            exist_ok=True
        )

        faiss.write_index(
            index,
            INDEX_PATH
        )

        return {
            "success": True,
            "total_vectors":
                index.ntotal
        }

    except Exception as e:

        return {
            "success": False,
            "error": str(e)
        }