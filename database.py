import numpy as np

class Database:
    def __init__(self, hashes=None, storedir=None, metadata=None, refresh=True):
        self.metadata = metadata
        
        if hashes is None:
            if storedir is None:
                raise ValueError
            else:
                self.hashes = np.load(storedir+".npy")
        else:
            self.hashes = np.array(hashes)
            if storedir is not None:
                np.save(storedir, hashes)

    def query(self, hash, k=1):
        similarity = (hash.reshape(1, -1) == self.hashes).sum(axis=1)

        inds = np.argpartition(similarity, -k)[-k:] # top k nearest hashes

        return_data = [{"index": ind, "hash": self.hashes[ind], "score": similarity[ind]} for ind in inds]

        if self.metadata is not None:
            for data in return_data:
                data["metadata"] = self.metadata[data["index"]]
        
        return return_data