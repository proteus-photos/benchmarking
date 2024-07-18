import numpy as np
from multiprocessing import Pool

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

    def query(self, hash, k=1, start_index=None, end_index=None):
        if start_index is None:
            start_index = 0
        if end_index is None:
            end_index = len(self.hashes)
        
        similarity = (hash.reshape(1, -1) == self.hashes[start_index:end_index]).sum(axis=1)

        inds = np.argpartition(similarity, -k)[-k:] # top k nearest hashes

        return_data = [{"index": start_index+ind, "hash": self.hashes[start_index+ind], "score": similarity[ind]} for ind in inds]

        if self.metadata is not None:
            for data in return_data:
                data["metadata"] = self.metadata[data["index"]]
        
        return return_data
    
    def parallel_query(self, hash, k=1, num_workers=16):
        start_indices = np.arange(0, len(self.hashes), len(self.hashes)//num_workers)
        end_indices = np.concatenate([start_indices[1:], [len(self.hashes)]])

        with Pool(num_workers) as p:
            queries_promise = p.starmap_async(self.query, [(hash, k, start_indices[i], end_indices[i]) for i in range(num_workers)])
            return_data = [point for points in queries_promise.get() for point in points]
        
        # since we have the top k nearest hashes for each worker, we need to find the top k nearest hashes overall
        similarity = np.array([point["score"] for point in return_data])
        inds = np.argpartition(similarity, -k)[-k:]

        return [return_data[ind] for ind in inds]