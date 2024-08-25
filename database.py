import numpy as np
from multiprocessing import Pool
import os
import json

class Database:
    def __init__(self, hashes=None, storedir=None, metadata=None, refresh=True):        
        if hashes is None:
            if storedir is None:
                raise ValueError
            else:
                self.hashes = np.load(storedir+".npy")
                if metadata is None:
                    try:
                        self.metadata = np.load(storedir+"_metadata.npy")
                    except FileNotFoundError:
                        self.metadata = None
                else:
                    self.metadata = np.array(metadata)
                    np.save(storedir+"_metadata", metadata)
        else:
            self.hashes = np.array(hashes)
            if storedir is not None:
                np.save(storedir, hashes)
                if metadata is not None:
                    self.metadata = np.array(metadata)
                    np.save(storedir+"_metadata", metadata)
                else:
                    self.metadata = None

    def query(self, hash, k=1):
        
        similarity = (hash.reshape(1, -1) == self.hashes).sum(axis=1)

        inds = np.argpartition(similarity, -k)[-k:] # top k nearest hashes

        return_data = [{"index": ind, "hash": self.hashes[ind], "score": similarity[ind]} for ind in inds]

        if self.metadata is not None:
            for data in return_data:
                data["metadata"] = self.metadata[data["index"]]
        
        return return_data
    
    def parallel_query(self, hashes, k=1, num_workers=8):
        with Pool(num_workers) as p:
            queries_promise = p.starmap_async(self.query, [(hash, k) for hash in hashes])
            return_data = [points for points in queries_promise.get()]

        return return_data
    
class TileDatabase:
    def __init__(self, n_tiles, hashes=None, storedir=None, anchors=None, refresh=True):  
        self.n_tiles = n_tiles      
        if hashes is None:
            if storedir is None:
                raise ValueError
            else:
                self.hashes = np.load(storedir+".npy")
                if anchors is None:
                    try:
                        self.anchors = np.load(storedir+"_anchors.npy")
                    except FileNotFoundError:
                        self.anchors = None
                else:
                    self.anchors = np.array(anchors)
                    np.save(storedir+"_anchors", anchors)
        else:
            self.hashes = np.array(hashes)
            if storedir is not None:
                np.save(storedir, hashes)
                if anchors is not None:
                    self.anchors = np.array(anchors)
                    np.save(storedir+"_anchors", anchors)
                else:
                    self.anchors = None
        self.hashes = self.hashes.reshape(-1, n_tiles, n_tiles, self.hashes.shape[-1])

    def query(self, hash, k=1):
        
        similarity = (hash.reshape(1, -1) == self.hashes).sum(axis=1)

        inds = np.argpartition(similarity, -k)[-k:] # top k nearest hashes

        return_data = [{"index": ind, "hash": self.hashes[ind], "score": similarity[ind]} for ind in inds]

        if self.anchors is not None:
            for data in return_data:
                data["anchors"] = self.anchors[data["index"]]
        
        return return_data
    
    def parallel_query(self, hashes, k=1, num_workers=8):
        with Pool(num_workers) as p:
            queries_promise = p.starmap_async(self.query, [(hash, k) for hash in hashes])
            return_data = [points for points in queries_promise.get()]

        return return_data