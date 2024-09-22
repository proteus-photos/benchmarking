import numpy as np
from multiprocessing import Pool
import os
import json
from tqdm import tqdm
from utils import tilize_by_anchors
import pickle

def n_range_overlap_slice(n_range1, n_range2):
    # gives the vertical and horizontal overlap indices, for 2 n_ranges
    l1, t1, r1, b1 = n_range1
    l2, t2, r2, b2 = n_range2

    l_max = max(l1, l2)
    t_max = max(t1, t2)
    r_min = min(r1, r2)
    b_min = min(b1, b2)

    return (
        ((l_max - l1, r_min - l1), (t_max - t1, b_min - t1)),
        ((l_max - l2, r_min - l2), (t_max - t2, b_min - t2))
    )

X = 0
Y = 1
W = 2
H = 3

X1 = 0
Y1 = 1
X2 = 2
Y2 = 3

def transform_point(point, anchors1, anchors2):
    # The resulting coordinates will be in the space of anchors2
    x11, y11, x12, y12 = anchors1.T
    x21, y21, x22, y22 = anchors2.T

    a_x = (x21 - x22) / (x11 - x12)
    b_x = x21 - a_x * x11
    
    a_y = (y21 - y22) / (y11 - y12)
    b_y = y21 - a_y * y11

    return a_x * point[X] + b_x, a_y * point[Y] + b_y

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
        
        similarities = (hash.reshape(1, -1) == self.hashes).mean(axis=1)

        inds = np.argpartition(similarities, -k)[-k:] # top k nearest hashes

        return_data = [{"index": ind, "hash": self.hashes[ind], "score": similarities[ind]} for ind in inds]

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
    def __init__(self, n_tiles, hashes=None, storedir=None, anchors=None):  
        self.n_tiles = n_tiles
        self.tile_size = 1 / n_tiles

        # the hashes are stored as (n_images, n_tiles (col), n_tiles (row), hash_size)

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

    def query(self, image, hasher, anchor_points, K_RETRIEVAL=1):
        # wrt the db images, what is the coordinates of the common portion?
        left, top = transform_point(
            np.array([0., 0.]),
            anchor_points[None].repeat(len(self.anchors), axis=0), 
            self.anchors
        )
        left, top = np.clip(left, 0, 1), np.clip(top, 0, 1)

        right, bottom = transform_point(
            np.array([1., 1.]),
            anchor_points[None].repeat(len(self.anchors), axis=0),
            self.anchors
        )
        right, bottom = np.clip(right, 0, 1), np.clip(bottom, 0, 1)

        tile_range_left = np.ceil(left / self.tile_size)
        tile_range_top = np.ceil(top / self.tile_size)
        tile_range_right = np.floor(right / self.tile_size)
        tile_range_bottom = np.floor(bottom / self.tile_size)

        h_ranges = [np.arange(int(left), int(right)) for left, right in zip(tile_range_left, tile_range_right)]
        v_ranges = [np.arange(int(top), int(bottom)) for top, bottom in zip(tile_range_top, tile_range_bottom)]

        # contains the indices of the tiles that are common between the query image and the db images
        tiles_indices_list = [np.array(np.meshgrid(v_range, h_range)).T.reshape(-1,2) for v_range, h_range in zip(h_ranges, v_ranges)]
        # we take v, h because while indexing y axis comes first then h

        tile_coordinates_list = [
            [
                (
                    [h * self.tile_size, v * self.tile_size],
                    [(h + 1) * self.tile_size, (v + 1) * self.tile_size]
                ) for v, h in tiles_indices
            ] for tiles_indices in tiles_indices_list
        ]

        similarities = []
        for i, (tiles_indices, tile_coordinates, db_anchor) in enumerate(zip(tiles_indices_list, tile_coordinates_list, self.anchors)):
            if len(tiles_indices) == 0:
                similarities.append(0)
                continue
            
            tile_images = []
            for left_top, right_bottom in tile_coordinates:
                # we find what the coordinates of the tile in the ORIGINAL image would be in our query image space
                left, top = transform_point(left_top, db_anchor[None], anchor_points[None])
                left, top = left[0]*image.size[0], top[0]*image.size[1]

                right, bottom = transform_point(right_bottom, db_anchor[None], anchor_points[None])
                right, bottom = right[0]*image.size[0], bottom[0]*image.size[1]

                tile_images.append(image.crop((left, top, right, bottom)))

            query_tile_hashes = hasher(tile_images)

            query_tile_hashes = np.array(query_tile_hashes)
            db_tile_hashes = self.hashes[i, tiles_indices[:, 1], tiles_indices[:, 0]]

            # avg similarity for each tile
            similarity = (query_tile_hashes == db_tile_hashes).mean()
            similarities.append(similarity)

        similarities = np.array(similarities)

        inds = np.argpartition(similarities, -K_RETRIEVAL)[-K_RETRIEVAL:]

        return_data = [{"index": ind, "score": similarities[ind]} for ind in inds]
        
        return return_data
    
    def multi_query(self, images, hasher, anchor_points_list, K_RETRIEVAL=1):
        return [self.query(image, hasher, anchor_points, K_RETRIEVAL) for image, anchor_points in zip(images, anchor_points_list)]

class TileDatabaseV2:
    def __init__(self, hashes=None, storedir=None, metadata=None, n_breaks=2):  
        # the hashes are stored as (n_images, n_tiles (col), n_tiles (row), hash_size)
        self.n_breaks = n_breaks
        if hashes is None:
            if storedir is None:
                raise ValueError
            else:
                with open(storedir+"_hashes.pkl", "rb") as f:
                    self.hashes = pickle.load(f)
                with open(storedir+"_metadata.json", "r") as f:
                    self.metadata = json.load(f)
        else:
            self.hashes = hashes
            self.metadata = metadata
            if storedir is not None:
                with open(storedir+"_hashes.pkl", "wb") as f:
                    pickle.dump(hashes, f)
                with open(storedir+"_metadata.json", "w") as f:
                    json.dump(metadata, f)

        self.anchors = self.metadata["anchors"]
        self.n_ranges = self.metadata["n_ranges"]


    def query(self, image, hasher, query_anchor, K_RETRIEVAL=1):
        query_tiles, query_n_range = tilize_by_anchors(image, self.n_breaks, query_anchor)
        grid_shape = (query_n_range[Y2] - query_n_range[Y1], query_n_range[X2] - query_n_range[X1])
        query_tile_hashes = hasher(query_tiles).reshape(*grid_shape, -1)

        nums = query_tile_hashes.reshape(-1, query_tile_hashes.shape[-1]).dot(1 << np.arange(query_tile_hashes.shape[-1])[::-1])

        similarities = []
        
        for db_tile_hashes, db_n_range in zip(self.hashes, self.n_ranges):
            query_overlap, db_overlap = n_range_overlap_slice(query_n_range, db_n_range)
            query_hashes = query_tile_hashes[query_overlap[Y][0]:query_overlap[Y][1],
                                             query_overlap[X][0]:query_overlap[X][1]]
            
            db_hashes = db_tile_hashes[db_overlap[Y][0]:db_overlap[Y][1],
                                       db_overlap[X][0]:db_overlap[X][1]]
            
            similarity = (query_hashes == db_hashes).mean()
            similarities.append(similarity)

        similarities = np.array(similarities)

        inds = np.argpartition(similarities, -K_RETRIEVAL)[-K_RETRIEVAL:]

        return_data = [{"index": ind, "score": similarities[ind]} for ind in inds]
        
        return return_data
    
    def multi_query(self, images, hasher, anchor_points_list, K_RETRIEVAL=1):
        return [self.query(image, hasher, anchor_points, K_RETRIEVAL) for image, anchor_points in zip(images, anchor_points_list)]