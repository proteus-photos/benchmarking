import sys
import onnxruntime
import numpy as np
from PIL import Image
import os
from tqdm import tqdm
from onnx import helper, TensorProto, load, save

seed1 = open("./hashes/neuralhash_128x96_seed1.dat", "rb").read()[128:]
seed1 = np.frombuffer(seed1, dtype=np.float32)
seed1 = seed1.reshape([96, 128])

# Load ONNX model
session = None

def neuralhash(ims, bits=128, *args, **kwargs):
    global session
    # Load output hash matrix
    
    image_arrays = []
    for im in ims:
        image = im.convert("RGB").resize([360, 360])
        arr = np.array(image).astype(np.float32) / 255.0
        image_arrays.append((arr * 2.0 - 1.0).transpose(2, 0, 1))

    arr = np.stack(image_arrays, axis=0)

    inputs = {session.get_inputs()[0].name: arr}
    outs = np.array(session.run(None, inputs))[0].squeeze((2,3))
    hash_output = outs @ seed1[:bits].T
    hash_bits = hash_output > 0
    # hash_hex = "{:0{}x}".format(int(hash_bits, 2), len(hash_bits) // 4)
    # hash_bits_list.append(hash_bits)
    return hash_bits


# def initialize_session():
#     global session
so = onnxruntime.SessionOptions()
so.log_severity_level = 3
session = onnxruntime.InferenceSession("./hashes/model.onnx", so)