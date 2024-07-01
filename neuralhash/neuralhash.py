import sys
import onnxruntime
import numpy as np
from PIL import Image
import os


def neuralhash(ims, *args, **kwargs):
    # Load ONNX model
    session = onnxruntime.InferenceSession("./neuralhash/model.onnx")

    # Load output hash matrix
    seed1 = open("./neuralhash//neuralhash_128x96_seed1.dat", "rb").read()[128:]
    seed1 = np.frombuffer(seed1, dtype=np.float32)
    seed1 = seed1.reshape([96, 128])
    
    hash_bits_list = []
    for im in ims:
        # Preprocess image
        image = im.convert("RGB")
        image = image.resize([360, 360])
        arr = np.array(image).astype(np.float32) / 255.0
        arr = arr * 2.0 - 1.0
        arr = arr.transpose(2, 0, 1).reshape([1, 3, 360, 360])

        inputs = {session.get_inputs()[0].name: arr}
        outs = session.run(None, inputs)
        # Convert model output to hex hash
        hash_output = seed1.dot(outs[0].flatten())
        hash_bits = "".join(["1" if it >= 0 else "0" for it in hash_output])
        # hash_hex = "{:0{}x}".format(int(hash_bits, 2), len(hash_bits) // 4)
        hash_bits_list.append(hash_bits)

    return hash_bits_list
