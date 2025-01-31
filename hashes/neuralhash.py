import onnxruntime
import numpy as np
import torch

seed1 = open("./hashes/neuralhash_128x96_seed1.dat", "rb").read()[128:]
seed1 = np.frombuffer(seed1, dtype=np.float32)
seed1 = seed1.reshape([96, 128])

preprocess = lambda x: np.array(x.convert("RGB").resize([360, 360])).astype(np.float32) / 255.0

# Load ONNX model

def neuralhash(ims, bits=128, *args, **kwargs):
    global session
    # Load output hash matrix
    
    image_arrays = []
    for im in ims:
        if isinstance(im, np.ndarray):
            arr = im
        elif isinstance(im, torch.Tensor):
            arr = im.numpy()
        else:
            arr = preprocess(im)
        image_arrays.append((arr * 2.0 - 1.0).transpose(2, 0, 1))

    arr = np.stack(image_arrays, axis=0)

    inputs = {session.get_inputs()[0].name: arr}
    outs = np.array(session.run(None, inputs))[0].squeeze((2,3))
    hash_output = outs @ seed1[:bits].T
    hash_bits = hash_output >= 0
    return hash_bits


so = onnxruntime.SessionOptions()
so.log_severity_level = 3
session = onnxruntime.InferenceSession("./hashes/model.onnx", so)