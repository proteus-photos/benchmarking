from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from PIL import Image
from transformer import Transformer

t = Transformer()
image = Image.open("test.jpg")
image = t.transform(t.transform(image, "screenshot"), "blur")
image.save("test_transformed.jpg")

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_xscale('log')
# ax.set_xlim(1e-24, None)
dino_df = pd.read_csv("results/dinohash_jpeg.csv")
fpr = dino_df["fpr"]
tpr = dino_df["tpr"]


ax.plot(fpr, tpr, label="dino")

neural_df = pd.read_csv("results/neuralhash_jpeg.csv")
fpr = neural_df["fpr"]
tpr = neural_df["tpr"]

ax.plot(fpr, tpr, label="neural")

plt.legend()
plt.savefig(f"./results/multi.png")
