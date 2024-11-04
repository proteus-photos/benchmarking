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
# ax.set_yscale('log')
ax.set_ylim(0.15, 1)
ax.set_xlim(1e-15, None)

blur_df = pd.read_csv("results/dinohash_blur.csv")
fpr = blur_df["fpr"]
tpr = blur_df["tpr"]

ax.plot(fpr, tpr, label="blur")

bright_df = pd.read_csv("results/dinohash_brightness.csv")
fpr = bright_df["fpr"]
tpr = bright_df["tpr"]

ax.plot(fpr, tpr, label="bright", linestyle=(0,(5,10)))

contrast_df = pd.read_csv("results/dinohash_contrast.csv")
fpr = contrast_df["fpr"]
tpr = contrast_df["tpr"]

ax.plot(fpr, tpr, label="contrast", linestyle=(7.5,(5,10)))

median_df = pd.read_csv("results/dinohash_median.csv")
fpr = median_df["fpr"]
tpr = median_df["tpr"]

ax.plot(fpr, tpr, label="median")


plt.legend()
plt.savefig(f"./results/multi.png")
