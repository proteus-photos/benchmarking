from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from PIL import Image
from transformer import Transformer

# t = Transformer()
# image = Image.open("test.jpg")
# image = t.transform(t.transform(image, "screenshot"), "blur")
# image.save("test_transformed.jpg")

transformations = "Blur", "Brightness", "Contrast", "Median"
methods = "DinoHash", "NeuralHash", "Stable Signature"
colors = "green", "blue", "orange"
styles = "dotted", (0,(1,14)), (9,(5,10)), "solid"
fig, ax = plt.subplots()

ax.set_xscale('log')
# ax.set_yscale('log')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
ax.set_title('ROC Curve')

ax.set_ylim(0.0, 1)
ax.set_xlim(1e-16, 3)

for t, s in zip(transformations, styles):
    for m, c in zip(methods, colors):
        df = pd.read_csv(f"results/{m.lower()}_{t.lower()}.csv")
        fpr = df["fpr"]
        tpr = df["tpr"]
        plt.plot(fpr, tpr, color=c, linestyle=s)


for m, c in zip(methods, colors):
    ax.plot(np.NaN, np.NaN, c=c, label=m)

ax2 = ax.twinx()
for t, s in zip(transformations, styles):
    ax2.plot(np.NaN, np.NaN, ls=s, label=t, c='black')
ax2.get_yaxis().set_visible(False)

ax.legend(loc=0)
ax2.legend(loc=4)

# plt.legend()
plt.tight_layout()
plt.savefig(f"./results/multi.png")
