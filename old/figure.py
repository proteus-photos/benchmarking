from PIL import Image
from transformer import Transformer

t = Transformer()
image = Image.open("explosion.jpeg").resize((256, 256))
transformations = ["blur", "brightness", "contrast", "median"]
image.save("explosion_small.jpg")
for transformation in transformations:
    newimage = t.transform(t.transform(image, "screenshot"), transformation)
    newimage.save(f"explosion_{transformation}.jpg")