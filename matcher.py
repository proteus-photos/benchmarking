from scipy.optimize import minimize
from neuralhash.neuralhash import neuralhash
from PIL import Image
from transformer import Transformer

t = Transformer()

def difference(hash1, hash2):
    return 1 - sum(c1 == c2 for c1, c2 in zip(hash1, hash2)) / len(hash1)

def minimize_function(x, target_hash, image2, hasher):
    left = x[0] / (1 + x[0] + x[2])
    top = x[1] / (1 + x[1] + x[3])
    right = 1 - x[2] / (1 + x[0] + x[2])
    bottom = 1 - x[3] / (1 + x[1] + x[3])

    diff = difference(target_hash, hasher([t.transform(image2, method='crop', left=left, top=top, right=right, bottom=bottom)])[0])
    print(diff)
    return diff

class Matcher:
    def __init__(self, hasher):
        self.hasher = hasher

    def match(self, image1, image2):
        """
        Applies crop transformations to image1 to try and match image2 using Powell's method
        """
        target_hash = self.hasher([image2])[0]
        match = minimize(
            minimize_function,
            x0=(0.5, 0.5, 0.5, 0.5),
            method='powell',
            bounds=[(0, 1), (0, 1), (0, 1), (0, 1)],
            args=(target_hash, image1, hasher),
        )
        x = match.x
        left = x[0] / (1 + x[0] + x[2])
        top = x[1] / (1 + x[1] + x[3])
        right = 1 - x[2] / (1 + x[0] + x[2])
        bottom = 1 - x[3] / (1 + x[1] + x[3])

        return left, top, right, bottom
    
if __name__ == "__main__":
    hasher = neuralhash
    matcher = Matcher(hasher)

    image1 = Image.open('lenna.png')
    image2 = t.transform(image1, 'crop', left=0.1, top=0.1, right=0.9, bottom=0.9)

    params = matcher.match(image1, image2)
    print(params)
    print(difference(hasher([image1])[0], hasher([t.transform(image2, method='crop', left=params[0], top=params[1], right=params[2], bottom=params[3])])[0]))