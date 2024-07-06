from scipy.optimize import minimize, fmin_cobyla
from neuralhash.neuralhash import neuralhash
from PIL import Image
from transformer import Transformer
from utils import match

t = Transformer()

def difference(hash1, hash2):
    return - match(hash1, hash2)

def minimize_function(x, target_hash, image2, hasher):
    left = x[0] #x[0] / (1 + x[0] + x[2])
    top = x[1] #x[1] / (1 + x[1] + x[3])
    right = x[2] #1 - x[2] / (1 + x[0] + x[2])
    bottom = x[3] #1 - x[3] / (1 + x[1] + x[3])

    print(left, top, right, bottom)
    diff = difference(target_hash, hasher([t.transform(image2, method='crop', left=left, top=top, right=right, bottom=bottom)])[0])
    print(diff)
    return diff

epsilon = 5e-3
def constraint_horizontal(x, *args):
    return -x[0] + x[2] - epsilon

def constraint_vertical(x, *args):
    return -x[1] + x[3] - epsilon

def constraint_left1(x, *args):
    return x[0]

def constraint_left2(x, *args):
    return 1 - x[0]

def constraint_top1(x, *args):
    return x[1]

def constraint_top2(x, *args):
    return 1 - x[1]

def constraint_right1(x, *args):
    return x[2]

def constraint_right2(x, *args):
    return 1 - x[2]

def constraint_bottom1(x, *args):
    return x[3]

def constraint_bottom2(x, *args):
    return 1 - x[3]

class Matcher:
    def __init__(self, hasher):
        self.hasher = hasher

    def match(self, image1, image2):
        """
        Applies crop transformations to image1 to try and match image2 using Powell's method
        """

        # constraints = [{'type': 'ineq', 'fun': ineq} for ineq in [constraint_horizontal, constraint_vertical, constraint_left1, constraint_left2, constraint_top1, constraint_top2, constraint_right1, constraint_right2, constraint_bottom1, constraint_bottom2]]

        target_hash = self.hasher([image2])[0]

        x = fmin_cobyla(
            minimize_function,
            x0=(0.05, 0.05, 0.95, 0.95),
            # method='COBYLA',
            # bounds=[(0, 1), (0, 1), (0, 1), (0, 1)],
            args=(target_hash, image1, hasher),
            rhobeg=0.05,
            cons=[constraint_horizontal, constraint_vertical, constraint_left1, constraint_left2, constraint_top1, constraint_top2, constraint_right1, constraint_right2, constraint_bottom1, constraint_bottom2]
        )

        # left = x[0] / (1 + x[0] + x[2])
        # top = x[1] / (1 + x[1] + x[3])
        # right = 1 - x[2] / (1 + x[0] + x[2])
        # bottom = 1 - x[3] / (1 + x[1] + x[3])

        return x # left, top, right, bottom
    
if __name__ == "__main__":
    hasher = neuralhash
    matcher = Matcher(hasher)

    image1 = Image.open('lenna.png')
    image2 = t.transform(image1, 'crop', left=0.1, top=0.1, right=0.9, bottom=0.9)

    params = matcher.match(image1, image2)
    print(params)
    print(difference(hasher([image1])[0], hasher([t.transform(image2, method='crop', left=params[0], top=params[1], right=params[2], bottom=params[3])])[0]))