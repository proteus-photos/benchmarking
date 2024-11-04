from PIL import Image, ImageEnhance, ImageFilter
import io
import random

class Transformer:
    def __init__(self):
        pass

    def transform(self, image, method, **kwargs):
        if method=="screenshot":
            return self.transform(self.transform(image, "crop"), 'jpeg')
    
        elif method == 'jpeg':
            quality = kwargs.get('quality', 30)

            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='JPEG', quality=quality)
            img_byte_arr = img_byte_arr.getvalue()
            
            return Image.open(io.BytesIO(img_byte_arr))

        elif method == 'crop':
            width, height = image.size

            # upto 20% crop by default, sum of both sides
            # horizontal = random() * 0.2
            # vertical = random() * 0.2

            # left = horizontal * random()
            # top = vertical * random()
            # right = 1 - (horizontal - left)
            # bottom = 1 - (vertical - top)

            left = 0.2
            top = 0.2
            right = 0.8
            bottom = 0.8

            left = kwargs.get('left', left) * width
            top = kwargs.get('top', top)    * height
            right = kwargs.get('right', right)   * width
            bottom = kwargs.get('bottom', bottom)* height

            return image.crop((left, top, right, bottom))
        
        elif method=="brightness":
            factor = random.uniform(0.7, 1.3)
            enhancer = ImageEnhance.Brightness(image)
            return enhancer.enhance(factor)
            
        elif method=="contrast":
            factor = random.uniform(0.7, 1.3)
            enhancer = ImageEnhance.Contrast(image)
            return enhancer.enhance(factor)
        
        elif method=="blur":
            radius = kwargs.get('radius', 2)
            return image.filter(ImageFilter.GaussianBlur(radius=radius))
        
        elif method=="median":
            size = kwargs.get('size', 3)
            return image.filter(ImageFilter.MedianFilter(size=size))
        
        elif method == "double screenshot":
            return self.transform(self.transform(image, "screenshot"), "screenshot")
        else:
            raise ValueError('Invalid method')