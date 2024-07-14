from PIL import Image
import io

class Transformer:
    def __init__(self):
        pass
    def transform(self, image, method, **kwargs):
        if method == 'jpeg':
            quality = kwargs.get('quality', 95)

            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='JPEG', quality=quality)
            img_byte_arr = img_byte_arr.getvalue()
            
            return Image.open(io.BytesIO(img_byte_arr))

        elif method == 'crop':
            width, height = image.size

            # 10% crop by default
            left = 0.1
            top = 0.1
            right = 0.9
            bottom = 0.9

            # override defaults if provided
            # the arguments provided are taken as percentages (fractions)

            left = kwargs.get('left', left) * width
            top = kwargs.get('top', top)    * height
            right = kwargs.get('right', right)   * width
            bottom = kwargs.get('bottom', bottom)* height


            return image.crop((left, top, right, bottom))
        
        elif method=="screenshot":
            return self.transform(self.transform(image, "crop"), 'jpeg')
        elif method == "double screenshot":
            return self.transform(self.transform(image, "screenshot"), "screenshot")
        else:
            raise ValueError('Invalid method')