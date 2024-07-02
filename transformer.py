from PIL import Image
import io

class Transformer:
    def __init__(self):
        pass
    def transform(self, image, method, **kwargs):
        if method == 'jpeg':
            quality = kwargs.get('quality', 100)

            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='JPEG', quality=quality)
            img_byte_arr = img_byte_arr.getvalue()
            
            return Image.open(io.BytesIO(img_byte_arr))

        elif method == 'crop':
            width, height = image.size
            side_length = min(width, height)

            # centered crop by default
            left = 1 - side_length / (2 * width)
            top = 1 - side_length / (2 * height)
            right = 1 - left
            bottom = 1 - top

            # override defaults if provided
            # the arguments provided are taken as percentages (fractions)

            left = kwargs.get('left', left)
            top = kwargs.get('top', top)
            right = kwargs.get('right', right)
            bottom = kwargs.get('bottom', bottom)

            left = round(left*width)
            top = round(top*height)
            right = round(right*width)
            bottom = round(bottom*height)

            return image.crop((left, top, right, bottom))
        
        elif method=="screenshot":
            width, height = image.size
            left = width * 0.1
            top = height * 0.1
            right = width * 0.9
            bottom = height * 0.9

            cropped_image = image.crop((left, top, right, bottom))


            return self.transform(cropped_image, 'jpeg')
        elif method == "double screenshot":
            return self.transform(self.transform(image, "screenshot"), "screenshot")
        else:
            raise ValueError('Invalid method')