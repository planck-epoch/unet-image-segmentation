import numpy
import PIL
import io
import base64
import re

def as_image_from_array(image):
    return PIL.Image.fromarray(image)

def as_url(data, source='zc', size=None):
    img = data
    
    if size is not None:
        img = img.resize(size, resample=PIL.Image.BILINEAR)
    buffered = io.BytesIO()
    img.save(buffered, format='png')
    b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return 'data:image/png;base64,%s' % (b64)

def from_url(url, target='zc', size=None):
    image_data = re.sub('^data:image/.+;base64,', '', url)
    im = PIL.Image.open(io.BytesIO(base64.b64decode(image_data)))
    
    return im