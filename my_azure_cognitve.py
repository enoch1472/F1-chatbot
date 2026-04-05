import cv2
import numpy as np
import os
import requests
from io import BytesIO
from PIL import Image, ImageDraw
try:
    from azure.cognitiveservices.vision.computervision import ComputerVisionClient
    from msrest.authentication import CognitiveServicesCredentials
    AZURE_ENABLED = True
except ImportError:
    AZURE_ENABLED = False

AZURE_KEY = 'BPf63DwMamB8E5h39saFI4b0Qc1bDuPJ5BPLdJ8uK1Cp2dCoZXH5JQQJ99BCACmepeSXJ3w3AAAFACOGUqNa'
AZURE_ENDPOINT = 'https://labcvresource.cognitiveservices.azure.com/'
def get_cascade_path(filename):
    import sys
    from glob import glob
    if hasattr(cv2, 'data') and hasattr(cv2.data, 'haarcascades'):
        path = os.path.join(cv2.data.haarcascades, filename)
        if os.path.isfile(path): return path
    try:
        cv2_dir = os.path.dirname(cv2.__file__)
        patterns = [
            os.path.join(cv2_dir, 'data', filename),
            os.path.join(cv2_dir, filename),
            os.path.join(sys.prefix, 'Library', 'etc', 'haarcascades', filename),                
            os.path.join(sys.prefix, 'share', 'opencv4', 'haarcascades', filename),        
        ]
        for p in patterns:
            if os.path.isfile(p): return p
    except:
        pass
    for folder in sys.path:
        if not folder or not os.path.isdir(folder): continue
        potential = glob(os.path.join(folder, 'cv2', 'data', filename))
        if potential: return potential[0]
        potential = glob(os.path.join(folder, 'cv2', filename))
        if potential: return potential[0]
    return filename
# Task C Extra: My Azure Cloud integration. 
# I am using Microsoft's Computer Vision API here to perform advanced 
# analysis like brand detection and natural language captioning.
def azure_analyze_image(image_source):
    if not AZURE_ENABLED:
        return "Azure Cognitive Services package not installed. Please 'pip install azure-cognitiveservices-vision-computervision'."
    
    try:
        import matplotlib.pyplot as plt
        client = ComputerVisionClient(AZURE_ENDPOINT, CognitiveServicesCredentials(AZURE_KEY))
        
        if image_source.startswith("http"):
            response = requests.get(image_source)
            img = Image.open(BytesIO(response.content))
        else:
            img = Image.open(image_source)

        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        w, h = img.size
        if w > 4200 or h > 4200:
            scale = 4200 / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
        
        img_buffer = BytesIO()
        img.save(img_buffer, format="JPEG", quality=85)
        
        for _ in range(3):
            if img_buffer.tell() <= 4000000:
                break
            quality = int(85 * (4000000 / img_buffer.tell()))
            img_buffer = BytesIO()
            img.save(img_buffer, format="JPEG", quality=quality)

        img_buffer.seek(0)
        description = client.describe_image_in_stream(img_buffer)
        
        img_buffer.seek(0)
        analysis = client.analyze_image_in_stream(img_buffer, visual_features=['Tags', 'Objects', 'Brands'])
        
        caption_text = ""
        if not description.captions:
            caption_text = "No caption detected"
        else:
            for caption in description.captions:
                caption_text += f"'{caption.text}'\n(Confidence: {caption.confidence * 100:.2f}%)\n"

        res = "[Azure F1 Analysis Mode]\n" + caption_text
        
        tags_list = [tag.name for tag in analysis.tags[:5]]
        if tags_list:
            res += f"Tags: {', '.join(tags_list)}\n"
            
        brands = [brand.name for brand in analysis.brands]
        if brands:
            res += f"Detected Sponsors/Brands: {', '.join(brands)}\n"

        fig = plt.figure(figsize=(16, 8))
        a = fig.add_subplot(1, 2, 1)

        draw = ImageDraw.Draw(img)
        if analysis.objects:
            for obj in analysis.objects:
                r = obj.rectangle
                bounding_box = ((r.x, r.y), (r.x + r.w, r.y + r.h))
                draw.rectangle(bounding_box, outline='magenta', width=5)
                plt.annotate(obj.object_property, (r.x, r.y), backgroundcolor='magenta')

        plt.axis("off")
        plt.imshow(img)
        plt.title(caption_text.strip())

        a = fig.add_subplot(1, 2, 2)

        tags = "Tags:"
        for tag in analysis.tags:
            tags += f"\n - {tag.name}"

        a.text(0, 0.4, tags, fontsize=12)
        plt.axis("off")
        plt.show(block=False)
        plt.pause(2)
            
        return res
    except Exception as e:
        return f"Error connecting to Azure: {e}"