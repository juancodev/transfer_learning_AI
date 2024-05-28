import os
from transformers import pipeline

"""
 -------------Comments-------------------------------------------------------
| #pipeline = Nos sirve para inferir en la mejor opción                      |
||
|
|
|
"""

# image classification

image_classification = pipeline(
    task="image-classification",
    model="google/vit-base-patch16-224",
    token="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",
)

dir_base = os.path.dirname(__file__)
image_url = os.path.join(dir_base, "assets/image.jpeg")

print(image_classification(image_url))
