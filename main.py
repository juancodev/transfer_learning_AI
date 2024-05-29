import os
import timm
from transformers import pipeline

"""
 ------------------------------------------Comments--------------------------------
| - pipeline = Nos sirve para inferir en la mejor opción                           |
| - crear variable de entorno para nuestro token generado en hugging face          |
| -                                                                                |
| -                                                                                |
| -                                                                                |
| -                                                                                |
 ---------------------------------------------------------------------------------
"""

dir_base = os.path.dirname(__file__)


"""
# image classification

image_classification = pipeline(
    task="image-classification",
    model="google/vit-base-patch16-224",
    token=os.environ["TOKEN_HUGGING_FACE"],
)
image_url = os.path.join(dir_base, "assets/image.jpeg")
print(image_classification(image_url))
"""

"""
# image segmentation

image_segmentation = pipeline("image-segmentation")

image_url_2 = os.path.join(dir_base, "assets/expo.JPG")

print(image_segmentation(image_url_2))
"""

# Text classification

text_classification = pipeline(
    task="text-classification",
    model="lxyuan/distilbert-base-multilingual-cased-sentiments-student",
    token=os.environ["TOKEN_HUGGING_FACE"],
    return_all_scores=True,
)

print(text_classification("Me gusta mucho comer pizza con maiz"))
