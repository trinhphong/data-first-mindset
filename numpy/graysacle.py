import numpy as np
from PIL import Image
from pathlib import Path

# Load image and convert to NumPy array
img_path = Path("numpy/photo.jpg")
img = np.array(Image.open(img_path))

# Apply weights using array slicing
# img[:,:,0] is Red, img[:,:,1] is Green, img[:,:,2] is Blue
grayscale = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114

# Convert back to an image object
final_img = Image.fromarray(grayscale.astype(np.uint8))
final_img.save("numpy/grayscale_photo.jpg")
