from PIL import Image
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
import os
 
image_path = "object.png"
if not os.path.exists(image_path):
    print(f"Error: Image file '{image_path}' not found in the current folder.")
    exit()
img = Image.open(image_path).convert("L")  

img_array = np.array(img) 
threshold = 128  
binary_img = img_array < threshold  # True = object, False = background

labeled_array, num_features = ndimage.label(binary_img)

print(f"Number of objects detected: {num_features}")

plt.imshow(labeled_array, cmap='nipy_spectral')
plt.title(f"Detected Objects: {num_features}")
plt.axis('off')
plt.colorbar()
plt.show()

