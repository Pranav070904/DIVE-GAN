import PIL
from PIL import Image
import cv2
import matplotlib.pyplot as plt


path = r"C:\Users\prana\Documents\AI Project\dataset\RAW\2_img_.png"

img = cv2.imread(path)
print(img.shape)

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # ✅ convert for matplotlib

plt.imshow(img)
plt.axis('off')
plt.title("Correct RGB Image")
plt.show()

