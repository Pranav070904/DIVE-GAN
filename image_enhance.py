import cv2
import numpy as np
from PIL import Image



#Applies 3 rudementary image enhancements and returns 3 images.
class Enhancer:
    def __init__(self):
        pass
    
    def en1(self,img,gamma): #gamma correction
        inv_gamma = 1.0/gamma
        table = np.array([(i/255.0)**inv_gamma*255 for i in range(256)]).astype("uint8")
        return cv2.LUT(img,table)
    def en2(self,img):#HE
        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
        return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    
    def en3(self,img):#White balance
        result = cv2.xphoto.createSimpleWB().balanceWhite(img)
        return result
    
    def apply_all(self,img,gamma):
        img = np.array(img)
        img1 = self.en1(img,gamma)
        img2 = self.en2(img)
        img3 = self.en3(img)

        return Image.fromarray(img1),Image.fromarray(img2),Image.fromarray(img3)
        
    