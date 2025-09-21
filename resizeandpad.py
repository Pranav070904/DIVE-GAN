import PIL
from PIL import Image

#Resizes and pads an image to required size.
#specify size (w,h) upon init
#pass PIL image when called
class ResizeAndPad:
    def __init__(self,size):
        self.size = size #(w,h)
    def __call__(self,img):
        w,h = img.size
        target_w,target_h = self.size

        scale = min(target_w/w,target_h/h)
        new_w,new_h = int(w*scale),int(h*scale)
        img = img.resize((new_w,new_h),Image.BICUBIC)

        new_image = Image.new("RGB",(target_w,target_h),(0,0,0)) #black padding
        paste_x = (target_w-new_w)//2
        paste_y = (target_h-new_h)//2

        new_image.paste(img,(paste_x,paste_y))

        return new_image
