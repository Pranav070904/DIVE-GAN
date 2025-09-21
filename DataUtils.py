import PIL
from PIL import Image
import os
import torch
import torchvision
from torchvision import transforms
from image_enhance import Enhancer
#Custom dataloader object that takes in batch size and train-test split size upon init
#when called will create a test set and trainset
#get item returns a batch of  "IMAGES"

class Loader:
    def __init__(self,batch_size=64,train_split = .8):
        self.batch_size = batch_size
        self.train_split = train_split
        self.test_set = None
        self.train_set = None
    
    def __call__(self,raw_dir,ref_dir):
        raw_paths = sorted([os.path.join(raw_dir,fname) for fname in os.listdir(raw_dir) if fname.endswith((".jpg",".png"))])
        ref_paths = sorted([os.path.join(ref_dir,fname) for fname in os.listdir(ref_dir) if fname.endswith((".jpg",".png"))])
        ###train-test split
        
        train_raw = raw_paths[:int(self.train_split*len(raw_paths))] 
        test_raw = raw_paths[int(self.train_split*len(raw_paths)):]
        train_ref = ref_paths[:int(self.train_split*len(ref_paths))]
        test_ref = ref_paths[int(self.train_split*len(ref_paths)):]
        #print(len(train_raw),len(test_raw))

        train = []
        batch = []
        for raw,ref in zip(train_raw,train_ref):
            batch.append([raw,ref])
            if len(batch) == self.batch_size:
                train.append(batch)
                batch = []
        
        self.train_set = train

        batch = []
        test = []
        for raw,ref in zip(test_raw,test_ref):
            batch.append([raw,ref])
            if(len(batch) == self.batch_size):
                test.append(batch)
                batch = []

        self.test_set = test

    def get_item(self,idx,train=True):
        dataset = self.train_set if train else self.test_set
        batch = dataset[idx]

        raw_image = []
        ref_image = []
        
        for raw_path,ref_path in batch:
            raw = Image.open(raw_path).convert("RGB")
            ref = Image.open(ref_path).convert("RGB")
            raw_image.append(raw)
            ref_image.append(ref)

        return raw_image,ref_image
    

class ImageToTensor:
    def __init__(self,mode="gan",size=(256,256)):
        '''gan = [-1,1] vgg for ImageNet normalization'''

        self.mode = mode
        self.size = size


        if mode == "gan":
            self.transform = transforms.Compose([transforms.Resize(size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)])
            
        elif mode == "vgg":
            self.transform = transforms.Compose([
                transforms.Resize(size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        elif(self.mode == "none"):
            self.transform = transforms.Compose([
                transforms.Resize(size),
                transforms.ToTensor()
            ])
        else:
            raise ValueError("Invalid mode. Use 'gan', 'vgg', or 'none'.")
        

    def __call__(self,image):
        return (self.transform(image))
    

class PrepInput:
    def __init__(self,device):
        self.tObject = ImageToTensor()
        self.enhance = Enhancer()
        self.dev = device
    

    def prepare_inputs(self,raw,ref,en1,en2,en3):
        raw_stack = torch.stack(raw)
        ref_stack = torch.stack(ref)
        en1_stack = torch.stack(en1)
        en2_stack = torch.stack(en2)
        en3_stack = torch.stack(en3)
        combined = torch.cat([raw_stack,en1_stack,en2_stack,en3_stack],dim=1)
        ref_stack = torch.stack(ref)
        pair_1 = torch.cat([raw_stack,en1_stack],dim=1)
        pair_2 = torch.cat([raw_stack,en2_stack],dim=1)
        pair_3 = torch.cat([raw_stack,en3_stack],dim=1)

        return combined.to(self.dev),raw_stack.to(self.dev),ref_stack.to(self.dev),pair_1.to(self.dev),pair_2.to(self.dev),pair_3.to(self.dev)
    
    def get_batch(self,raw,ref):
        raw_batch = []
        ref_batch = []
        en1_batch = []
        en2_batch = []
        en3_batch = []
        for x,y in zip(raw,ref):
            raw_batch.append(self.tObject(x))
            ref_batch.append(self.tObject(y))
            i1,i2,i3 = self.enhance.apply_all(x,gamma=.2)
            en1_batch.append(self.tObject(i1))
            en2_batch.append(self.tObject(i2))
            en3_batch.append(self.tObject(i3))
        
        return self.prepare_inputs(raw_batch,ref_batch,en1_batch,en2_batch,en3_batch)
    
    def __call__(self, raw,refs):
        return self.get_batch(raw,refs) # Output =  combined , raw,ref, pair1,pair2,pair3
        







