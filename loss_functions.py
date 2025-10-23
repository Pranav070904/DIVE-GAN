import torch
import torch.nn as nn
from torchvision import models


'''Credit for Some functions:
    GitHub:xahidbuffon/FUnIE-GAN'''


#-----------------------------Single Layer Percept Loss version-------------------------------------------------#

class VGG19_PerceptLoss_Single(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features.to(device)
        for param in self.vgg.parameters():
            param.requires_grad_(False)

    def get_features(self, image, layers=None):
        if layers is None:
            layers = {'30': 'conv5_2'}
        features = {}
        x = image
        for name, layer in self.vgg._modules.items():
            x = layer(x)
            if name in layers:
                features[layers[name]] = x
        return features

    def forward(self, pred, true, layer='relu4_5'):
        true_f = self.get_features(true, layers={'29': 'relu4_5'})
        pred_f = self.get_features(pred, layers={'29': 'relu4_5'})
        return torch.mean((true_f[layer] - pred_f[layer]) ** 2)


class GANLoss_Single:
    def __init__(self, device="cpu", lambda_pixel=0.05, lambda_perc=1.0, lambda_adv=1.0):
        self.dev = device
        self.advLoss = nn.MSELoss()
        self.percept = VGG19_PerceptLoss_Single(self.dev)
        self.pixel_loss = nn.MSELoss()
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.lp = lambda_pixel
        self.lperc = lambda_perc
        self.la = lambda_adv

    def normalize(self, img, mean, std):
        img = (img + 1.0) / 2.0
        mean = torch.tensor(mean).view(1, 3, 1, 1).to(self.dev)
        std = torch.tensor(std).view(1, 3, 1, 1).to(self.dev)
        return (img - mean) / std

    def compute_gen_loss(self, fake, D_fake, target):
        # fake -> generator output
        # D_fake -> result obtained from discriminator
        # target -> ground truth
        loss_adv = self.advLoss(D_fake, torch.ones_like(D_fake))
        loss_pixel = self.pixel_loss(fake, target)
        imagenet_normalized_fake = self.normalize(fake, self.mean, self.std)
        imagenet_normalized_target = self.normalize(target, self.mean, self.std)
        loss_perc = self.percept(imagenet_normalized_fake, imagenet_normalized_target, layer='relu4_5')
        total_loss = (self.la * loss_adv) + (self.lp * loss_pixel) + (self.lperc * loss_perc)

        return total_loss,{
            'adv_loss': loss_adv.item(),
            'pixel_loss': loss_pixel.item(),
            'perc_loss': loss_perc.item()
        }

    def compute_disc_loss(self, pred_real, pred_fake):
        valid = torch.full_like(pred_real, .9, device=self.dev)
        fake = torch.full_like(pred_fake, .0, device=self.dev)
        loss_real = self.advLoss(pred_real, valid)
        loss_fake = self.advLoss(pred_fake, fake)

        lossD = 0.5 * (loss_real + loss_fake) * 10
        return lossD,{
            'real_loss': loss_real.item(),
            'fake_loss': loss_fake.item()
        }
    

#----------------------------------------------------Multi Layer Percept Loss Version--------------------------------------------------#
# Set percept loss = .25 
class VGG19_PerceptLoss_Multi(nn.Module):
    def __init__(self, device, layers=None):
        super().__init__()
        self.vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features.to(device)
        # Define the layers you want to use for the loss
        if layers is None:
            
            self.layers = {'8': 'relu2_2', '17': 'relu3_4', '26': 'relu4_4', '35': 'relu5_4'}
        else:
            self.layers = layers
            
        for param in self.vgg.parameters():
            param.requires_grad_(False)
        
        
        self.vgg.eval()
    
    def get_features(self, image):
        features = {}
        x = image
        # Iterate through the VGG layers
        for name, layer in self.vgg._modules.items():
            x = layer(x)
            
            if name in self.layers:
                features[self.layers[name]] = x
        return features
    
    def forward(self, pred, true):
        true_f = self.get_features(true)
        pred_f = self.get_features(pred)
        
        # Calculate the loss for each layer and sum them up
        loss = 0
        for key in true_f.keys():
            loss += torch.mean((true_f[key] - pred_f[key])**2)
            
        return loss

class GANLoss_Multi:
    def __init__(self, device="cpu", lambda_pixel=0.05, lambda_perc=1.0, lambda_adv=1.0):
        self.dev = device
        self.advLoss = nn.MSELoss()
        self.percept = VGG19_PerceptLoss_Multi(self.dev)
        self.pixel_loss = nn.MSELoss()
        # ImageNet normalization parameters
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.lp = lambda_pixel
        self.lperc = lambda_perc
        self.la = lambda_adv

    def normalize(self, img, mean, std):
        """
        Normalize image from [-1, 1] to ImageNet normalization
        """
        # Convert from [-1, 1] to [0, 1]
        img = (img + 1.0) / 2.0
        mean = torch.tensor(mean).view(1, 3, 1, 1).to(self.dev)
        std = torch.tensor(std).view(1, 3, 1, 1).to(self.dev)
        return (img - mean) / std

    def compute_gen_loss(self, fake, D_fake, target):
        """
        Compute generator loss
        Args:
            fake: generator output
            D_fake: discriminator output for fake images
            target: ground truth images
        """
       
        loss_adv = self.advLoss(D_fake, torch.ones_like(D_fake))
        
        # Pixel-wise loss
        loss_pixel = self.pixel_loss(fake, target)
        
        # Perceptual loss
        imagenet_normalized_fake = self.normalize(fake, self.mean, self.std)
        imagenet_normalized_target = self.normalize(target, self.mean, self.std)
        loss_perc = self.percept(imagenet_normalized_fake, imagenet_normalized_target)
        
        # Total loss
        total_loss = (self.la * loss_adv) + (self.lp * loss_pixel) + (self.lperc * loss_perc)

        return total_loss, {
            'adv_loss': loss_adv.item(),
            'pixel_loss': loss_pixel.item(),
            'perc_loss': loss_perc.item()
        }

    def compute_disc_loss(self, pred_real, pred_fake):
        """
        Compute discriminator loss
        Args:
            pred_real: discriminator output for real images
            pred_fake: discriminator output for fake images
        """
       
        valid = torch.full_like(pred_real, 0.9, device=self.dev)
        fake = torch.full_like(pred_fake, 0.0, device=self.dev)
        
        loss_real = self.advLoss(pred_real, valid)
        loss_fake = self.advLoss(pred_fake, fake)

        
        lossD = 0.5 * (loss_real + loss_fake)
        
        return lossD, {
            'real_loss': loss_real.item(),
            'fake_loss': loss_fake.item()
        }