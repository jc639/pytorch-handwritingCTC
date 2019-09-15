# transforms
# Transform and Data Augmentation
from skimage import transform, color, filters
import cv2 as cv
import numpy as np
import torch
from torchvision.transforms import Normalize

class Rescale(object):
    
    def __init__(self, output_size, random_pad=False, border_pad=(0, 0), random_rotation=0, random_stretch=1,
                 fill_space=False, fill_threshold=200):
        assert isinstance(output_size, (tuple))
        assert isinstance(random_pad, bool)
        assert isinstance(border_pad, (tuple))
        assert isinstance(random_rotation, (float, int))
        assert isinstance(random_stretch, float)
        assert isinstance(fill_space, bool)
        assert isinstance(fill_threshold, int) and 0 <= fill_threshold < 255
        self.output_size = output_size
        self.random_pad = random_pad
        self.border_pad = border_pad
        self.rotation = random_rotation
        self.random_stretch = random_stretch
        self.fill_space = fill_space
        self.fill_threshold = fill_threshold

    def __call__(self, sample):
        
        image, word = sample['image'], sample['word']
        if self.fill_space:
            image[image < self.fill_threshold] = 255
        
        if self.border_pad[0] > 0 or self.border_pad[1] > 0:
            resize = (self.output_size[0] - self.border_pad[0], self.output_size[1] - self.border_pad[1])
        else:
            resize = self.output_size
            
        h, w = image.shape[:2]
        fx = w / resize[1]
        fy = h / resize[0]
        
        f = max(fx, fy)
        
        new_size = (max(min(resize[0], int(h / f)), 1), max(min(resize[1], int(w / f * self.random_stretch)), 1))
        
        image = transform.resize(image, new_size, preserve_range=True, mode='constant', cval=255)
        if self.rotation != 0:
            rot = np.random.choice(np.arange(-self.rotation, self.rotation), 1)
            image = transform.rotate(image, rot, mode='constant', cval=255, preserve_range=True)
        
        canvas = np.ones(self.output_size, dtype=np.uint8) * 255
        
        if self.random_pad:
            v_pad_max = self.output_size[0] - new_size[0] 
            h_pad_max = self.output_size[1] - new_size[1]
            
            v_pad = int(np.random.choice(np.arange(0, v_pad_max + 1), 1))
            h_pad = int(np.random.choice(np.arange(0, h_pad_max + 1), 1))
            
            canvas[v_pad:v_pad + new_size[0], h_pad:h_pad + new_size[1]] = image            
        else:
            canvas[0:new_size[0], 0:new_size[1]] = image
         
        # rotate adds extra column
        canvas = transform.rotate(canvas, -90, resize=True)[:, :-1]

                    
        return {'image': canvas, 'word': word}
    
class Deslant(object):
    """Deslant handwriting samples"""
    
    def __call__(self, sample):
        image, word = sample['image'], sample['word']
        
        try:
            threshold = filters.threshold_otsu(image)
        except ValueError:
            return {'image':image, 'word':word}
        
        binary = image.copy() < threshold
        
        # array of alpha values
        alphas = np.arange(-1, 1.1, 0.25)
        alpha_res = np.array([])
        alpha_params = []
        
        for a in alphas:
            alpha_sum = 0
            shift_x = np.max([-a*binary.shape[0], 0])
            M = np.array([[1, a, shift_x],
                          [0,1,0]], dtype=np.float64)
            img_size = (np.int(binary.shape[1] + np.ceil(np.abs(a*binary.shape[0]))), binary.shape[0])
            alpha_params.append((M, img_size))
            
            
            img_shear = cv.warpAffine(src=binary.astype(np.uint8),
                                      M=M, dsize=img_size, 
                                      flags=cv.INTER_NEAREST)
            
            for i in range(0, img_shear.shape[1]):
                if not np.any(img_shear[:, i]):
                    continue
                
                h_alpha = np.sum(img_shear[:, i])
                fgr_pos = np.where(img_shear[:, i] == 1)
                delta_y_alpha = fgr_pos[0][-1] - fgr_pos[0][0] + 1
                
                if h_alpha == delta_y_alpha:
                    alpha_sum += h_alpha**2
                
            alpha_res = np.append(alpha_res, alpha_sum)
            
        best_M, best_size = alpha_params[alpha_res.argmax()]
        deslanted_img = cv.warpAffine(src=image, M=best_M, dsize=best_size,
                                      flags=cv.INTER_LINEAR, 
                                      borderMode=cv.BORDER_CONSTANT,
                                      borderValue=255)
        
        return {'image':deslanted_img, 'word':word}
    
class toRGB(object):
    """Convert the ndarrys to RGB tensors.
       Required if using ImageNet pretrained Resnet."""
       
    def __call__(self, sample):
        image, word = sample['image'], sample['word']
        image = color.grey2rgb(image)
        
        return {'image': image, 'word': word}
        
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    
    def __init__(self, rgb=True):
        assert isinstance(rgb, bool)
        self.rgb = rgb

    def __call__(self, sample):
        image, word = sample['image'], sample['word']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        if self.rgb:
            image = torch.from_numpy(image.transpose((2, 0, 1))).float()
        else:
            image = torch.from_numpy(image)[None, :, :].float()
        return {'image': image,
                'word': word}
    
class Normalise(object):
    """Normalise by channel mean and std"""
    
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean, dtype=torch.float)
        self.std = torch.tensor(std, dtype=torch.float)
        self.norm = Normalize(mean, std)
    
    def __call__(self, sample):
        image, word = sample['image'], sample['word']
        return {'image': self.norm(image),
                'word': word}