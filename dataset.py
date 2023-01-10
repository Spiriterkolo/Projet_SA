from utils.sbi import SBI_Dataset
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import argparse
import random
from utils.funcs import load_json
import albumentations as alb
import cv2

from utils.funcs import IoUfrom2bboxes,crop_face,RandomDownScale

class SeeAbleDataset(SBI_Dataset):
    def __init__(self, phase='train', image_size=224, n_frames=8,n_loc=16):
        super().__init__(phase, image_size, n_frames)
        self.inv_transforms=self.get_inv_transforms()
        self.spatial_transforms = self.get_spatial_transforms()
        self.frequential_transforms = self.get_frequential_transforms()
        self.n_loc = n_loc
    
    def __getitem__(self,idx):
        flag=True
        while flag:
            try:
                filename=self.image_list[idx]
                img=np.array(Image.open(filename))
                landmark=np.load(filename.replace('.png','.npy').replace('/frames/',self.path_lm))[0]
                bbox_lm=np.array([landmark[:,0].min(),landmark[:,1].min(),landmark[:,0].max(),landmark[:,1].max()])
                bboxes=np.load(filename.replace('.png','.npy').replace('/frames/','/retina/'))[:2]
                iou_max=-1
                for i in range(len(bboxes)):
                    iou=IoUfrom2bboxes(bbox_lm,bboxes[i].flatten())
                    if iou_max<iou:
                        bbox=bboxes[i]
                        iou_max=iou

                landmark=self.reorder_landmark(landmark)
                if self.phase=='train':
                    if np.random.rand()<0.5:
                        img,_,landmark,bbox=self.hflip(img,None,landmark,bbox)
                        
                img,_,__,___,y0_new,y1_new,x0_new,x1_new=crop_face(img,landmark,bbox,margin=False,crop_by_bbox=True,abs_coord=True,phase=self.phase)
                img=cv2.resize(img,self.image_size,interpolation=cv2.INTER_LINEAR).astype('float32')/255
                img = self.generate_soft_discrepancies(img)
                flag=False
            except Exception as e:
                print(e)
                idx=torch.randint(low=0,high=len(self),size=(1,)).item()

        return img
    
    def mask_maker(self,size,indice):
        if indice > self.n_loc - 1:
            print("Mask index out of range")
            return np.zeros(size)
        mask = np.zeros(size)
        indice_i = indice // np.sqrt(self.n_loc)
        indice_j = indice % np.sqrt(self.n_loc)
        offset = int(size[0]/np.sqrt(self.n_loc))
        start_i = offset * int(indice_i)
        start_j = offset * int(indice_j)
        mask[start_i:start_i+offset, start_j:start_j+offset] = np.ones(np.shape(mask[start_i:start_i+offset, start_j:start_j+offset]))
        return torch.tensor(mask)
    
    def collate_fn(self,batch):
        data={}
        data['img']=torch.cat([batch[i].float() for i in range(len(batch))],0)
        data['label']=torch.cat([torch.arange(0,batch[0].shape[0],1) for _ in range(len(batch))],0)
        return data
    
    def generate_soft_discrepancies(self,img):
        #print(img.shape)
        x = [torch.tensor(img)]
        for i in range(2*self.n_loc):
            target = self.inv_transforms(image=img)['image']
            target = torch.tensor(target)
            source = self.spatial_transforms(image=img)['image'] if i%2==0 else self.frequential_transforms(image=img)['image']
            source = torch.tensor(source)
            mask = torch.stack([self.mask_maker(img.shape[:2],i//2) for _ in range(img.shape[2])])
            #mask = torch.stack([mask for _ in range(img.shape[0])])
            mask = mask.permute((1,2,0))
            """print(mask.shape)
            print(source.shape)
            print(target.shape)"""
            x.append(torch.mul(mask,source) + torch.mul((1-mask),target))
        return(torch.stack(x))
    
    def get_inv_transforms(self):
        return (alb.Compose([
                alb.Affine(translate_percent=(0.015,0.03),p=0.3),
                #alb.RandomScale(0.80,p=0.3),
                alb.HueSaturationValue(hue_shift_limit=(-0.1,0.1), sat_shift_limit=(-0.1,0.1), val_shift_limit=(-0.1,0.1), p=0.3),
                ],
                additional_targets={f'image1': 'image'},
                p=1.))
	
    def get_spatial_transforms(self):
        return alb.Compose([

                alb.RGBShift((-20,20),(-20,20),(-20,20),p=0.3),
                alb.HueSaturationValue(hue_shift_limit=(-0.3,0.3), sat_shift_limit=(-0.3,0.3), val_shift_limit=(-0.3,0.3), p=0.3),
                alb.RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1,0.1), p=0.3),

                ], 
                additional_targets={f'image1': 'image'},
                p=1.)
	
    def get_frequential_transforms(self):
        return alb.OneOf([
                alb.Downscale(0.25,0.5,p=1),
                alb.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1),
                alb.ImageCompression(quality_lower=30,quality_upper=70,p=1)
                ],p=1)


def main(args):
    cfg=load_json(args.config)
    seed=5
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device('cuda')


    image_size=cfg['image_size']
    batch_size=cfg['batch_size']
    number_of_blocks = cfg['n_loc']
    trainset = SeeAbleDataset('train')

    for k,data in enumerate(trainset):
        print(k,data.shape)
        if k>=9:
            break


if __name__=='__main__':


    parser=argparse.ArgumentParser()
    parser.add_argument(dest='config')
    args=parser.parse_args()
    main(args)