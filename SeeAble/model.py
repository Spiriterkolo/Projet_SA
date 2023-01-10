import torch
import torch.nn as nn
import numpy as np
import torchvision.models as models
from dataset import SeeAbleDataset
import argparse
from utils.funcs import load_json
import random
import matplotlib.pyplot as plt
import albumentations as alb
from tqdm import tqdm
from efficientnet_pytorch import EfficientNet

class SeeAble(nn.Module):
    def __init__(self,n_loc,device,loss_type="CE",L2weight=0.5):
        super().__init__()

        losses = {"CE":nn.CrossEntropyLoss()}
        out_features = {"CE":2*n_loc+1}
        self.prototypes = self.get_prototypes(2*n_loc + 1)
        self.n_loc = n_loc
        self.encoder = EfficientNet.from_name('efficientnet-b4')
        self.projector = nn.Linear(in_features= 1000,out_features=out_features[loss_type])

        self.optims = [torch.optim.SGD(params=self.encoder.parameters(),lr=0.5),
                        torch.optim.SGD(params=self.projector.parameters(),lr=0.5,weight_decay=L2weight)]
        self.schedulers = [torch.optim.lr_scheduler.CosineAnnealingLR(optim,100) for optim in self.optims]
        self.loss = losses[loss_type]


        self.inv_transforms=self.get_inv_transforms()
        self.spatial_transforms = self.get_spatial_transforms()
        self.frequential_transforms = self.get_frequential_transforms()
        self.device = device
    def get_prototypes(self,n_dim):
        vertices = np.eye(n_dim)
        vertices = np.concatenate((vertices,((1+np.sqrt(n_dim+1))/n_dim)*np.ones((1,n_dim))),axis=0)
        vertices = (vertices - np.mean(vertices,axis=0))
        vertices = vertices/np.linalg.norm(vertices[0])
        return torch.tensor(vertices)

    def forward(self,input):
        input = input.permute((0,3,1,2))
        torch.cuda.empty_cache()
        #print(torch.cuda.memory_summary(device=None, abbreviated=True))
        input = input.to(self.device)
        self.encoder.to(self.device)
        #print(torch.cuda.memory_summary(device=None, abbreviated=True))
        embedding = self.encoder(input)
        self.encoder.cpu()
        input.cpu()
        #print(torch.cuda.memory_summary(device=None, abbreviated=True))
        embedding.to(self.device)
        self.projector.to(self.device)
        output = self.projector(embedding)
        embedding.cpu()
        self.projector.cpu()
        output = output/torch.linalg.norm(output,dim=(1,2,3),keepdim=True)
        return output
        

    def training_step(self, batch, batch_idx):
        x = batch['img']
        y = batch['label']
        predictions = self(x)
        loss = self.loss(predictions,y)
        #self.log('train_loss', loss)
        return loss
        
    def validation_step(self, batch, batch_idx):
        x,y = batch
        predictions = self.forward(x)
        loss = self.loss(predictions,y)
        acc = self.accuracy(predictions,y)
        
        self.log('val_acc', acc)
        self.log('val_loss', loss)

    def test_step(self, batch, batch_idx):
        
        x,y = batch
        predictions = self.forward(x)
        loss = self.loss(predictions,y)
        acc = self.accuracy(predictions,y)
        self.acc = self.acc + acc # We accumulate every accuracy
        
        self.log('test_loss', loss)
        self.log('test_acc', acc)

    def get_inv_transforms(self):
        return (alb.Compose([
                alb.Affine(translate_percent=(0.015,0.03),p=0.3),
                alb.RandomScale(5,p=0.3),
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

    epochs = cfg['epochs']
    image_size=cfg['image_size']
    batch_size=cfg['batch_size']
    number_of_blocks = cfg['n_loc']
    loss_type = cfg['loss_type']
    train_dataset=SeeAbleDataset(phase='train',image_size=image_size,n_loc=number_of_blocks)
    #val_dataset=SeeAbleDataset(phase='val',image_size=image_size,n_loc = number_of_blocks)
    model = SeeAble(number_of_blocks,device)
    
    train_loader=torch.utils.data.DataLoader(train_dataset,
                        batch_size=batch_size,
                        shuffle=True,
                        collate_fn=train_dataset.collate_fn,
                        num_workers=2,
                        pin_memory=True,
                        drop_last=True,
                        worker_init_fn=train_dataset.worker_init_fn
                        )
    #print(train_loader)
    print("Training start")
    for i in range(epochs):
        for k,batch in tqdm(enumerate(train_loader),desc=f"Epoch {i}"):
            loss = model.training_step(batch,k)
            print(loss.item())
            for optim in model.optims:
                a=0
                #optim.step()
        for scheduler in model.schedulers:
            a=0
            #scheduler.step()
        
        
    """
    val_loader=torch.utils.data.DataLoader(val_dataset,
                        batch_size=batch_size,
                        shuffle=False,
                        collate_fn=val_dataset.collate_fn,
                        num_workers=4,
                        pin_memory=True,
                        worker_init_fn=val_dataset.worker_init_fn
                        )"""

if __name__=='__main__':

    torch.cuda.empty_cache()
    parser=argparse.ArgumentParser()
    parser.add_argument(dest='config')
    args=parser.parse_args()
    main(args)


