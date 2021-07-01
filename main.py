#https://discuss.pytorch.org/t/how-the-pytorch-freeze-network-in-some-layers-only-the-rest-of-the-training/7088/6
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import logging
from torch.optim import lr_scheduler
from model import *

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
Tensor = FloatTensor

#https://discuss.pytorch.org/t/how-to-create-a-tensor-on-gpu-as-default/2128/3
handler = logging.basicConfig(level=logging.INFO)
lgr = logging.getLogger(__name__)
if use_cuda:
    lgr.info ("Using the GPU")    

else:
    lgr.info ("Using the CPU")


# get pretrained resnet; freeze 1st lt layers
model_ft = models.resnet18(pretrained=True)
lt = 7
cntr = 0

for child in model_ft.children():
    #print('cntr: ', cntr)
    cntr += 1
    if cntr < lt:
        for param in child.parameters():
            param.requires_grad = False
    if cntr >= lt:
        #print('do grad for: ')
        #print(child)
        pass

# change output to 11 classes
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 11)

# load model onto parallel GPUs
model_ft = nn.DataParallel(model_ft)
model_ft = model_ft.to(device)

predict_via_init = False

ft = 1


warm_start = False
if warm_start:
    print('uscda: ', use_cuda)
    if not use_cuda:
        #model_ft = torch.load('ptsaved/fold'+str(ft)+'.pt',map_location=torch.device('cpu'))
        model_ft = torch.load('fold'+str(ft)+'.pt',map_location=torch.device('cpu'))
        #model_ft.load_state_dict(torch.load('checkpoint.pt',map_location=torch.device('cpu')))
    else:
        #model_ft = torch.load('best_at_fold_'+str(ft)+'.pt')
        model_ft = torch.load('f1_save.pt')
        #model_ft = torch.load('ptsaved/fold'+str(ft)+'.pt')
        #model_ft =  torch.load('/Users/rragodos/git_anomaly/parallel/f1/best_at_fold_1.pt')#torch.load('fold'+str(ft)+'.pt')
        #model_ft.load_state_dict(torch.load('checkpoint.pt'))
        model_ft = model_ft.to(device)

#SELF.FOlDERSIZES:  [2478, 9292, 1975, 12819, 256, 1062, 2371]
kfold_trainNet(model_ft,  # model you wanna train
               batch_size=512, 
               n_epochs=2, 
               patience=60, 
               learning_rate=1.34E-6, # I found this to be optimal per an lr finding library
               name='fold'+str(ft)+'tes5t.pt', # where to save best model
               folds=5, # number of CV folds
               folders=[5], # which folders to include; subset of [1,2,3,4,5,6,7] corresponding to regions of data
               fold_thread = ft, # which of the folds you want to do (fold data is generated deterministically)
               predict_via_init = predict_via_init) # if you wanna skip training and just eval the warm start model on test set
