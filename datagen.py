#https://discuss.pytorch.org/t/how-the-pytorch-freeze-network-in-some-layers-only-the-rest-of-the-training/7088/6
import torch
import torchvision
import pandas as pd
import numpy as np
import os
from PIL import ImageFile, Image
import logging
from torchvision.datasets import ImageFolder, DatasetFolder
from torch.utils.data import Dataset, DataLoader

ImageFile.LOAD_TRUNCATED_IMAGES = True

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
Tensor = FloatTensor

#https://discuss.pytorch.org/t/how-to-create-a-tensor-on-gpu-as-default/2128/3
handler = logging.basicConfig(level=logging.INFO)
lgr = logging.getLogger(__name__)
noncleft = 0
cleft = 0 
checked = []

def my_numpy_loader(filename):
    return np.load(filename)

#/Shared/howe_lab/OFC1/IOP/DatawithID/IOP_DemoComposite_03282017 with ID.csv
#/Shared/howe_lab/OFC1/IOP/Photos 
class DentalDataset(Dataset):
    def __init__(self, root, transform = None,folders = [1,2,3,4,5,6,7]):
        self.root = root
        valid = folders
        
        self.foldersizes = list(range(len(valid)))
        self.valid = valid
        self.folders = [os.listdir(os.path.join(self.root, 'folder_{}'.format(i), 'OFC1')) for i in valid]
        self.table = pd.read_csv('/Shared/howe_lab/OFC1/IOP/DatawithID/IOP_DemoComposite_03282017 with ID.csv')
        self.subset = []
        self.anomct = {}

        self.anomalies = ['Mammalons', 'Impacted', 'Hypoplasia', 'IncisalFissure', 'Fluorosis', 'Hypocalcification', 'Displaced', 'Microdontia', 'Supernumerary', 'Rotation','Agenesis']
        self.acols = ['Tooth18Fluorosis', 'Tooth18Hypoplasia', 'Tooth18Hypocalcification', 'Tooth18Microdontia', 'Tooth18Impacted', 'Tooth18Rotation', 'Tooth18Displaced', 'Tooth18Mammalons', 'Tooth18IncisalFissure', 'Tooth17Fluorosis', 'Tooth17Hypoplasia', 'Tooth17Hypocalcification', 'Tooth17Microdontia', 'Tooth17Impacted', 'Tooth17Rotation', 'Tooth17Displaced', 'Tooth17Mammalons', 'Tooth17IncisalFissure', 'Tooth16Fluorosis', 'Tooth16Hypoplasia', 'Tooth16Hypocalcification', 'Tooth16Microdontia', 'Tooth16Impacted', 'Tooth16Rotation', 'Tooth16Displaced', 'Tooth16Mammalons', 'Tooth16IncisalFissure', 'Tooth15Fluorosis', 'Tooth15Hypoplasia', 'Tooth15Hypocalcification', 'Tooth15Microdontia', 'Tooth15Impacted', 'Tooth15Rotation', 'Tooth15Displaced', 'Tooth15Mammalons', 'Tooth15IncisalFissure', 'Tooth14Fluorosis', 'Tooth14Hypoplasia', 'Tooth14Hypocalcification', 'Tooth14Microdontia', 'Tooth14Impacted', 'Tooth14Rotation', 'Tooth14Displaced', 'Tooth14Mammalons', 'Tooth14IncisalFissure', 'Tooth13Fluorosis', 'Tooth13Hypoplasia', 'Tooth13Hypocalcification', 'Tooth13Microdontia', 'Tooth13Impacted', 'Tooth13Rotation', 'Tooth13Displaced', 'Tooth13Mammalons', 'Tooth13IncisalFissure', 'Tooth12Fluorosis', 'Tooth12Hypoplasia', 'Tooth12Hypocalcification', 'Tooth12Microdontia', 'Tooth12Impacted', 'Tooth12Rotation', 'Tooth12Displaced', 'Tooth12Mammalons', 'Tooth12IncisalFissure', 'Tooth11Fluorosis', 'Tooth11Hypoplasia', 'Tooth11Hypocalcification', 'Tooth11Microdontia', 'Tooth11Impacted', 'Tooth11Rotation', 'Tooth11Displaced', 'Tooth11Mammalons', 'Tooth11IncisalFissure', 'Tooth21Fluorosis', 'Tooth21Hypoplasia', 'Tooth21Hypocalcification', 'Tooth21Microdontia', 'Tooth21Impacted', 'Tooth21Rotation', 'Tooth21Displaced', 'Tooth21Mammalons', 'Tooth21IncisalFissure', 'Tooth22Fluorosis', 'Tooth22Hypoplasia', 'Tooth22Hypocalcification', 'Tooth22Microdontia', 'Tooth22Impacted', 'Tooth22Rotation', 'Tooth22Displaced', 'Tooth22Mammalons', 'Tooth22IncisalFissure', 'Tooth23Fluorosis', 'Tooth23Hypoplasia', 'Tooth23Hypocalcification', 'Tooth23Microdontia', 'Tooth23Impacted', 'Tooth23Rotation', 'Tooth23Displaced', 'Tooth23Mammalons', 'Tooth23IncisalFissure', 'Tooth24Fluorosis', 'Tooth24Hypoplasia', 'Tooth24Hypocalcification', 'Tooth24Microdontia', 'Tooth24Impacted', 'Tooth24Rotation', 'Tooth24Displaced', 'Tooth24Mammalons', 'Tooth24IncisalFissure', 'Tooth25Fluorosis', 'Tooth25Hypoplasia', 'Tooth25Hypocalcification', 'Tooth25Microdontia', 'Tooth25Impacted', 'Tooth25Rotation', 'Tooth25Displaced', 'Tooth25Mammalons', 'Tooth25IncisalFissure', 'Tooth26Fluorosis', 'Tooth26Hypoplasia', 'Tooth26Hypocalcification', 'Tooth26Microdontia', 'Tooth26Impacted', 'Tooth26Rotation', 'Tooth26Displaced', 'Tooth26Mammalons', 'Tooth26IncisalFissure', 'Tooth27Fluorosis', 'Tooth27Hypoplasia', 'Tooth27Hypocalcification', 'Tooth27Microdontia', 'Tooth27Impacted', 'Tooth27Rotation', 'Tooth27Displaced', 'Tooth27Mammalons', 'Tooth27IncisalFissure', 'Tooth28Fluorosis', 'Tooth28Hypoplasia', 'Tooth28Hypocalcification', 'Tooth28Microdontia', 'Tooth28Impacted', 'Tooth28Rotation', 'Tooth28Displaced', 'Tooth28Mammalons', 'Tooth28IncisalFissure', 'Tooth31Fluorosis', 'Tooth31Hypoplasia', 'Tooth31Hypocalcification', 'Tooth31Microdontia', 'Tooth31Impacted', 'Tooth31Rotation', 'Tooth31Displaced', 'Tooth31Mammalons', 'Tooth31IncisalFissure', 'Tooth32Fluorosis', 'Tooth32Hypoplasia', 'Tooth32Hypocalcification', 'Tooth32Microdontia', 'Tooth32Impacted', 'Tooth32Rotation', 'Tooth32Displaced', 'Tooth32Mammalons', 'Tooth32IncisalFissure', 'Tooth33Fluorosis', 'Tooth33Hypoplasia', 'Tooth33Hypocalcification', 'Tooth33Microdontia', 'Tooth33Impacted', 'Tooth33Rotation', 'Tooth33Displaced', 'Tooth33Mammalons', 'Tooth33IncisalFissure', 'Tooth34Fluorosis', 'Tooth34Hypoplasia', 'Tooth34Hypocalcification', 'Tooth34Microdontia', 'Tooth34Impacted', 'Tooth34Rotation', 'Tooth34Displaced', 'Tooth34Mammalons', 'Tooth34IncisalFissure', 'Tooth35Fluorosis', 'Tooth35Hypoplasia', 'Tooth35Hypocalcification', 'Tooth35Microdontia', 'Tooth35Impacted', 'Tooth35Rotation', 'Tooth35Displaced', 'Tooth35Mammalons', 'Tooth35IncisalFissure', 'Tooth36Fluorosis', 'Tooth36Hypoplasia', 'Tooth36Hypocalcification', 'Tooth36Microdontia', 'Tooth36Impacted', 'Tooth36Rotation', 'Tooth36Displaced', 'Tooth36Mammalons', 'Tooth36IncisalFissure', 'Tooth37Fluorosis', 'Tooth37Hypoplasia', 'Tooth37Hypocalcification', 'Tooth37Microdontia', 'Tooth37Impacted', 'Tooth37Rotation', 'Tooth37Displaced', 'Tooth37Mammalons', 'Tooth37IncisalFissure', 'Tooth38Fluorosis', 'Tooth38Hypoplasia', 'Tooth38Hypocalcification', 'Tooth38Microdontia', 'Tooth38Impacted', 'Tooth38Rotation', 'Tooth38Displaced', 'Tooth38Mammalons', 'Tooth38IncisalFissure', 'Tooth48Fluorosis', 'Tooth48Hypoplasia', 'Tooth48Hypocalcification', 'Tooth48Microdontia', 'Tooth48Impacted', 'Tooth48Rotation', 'Tooth48Displaced', 'Tooth48Mammalons', 'Tooth48IncisalFissure', 'Tooth47Fluorosis', 'Tooth47Hypoplasia', 'Tooth47Hypocalcification', 'Tooth47Microdontia', 'Tooth47Impacted', 'Tooth47Rotation', 'Tooth47Displaced', 'Tooth47Mammalons', 'Tooth47IncisalFissure', 'Tooth46Fluorosis', 'Tooth46Hypoplasia', 'Tooth46Hypocalcification', 'Tooth46Microdontia', 'Tooth46Impacted', 'Tooth46Rotation', 'Tooth46Displaced', 'Tooth46Mammalons', 'Tooth46IncisalFissure', 'Tooth45Fluorosis', 'Tooth45Hypoplasia', 'Tooth45Hypocalcification', 'Tooth45Microdontia', 'Tooth45Impacted', 'Tooth45Rotation', 'Tooth45Displaced', 'Tooth45Mammalons', 'Tooth45IncisalFissure', 'Tooth44Fluorosis', 'Tooth44Hypoplasia', 'Tooth44Hypocalcification', 'Tooth44Microdontia', 'Tooth44Impacted', 'Tooth44Rotation', 'Tooth44Displaced', 'Tooth44Mammalons', 'Tooth44IncisalFissure', 'Tooth43Fluorosis', 'Tooth43Hypoplasia', 'Tooth43Hypocalcification', 'Tooth43Microdontia', 'Tooth43Impacted', 'Tooth43Rotation', 'Tooth43Displaced', 'Tooth43Mammalons', 'Tooth43IncisalFissure', 'Tooth42Fluorosis', 'Tooth42Hypoplasia', 'Tooth42Hypocalcification', 'Tooth42Microdontia', 'Tooth42Impacted', 'Tooth42Rotation', 'Tooth42Displaced', 'Tooth42Mammalons', 'Tooth42IncisalFissure', 'Tooth41Fluorosis', 'Tooth41Hypoplasia', 'Tooth41Hypocalcification', 'Tooth41Microdontia', 'Tooth41Impacted', 'Tooth41Rotation', 'Tooth41Displaced', 'Tooth41Mammalons', 'Tooth41IncisalFissure']
        self.extra = ['Extra1817', 'Extra1716', 'Extra1615', 'Extra1514', 'Extra1413', 'Extra1312', 'Extra1211', 'Extra1121', 'Extra2122', 'Extra2223', 'Extra2324', 'Extra2425', 'Extra2526', 'Extra2627', 'Extra2728', 'Extra3132', 'Extra3233', 'Extra3334', 'Extra3435', 'Extra3536', 'Extra3637', 'Extra3738', 'Extra4847', 'Extra4746', 'Extra4645', 'Extra4544', 'Extra4443', 'Extra4342', 'Extra4241', 'Extra4131']
        self.tooth = ['Tooth17','Tooth27','Tooth14','Tooth15','Tooth24','Tooth25','Tooth34','Tooth35','Tooth44','Tooth45']

        self.acols = self.acols + self.extra
        self.label_proportions = np.zeros(len(self.anomalies))
        self.weights = []
        self.nsamples = 0
        self.ncontrols = 0
        global noncleft 

        global cleft
        ids = self.table['StudyID'].tolist()
        i = 1
        self.piclist = []
        invalid2 = []
        for k, folder in enumerate(self.folders):
            print('FOLDER: ', k)
            if '.DS_Store' in folder:
                folder.remove('.DS_Store')
            if 'Archive' in folder:
                folder.remove('Archive')
            names = [f for f in folder if '.' in f] 
            for name in names:
                folder.remove(name)

            invalid = [f for f in folder if f not in ids]

            for invf in invalid:
                folder.remove(invf)

            if valid[k] == 6:
                folder.remove('PR0027')
            for f in folder:
                files = os.listdir( os.path.join(self.root, 'folder_{}'.format(self.valid[k]), 'OFC1', f))
                
                files = [f for f in files if (f[-4:] == '.JPG' or f[-4:] == '.jpg') and ('t' in f or 'T' in f)]
                

                for ff in files:
                    if ff[:2] == 'FC':
                        if '-1' in ff:
                            invalid2.append(ff)
                    if ff[:2] in ['PH', 'TX', 'PR', 'CO']:
                        if 't1' in ff:
                            invalid2.append(ff)


                for ff in files:
                    name = ff.split('.')[0]
                    if 't' in name:
                        name = name.split('t')[0]
                    elif 'T' in name:
                        name = name.split('T')[0]
                    name = name.upper()
                    name =name.strip(' \t\n\r')

                    picrow = self.table[self.table['StudyID'] == name]
                    #print('NAME: ', name)
                    if 'gw' in name or '0885' in name:
                        #print('CHECKING gw: ', name)
                        pass
                    for col in self.acols:
                        if len(picrow[col].values) == 0:
                            invalid2.append(ff)
                            #print('ff: ', ff, ' name: ', name, ' FAILED!')
                            break
                        if len(picrow[col].values) > 0:
                            #print('v2 vals: ', col, int(picrow[col].values[0])) 
                            if int(picrow[col].values[0]) == -6666:
                                picrow[col].values[0] = 0
                            if int(picrow[col].values[0]) not in [0,1]:
                                 invalid2.append(ff)
                                 #print('ff: ', ff, ' name: ', name, ' FAILED ver 2!')
                                 break

                #print("INVALID: ", invalid2)
                files = [f for f in files if f not in invalid2]
                if 'PR0027t2.JPG' in files:
                    folder.remove(f)
                if len(files) < 2:
                    folder.remove(f)
                else:
                    self.foldersizes[k] += len(files)
           
                for j, ff in enumerate(files):
                    name = ff.split('.')[0]
                    if 't' in name:
                        name = name.split('t')[0]
                    elif 'T' in name:
                        name = name.split('T')[0]
                    name = name.upper()
                    name =name.strip(' \t\n\r')

                    fullpath = os.path.join(self.root, 'folder_{}'.format(self.valid[k]), 'OFC1', f, ff)
                    
                    label = self.get_label(name)

                    self.piclist = self.piclist + [(ff, fullpath, name, label)]
                         
            i = i + 1
        
        # Make weights for loss function to address class imbalance
        print('adding epsilon to weight vector...')
        self.label_proportions = self.label_proportions + .000001*np.ones(len(self.anomalies))
        print('self.label_proportions: ', self.label_proportions)
        self.loader = torchvision.datasets.ImageFolder
        self.transform = transform
        print("SELF.FOlDERSIZES: ", self.foldersizes)

    def __len__(self):
        return len(self.piclist)

        
    def __getitem__(self, idx):
        global cleft
        global noncleft
        
        fullpath = self.piclist[idx][1]
        caseid = self.piclist[idx][2]
        label = self.piclist[idx][3]
        
        img = Image.open(fullpath)
        sample = {'image': img, 'label': label, 'cids' : caseid}

        if self.transform:
           sample['image'] = self.transform(img)
        
        sample = {'image': sample['image'], 'label': label, 'cids' : caseid}#y[idx]}
        
        #print('sample cid:' , caseid)
        return sample

    def get_label(self, caseid):
        global cleft
        global noncleft
        global checked


        picrow = self.table[self.table['StudyID'] == caseid]
        cols = list(picrow)

        if self.subset == []:
            self.subset = [col for col in cols if 
                       ('Mammalons' in col or
                        'Impacted' in col or 
                        'Hypoplasia' in col or 
                        'Rotation' in col or 
                        'IncisalFissure' in col or
                        'Fluorosis' in col or
                        'Hypocalcification' in col or
                        'Displaced' in col) and
                       (np.issubdtype(self.table[col].dtype,np.number) 
                        and not self.table[col].empty
                        and not self.table[col].isnull().values.any())]

        val = [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]

        
        if caseid not in checked:
            try:
                if picrow['CleftInitials'].values[0] == '-':
                    noncleft += 1
                else:
                    cleft += 1
                checked.append(caseid)
            except Exception:
                print(
picrow['CleftInitials']     , picrow['CleftInitials'].values)
                cleft += 1 

        for i, anomaly in enumerate(self.anomalies):
            val[i] = 0
            for col in self.acols:
                if self.anomalies[i] in col and i != 8:
                    try:
                        temp = picrow[col].values[0]
                        if int(temp) == 1:
                            val[i] = 1
                            break
            
                    except Exception:
                        print("COULD NOT FIND LABEL", col, " for caseid: ", caseid, ' with anomaly ', self.anomalies[i])
            if i == 8:
                try:
                    for col in self.extra:
                        temp = picrow[col].values[0]
                        if int(temp) == 1:
                            val[i] = 1
                            break
                except Exception:
                    print("COULD NOT FIND LABEL", col, " for caseid: ", caseid, ' with anomaly ', self.anomalies[i])

            if i == len(val)-1:
                try:
                    for col in self.tooth:
                        temp = picrow[col].values[0]
                        tnum  = int(col[-2:])
                        
                        if int(temp) == 2:
                            val[i] = 1
                            break
                        
                        if picrow['GivenAge'].values[0] >= 14 or (tnum in [17,27] and temp == 7):
                          
                            if tnum in [14,15,24,25,34,35,44,45] and temp==7:
                          
                                if picrow['Type'+str(tnum)].values[0] == 1:
                                    val[i] = 1
                                    break
                        

                except Exception as e:
                    print("COULD NOT FIND LABEL", col, " for caseid: ", caseid, ' with anomaly ', self.anomalies[i])
                    print(e)
        label = val
        #print('val: ', val)
        self.label_proportions = self.label_proportions + np.array(val)
        #print('incremented: ', self.label_proportions)
        self.nsamples += 1
        

        overlap = np.multiply(np.array(val), np.array([ 7752,1177,4376,1406,50,21774,22640,2665,1235,37320,
  1486]))
        #print('overlap: ', overlap)

        if 1 in val:
            #print(np.min(overlap[np.nonzero(overlap)]))
            self.weights.append(1. / np.min(overlap[np.nonzero(overlap)]))

        # FIX THIS; ITS BAD
        # ncontrols actually ~10 but want to encourage positive class
        else:
            self.weights.append(1. / 37320)
            self.ncontrols += 1

        act = np.sum(val)
        if act not in list(self.anomct):
            self.anomct[act] = 1
        else:
            self.anomct[act] = self.anomct[act] + 1
        
        #print('anomct: ', self.anomct)
        return val
