#https://discuss.pytorch.org/t/how-the-pytorch-freeze-network-in-some-layers-only-the-rest-of-the-training/7088/6
import torch
import torch.optim as optim
from torch.autograd import Variable
import torchvision.transforms as transforms
import time
from sklearn.metrics import accuracy_score, confusion_matrix
import gc
from pytorchtools import EarlyStopping
from datagen import DentalDataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold
import numpy as np
from torch.utils.data import Dataset, DataLoader
import sys
from sklearn.metrics import roc_auc_score, roc_curve, auc, f1_score, precision_recall_curve
import matplotlib.pyplot as plt

# delete here
from flashtorch.utils import apply_transforms, load_image
from flashtorch.saliency import Backprop
from flashtorch.utils import (denormalize, format_for_plotting, standardize_and_clip)
import cv2

from CV_utils import *
from utils import compute_pos_weights

from dice import *

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
Tensor = FloatTensor


def createLossAndOptimizer(net, learning_rate = .001, weights=None):
    loss = torch.nn.BCEWithLogitsLoss(pos_weight=weights)
    #loss = FocalLoss()
    optimizer = optim.AdamW(net.parameters(), lr = learning_rate, amsgrad=True)
    return (loss, optimizer)


#https://stackoverflow.com/questions/31324218/scikit-learn-how-to-obtain-true-positive-true-negative-false-positive-and-fal
def perf_measure(true, out):
    print('perf measure true shape, outshape: ', true.shape, out.shape)
    ntot = 0
    ncorr = 0
    if type(true) is np.ndarray:
        if true.shape[0] != 11:
            true = np.transpose(true)
    elif type(true) is torch.Tensor:
        if true.shape[0] != 11:
            true = true.t()

    if type(out) is np.ndarray:
        if out.shape[0] != 11:
            out = np.transpose(out)
    elif type(out) is torch.Tensor:
        if out.shape[0] != 11:
            out = out.t()
    print('shapes after transform: ', true.shape, out.shape)

    
    for k in range(len(true)):
        anomalies = ['Mammalons', 'Impacted', 'Hypoplasia', 'IncisalFissure', 'Fluorosis', 'Hypocalcification', 'Displaced', 'Microdontia', 'Supernumerary', 'Rotation', 'Agenesis']
        print("anomaly: ", anomalies[k])
        print('median prediction: ', np.median(out[k]))

        
        test = out[k].round()
        y_actual = true[k]
        y_hat = out[k].round()
        if use_cuda and False:
            ncorr += sum(a==b for a,b in zip(y_actual.cpu().numpy().tolist(), y_hat.cpu().numpy().tolist()))
        else:
            ncorr += sum(a==b for a,b in zip(y_actual, y_hat)).item()
        
        ntot += len(true[k])
        
        print("TN FP FN TP")
       
        if use_cuda and False:
            print(confusion_matrix(y_actual.cpu().numpy(), y_hat.cpu().numpy()).ravel())
            print('number true: ', len([n for n in true[k] if n == 1]))
            print('number false: ', len([n for n in true[k] if n == 0]))
            print("Anomaly accuracy: ", accuracy_score(y_actual.cpu().numpy(), y_hat.cpu().numpy()))
        else:
            print(confusion_matrix(y_actual, y_hat).ravel())
        
            print("Anomaly accuracy: ", accuracy_score(y_actual, y_hat))
            print('F1 based on g-mean: ', f1_score(y_actual, y_hat))


    return ncorr / ntot



def threshold_vec(vec, thresholds, rowshape):
    print('==========================', type(vec))
    print(vec)
    if vec.shape[0] != rowshape:
        vec = np.transpose(vec)
    for i, row in enumerate(vec):
        for j, col in enumerate(vec[i]):
            if vec[i][j] >= thresholds[i]:
                vec[i][j] = 1
            else:
                vec[i][j] = 0

    if vec.shape[0] == rowshape:
        vec = np.transpose(vec)

    return vec

def kfold_trainNet(net, 
                   batch_size, 
                   n_epochs, 
                   patience, 
                   learning_rate=.001, 
                   name='best_net.pt', 
                   folds=5,
                   folders=[1,2,3,4,5,6,7], 
                   fold_thread = None, 
                   predict_via_init = False):
    # save initial state dict
    best_loss = 99999999999999
    data_proportion = 1.
    torch.save(net, 'initial_model.pt')

    # Facilitate Early Stopping
    early_stopping = EarlyStopping(patience=patience, verbose=True)#,delta=.0001)
    
    # Track losses
    train_losses = []
    valid_losses = []
    avg_train_losses = []
    avg_valid_losses = []

    # standard imagenet transforms necessary for resnet
    t = transforms.Compose([ transforms.Resize((224,224)), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # Construct dataset
    dataset = DentalDataset('/Shared/howe_lab/OFC1/IOP/Photos',t, folders)

    # Set loss and optimizer
    # You can also try focal loss or weighted BCE
    # But in theory and from my tests, multiclass dice is best option
    loss_fn = MulticlassDiceLoss()
    optimizer = optim.AdamW(net.parameters(), lr = learning_rate, amsgrad=True)
    lr_sched = optim.lr_scheduler.StepLR(optimizer, step_size = 5, gamma=.1)

    #  Set up 70-15-15 split for early stopping
    dataset_size = len(dataset)
    idx = list(range(dataset_size))
    idx = idx[:int(len(idx) * data_proportion)]

    vtsize = .3
    tsize = .15
    trainsplit = int(np.floor(.7*dataset_size*data_proportion))
    validsplit = int(np.floor(.85*dataset_size*data_proportion))
    train_idx, valid_idx, test_idx = idx[:trainsplit], idx[trainsplit:validsplit], idx[validsplit:]
   
    y_labels = [str(dataset.piclist[i][3]) for i in range(len(dataset.piclist))]
    group_labels = [str(dataset.piclist[i][2]) for i in range(len(dataset.piclist))]
    
    le_a, le_g, le_y = LabelEncoder(), LabelEncoder(), LabelEncoder()
    le_y.fit(y_labels)
    le_g.fit(group_labels)

    print('Constructed Dataset...')
    dataset_size = len(dataset)
    print('sanity check labels size: ', len(y_labels))
    print('dset size: ', dataset_size)

    # I found group K fold grouped on patient ID to work best
    KF_splits = GroupKFold(n_splits = folds)

    idx = list(range(dataset_size))
    fold_counter = 1
    skip_counter = 1

    for fld, (train_idx, test_idx) in enumerate(KF_splits.split(idx, y_labels, group_labels)):#stratKF_splits.split(idx):
        if fold_thread is not None:
            if fld + 1 != fold_thread:
                print('continue at fold', fld+1)
                continue

        print('Beginning fold ', fld)
        # reset model for each run of CV
        valid_idx = train_idx[int(len(train_idx)*.8  + 1):]
        train_idx = train_idx[:int((len(train_idx)+1)*.8)]

        tr_labels = np.array([dataset.piclist[i][3] for i in train_idx])
        v_labels = np.array([dataset.piclist[i][3] for i in valid_idx])
        tst_labels = np.array([dataset.piclist[i][3] for i in test_idx])

        tr_a_cts = tr_labels.sum(axis=0)
        v_a_cts = v_labels.sum(axis=0)
        tst_a_cts = tst_labels.sum(axis=0)
        
        loss_fn = MulticlassDiceLoss()
        
        net = torch.load('initial_model.pt')
        
        # You could also use a weighted sampler to address class imbalance
        # but I found this to work better
        train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_idx)
        valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(valid_idx)
        test_sampler = torch.utils.data.sampler.SubsetRandomSampler(test_idx)

            
        print('train, valid, test sizes: ', len(train_sampler), len(valid_sampler), len(test_sampler))


        train_loader = DataLoader(dataset,batch_size=batch_size,sampler=train_sampler,num_workers=16) 
        valid_loader = DataLoader(dataset,batch_size=batch_size,sampler=valid_sampler,num_workers=16)
        test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler,num_workers=16)
        
        print('Beginning Training',flush=True)
        n_batches = len(train_loader)
        print("Batch size, num batches: ", batch_size, len(train_loader))
        
        for epoch in range(n_epochs):
            running_loss = 0.0
            total_loss = 0

            start_time = time.time()

            if epoch % 5 == 0: 
                print('stepping LR sched')
            lr_sched.step()

            for i, data in enumerate(train_loader):
                print("Epoch, batch: ", epoch, i)
                inputs, labels = data['image'], data['label']
                data['label'] = torch.stack(data['label'])

                inputs, labels = Variable(inputs, volatile=True), Variable(data['label'], volatile=True)
                inputs = inputs.type(torch.FloatTensor)
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                

                #Fwd pass
                outputs = net(inputs)
                                
                labels = torch.FloatTensor([[float(label) for label in v] for v in labels])
                labels = labels.t()
                if use_cuda:
                    outputs = outputs.to(device)
                    labels = labels.to(device)

                                
                loss = loss_fn((outputs.cpu()).float(), (labels.cpu()).float())
                loss.backward()
                optimizer.step()

                # Record a Training Loss
                train_losses.append(loss.item())

                labels.detach()
                outputs.detach()
                gc.collect()
                optimizer.zero_grad()
                

            # Go to eval mode for validation step
            net.eval()
            val_start = time.time()
            with torch.no_grad():
                for i, data in enumerate(valid_loader):
                    inputs, labels = data['image'], data['label']
                    data['label'] = torch.stack(data['label'])
                    inputs, labels = Variable(inputs, volatile=True), Variable(data['label'], volatile=True)
                    inputs = inputs.type(torch.FloatTensor)
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    output = net(inputs)
                    labels = torch.FloatTensor([[float(label) for label in v] for v in labels])
                    labels = labels.t()
                    

                    if use_cuda:
                        outputs = outputs.to(device)
                        labels = labels.to(device)
                        truth = data['label'].detach().numpy()
                    else:
                        truth = data['label'].detach()
                        
                    
                    if truth.shape[0] == len(dataset.anomalies):
                        truth = np.transpose(truth)
                                    
                    loss = loss_fn((output.cpu()).float(), (labels.cpu()).float())
                    if loss < best_loss:
                        best_loss = loss
                        torch.save(net, name)
                        torch.save(net, 'net_fold='+str(fld+1)+'+.pt')
                        torch.save(net, 'best_at_fold_'+str(fld+1)+'.pt')
                    valid_losses.append(loss.item())

            # Keep track of losses for early stopping
            train_loss = np.average(train_losses)
            valid_loss = np.average(valid_losses)
            avg_train_losses.append(train_loss)
            avg_valid_losses.append(valid_loss)

            plt.clf()
            x = list(range(len(avg_train_losses)))
            plt.plot(x, avg_train_losses)
            plt.savefig('orglc.png')
            plt.clf()

            # Clear for next epoch
            train_losses = []
            valid_losses = []
            
            # check if validation loss decreased
            early_stopping(valid_loss, net)
            if early_stopping.early_stop:
                print("EARLY STOPPING: ")
                torch.cuda.empty_cache()
                gc.collect()
                break
            print("epoch ", epoch, " done")
                                                


        
                                                        

        # LOAD BEST MODEL AND TEST
        if not predict_via_init:
            net.load_state_dict(torch.load('checkpoint.pt'))
        else:
            net = torch.load('initial_model.pt')
        with torch.no_grad():
            print('Beginning test routine for fold: ', fld+1)
            test_start = time.time()
            FPR = dict()
            TPR = dict()
            roc_auc = dict()
            thresholds = dict()
            precisions = dict()
            recalls = dict()
            pr_thresh = dict()
            y_test = np.array([])
            y_score = np.array([])
            for i, data in enumerate(test_loader):
                print('Batch ', i, ' results: ')

                imgs, labels = data['image'], data['label']
                data['label'] = torch.stack(data['label'])
                imgs, labels = Variable(imgs, volatile=True), Variable(data['label'], volatile=True)
                imgs = imgs.type(torch.FloatTensor)
                imgs = imgs.to(device)
                labels = labels.t()
                labels = labels.to(device)

                net.eval()
                optimizer.zero_grad()

                out = net(imgs)
                out = torch.nn.functional.softmax(out, dim=0)
                
                if not use_cuda:
                    truth = data['label'].detach().numpy()
                else:
                    truth = data['label'].detach()

                if truth.shape[0] == len(dataset.anomalies):
                    truth = np.transpose(truth)


                if type(truth) is torch.Tensor:
                    if truth.is_cuda:
                        truth = truth.cpu().numpy()
                    else:
                        truth = truth.numpy()
                if type(out) is torch.Tensor:
                    if out.is_cuda:
                        out = out.detach().cpu().numpy()
                    else:
                        out = out.numpy()

                # construct output and truth vectors from all batches
                if i == 0 or len(y_test) == 0:
                    y_test = np.transpose(truth)
                    y_score = np.transpose(out)
                else:
                    y_test = np.concatenate((y_test, np.transpose(truth)), axis = 1)
                    y_score = np.concatenate((y_score, np.transpose(out)), axis = 1)

                gc.collect()
            

        # Increment fold and begin constructing ROC/AUC

        print('detatch image')
        imgs = imgs.detach().cpu()
       
        for j, row in enumerate(y_test):
            FPR[j], TPR[j], thresholds[j] = roc_curve(y_test[j], y_score[j])
            precisions[j], recalls[j], pr_thresh[j] = precision_recall_curve(y_test[j],y_score[j])
            roc_auc[j] = auc(FPR[j], TPR[j])
            print('ROC_AUC for class ', j, ': ', roc_auc[j])
            
        FPR['micro'], TPR['micro'], thresholds['micro'] = roc_curve(y_test.ravel(), y_score.ravel())
        roc_auc['micro'] = auc(FPR['micro'], TPR['micro'])
        print('micro average auc: ', roc_auc['micro'])

        thresh_opt = []
        for j in range(len(FPR)-1):
            plt.figure()
            lw = 2
            gmeans = np.sqrt(TPR[j] * (1 - FPR[j]))
            ix = np.argmax(gmeans)
            #print('Best threshold, gmeans for anomaly: ', j, thresholds[j][ix], gmeans[ix])
            thresh_opt.append(thresholds[j][ix])

            plt.plot(FPR[j], TPR[j], color='darkorange',
                     lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[j])
            plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
            plt.scatter(FPR[j][ix], TPR[j][ix], marker = 'o', color = 'black', label = 'Best')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver operating characteristic for ' + dataset.anomalies[j] + ' fold ' +  str(fld+1))
            plt.legend(loc="lower right")
            plt.savefig('ROC_fold_'+str(fld+1)+'_'+str(j)+'.png')
            plt.clf()
        
        for j in range(len(precisions)):
            plt.figure()
            lw = 2
            area = auc(recalls[j], precisions[j])
            fscores = (2 * precisions[j] * recalls[j]) / (precisions[j] + recalls[j])
            try:            
                ix = np.nanargmax(fscores)
                print(dataset.anomalies[j], ' F1 at best point: ', fscores[ix], ' via P-R: ', precisions[j][ix], recalls[j][ix])
            except Exception:
                ix = np.argmax(fscores)
                print('F1 at best point: ', fscores[ix], ' via P-R: ', precisions[j][ix], recalls[j][ix])
            plt.plot(recalls[j], precisions[j], color='darkorange',
                     lw=lw, label='PR curve (area = %0.2f)' % area)
            #plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.ylabel('Precision')
            plt.xlabel('Recall')
            plt.title('Precision-Recall curve for ' + dataset.anomalies[j] + ' fold ' +  str(fld+1))
            plt.legend(loc="lower left")
            plt.savefig('PR_fold_'+str(fld+1)+'_'+str(j)+'.png')
            plt.clf()
        print('took ', time.time() - test_start, ' to do all testing')

        '''
        print("FINAL PARRT: ")
        y_score_thresh = threshold(y_score, thresh_opt)
        print('micro F1: ', f1_score(y_test, y_score_thresh, average='micro'))
        print('macro F1: ', f1_score(y_test, y_score_thresh, average='macro'))
        print('weighted F1: ', f1_score(y_test, y_score_thresh, average='weighted'))
        #perf_measure(truth.cpu().numpy(), out3.cpu().numpy())
        perf_measure(np.transpose(y_test), np.transpose(y_score_thresh))
        '''
    
def threshold(vec, thresh):
    for i, row in enumerate(vec):
        for j, col in enumerate(vec[i]):
            if vec[i][j] >= thresh[i]:
                vec[i][j] = 1
            else:
                vec[i][j] = 0

    return vec

            
