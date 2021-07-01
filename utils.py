#https://discuss.pytorch.org/t/how-the-pytorch-freeze-network-in-some-layers-only-the-rest-of-the-training/7088/6
import torch
import torch.optim as optim
from torch.autograd import Variable
import torchvision.transforms as transforms
import time
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import roc_auc_score, roc_curve, auc, f1_score, precision_recall_curve
import gc
from pytorchtools import EarlyStopping
from datagen import DentalDataset
from sklearn.model_selection import KFold, StratifiedKFold, GroupShuffleSplit, StratifiedShuffleSplit
import numpy as np
from torch.utils.data import Dataset, DataLoader
import sys

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D 

# delete here
from flashtorch.utils import apply_transforms, load_image
from flashtorch.saliency import Backprop
from flashtorch.utils import (denormalize, format_for_plotting, standardize_and_clip)
import cv2

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
Tensor = FloatTensor

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imsave('samples.png', inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated



def createLossAndOptimizer(net, learning_rate = .001, weights=None):
    loss = torch.nn.BCEWithLogitsLoss(pos_weight=weights)
    #loss = torch.nn.BCELoss(weight=weights)
    optimizer = optim.AdamW(net.parameters(), lr = learning_rate, amsgrad=True)
    return (loss, optimizer)


#https://stackoverflow.com/questions/31324218/scikit-learn-how-to-obtain-true-positive-true-negative-false-positive-and-fal
def perf_measure(true, out, anomalies,verbose=True):
    #print('perf measure true shape, outshape: ', true.shape, out.shape)
    ntot = 0
    ncorr = 0
    if type(true) is np.ndarray:
        if true.shape[0] != len(anomalies):
            true = np.transpose(true)
    elif type(true) is torch.Tensor:
        if true.shape[0] != len(anomalies):
            true = true.t()

    if type(out) is np.ndarray:
        if out.shape[0] != len(anomalies):
            out = np.transpose(out)
    elif type(out) is torch.Tensor:
        if out.shape[0] != len(anomalies):
            out = out.t()
    #print('shapes after transform: ', true.shape, out.shape)

    
    for k in range(len(true)):
        
        if verbose:
            print('\n')
        print("anomaly: ", anomalies[k])
               
        test = out[k].round()
        y_actual = true[k]
        y_hat = out[k].round()
        if use_cuda and False:
            ncorr += sum(a==b for a,b in zip(y_actual.cpu().numpy().tolist(), y_hat.cpu().numpy().tolist()))
        else:
            ncorr += sum(a==b for a,b in zip(y_actual, y_hat)).item()
        
        ntot += len(true[k])
        
        if verbose:
            print("TN FP FN TP")
       
        if use_cuda and False:
            print(confusion_matrix(y_actual.cpu().numpy(), y_hat.cpu().numpy()).ravel())
            print('number true: ', len([n for n in true[k] if n == 1]))
            print('number false: ', len([n for n in true[k] if n == 0]))
            print("Anomaly accuracy: ", accuracy_score(y_actual.cpu().numpy(), y_hat.cpu().numpy()))
        else:
            print("accuracy: ", accuracy_score(y_actual, y_hat),'\n')
            if verbose:
                print(confusion_matrix(y_actual, y_hat).ravel())
                try:
                    print(anomalies[k], ' clf report')
                    print(classification_report(y_actual, y_hat))
                except Exception as e:
                    print('couldnt make anomaly clf report')
                    print(e)

    
    try:
        if verbose:
            print('Full CLF report: ')
            print(classification_report(true.astype(np.int), out.astype(np.int), labels=anomalies))
    except Exception as e:
        if verbose:
            print('Could not make clf report!')
            print(e)
            print('True shape: ', true.shape)
            print('out.shape: ', out.shape)
            print('True: ', true)
            print('out: ', out)
            print('anom: ', anomalies)
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

def threshold(vec, thresh):
    for i, row in enumerate(vec):
        for j, col in enumerate(vec[i]):
            if vec[i][j] >= thresh[i]:
                vec[i][j] = 1
            else:
                vec[i][j] = 0

    return vec

            

def predict(net, batch_size, 
            name='balance2.pt', 
             folders=[1,2,3,4,5,6,7],
             data_proportion = 1,
             imbalance_method='weighted',
             thresh=[],
             photopath='/Shared/howe_lab/OFC2/OFC2 Validation set/IOPs for Validation Set JDR paper/merged/',
             tablepath='/Shared/howe_lab/OFC2/OFC2 Validation set/IOPs.csv'):

    
    t = transforms.Compose([ transforms.Resize((224,224)), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # Construct dataset
    dataset = ValidationDataset(photopath,t, folders,tablepath,['Fluorosis'])
    dataset_size = len(dataset)
    idx = list(range(dataset_size))
    
    test_sampler = torch.utils.data.sampler.SubsetRandomSampler(idx)
    test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler, num_workers=16)

    net.load_state_dict(torch.load('checkpoint.pt'))
    net.eval()
    with torch.no_grad():
        FPR = dict()
        TPR = dict()
        thresholds = dict()
        roc_auc = dict()
        y_test = np.array([])
        y_score = np.array([])
        print("BEGINNING TEST ROUTINE")
        for i, data in enumerate(test_loader):
            print('on batch ', i)
            test_start = time.time()
            #dataiter = iter(test_loader)
            #data = dataiter.next()
            imgs, labels  = data['image'], data['label']
            print('label shape: ', data['label'].shape)
            #data['label'] = torch.stack(data['label'])
            imgs, labels = Variable(imgs,volatile=True), Variable(data['label'],volatile=True)
            imgs = imgs.type(torch.FloatTensor)

            imgs = imgs.to(device)
            labels = labels.t()

            labels = labels.to(device)
            net.eval()
            
            out = net(imgs)#.t()
            out = torch.nn.functional.softmax(out, dim=0)

            print("Truth: ", data['label'])
            print("Prediction: ", out)
            if not use_cuda:
                truth = data['label'].detach().numpy()
            else:
                truth = data['label'].detach()


            if truth.shape[0] == len(dataset.anomalies):
                truth = np.transpose(truth)

            outputs = out

            if not use_cuda:
                out = out.detach().numpy()
            else:
                out = out.detach()

            out3 = out

            # make numpy 
            print('truth, out3 in test shape: ', truth.shape, out3.shape)
            if type(truth) is torch.Tensor:
                if truth.is_cuda:
                    truth = truth.cpu().numpy()
                else:
                    truth = truth.numpy()
            if type(out3) is torch.Tensor:
                if out3.is_cuda:
                    out3 = out3.cpu().numpy()
                else:
                    out3 = out3.numpy()


            
            # construct output and truth vectors from all batches
            print('truth, outshape: ', truth.shape, out3.shape)
            if i == 0 or len(y_test) == 0:
                y_test = np.transpose(truth)
                y_score = np.transpose(out3)
            else:
                print('else part truth, outsshape: ', truth.shape, out3.shape)
                print('test, scoreshape: ', y_test.shape, y_score.shape)
                y_test = np.concatenate((y_test, np.transpose(truth)), axis=1)
                y_score = np.concatenate((y_score, np.transpose(out3)), axis=1)

            '''
            for j, row in enumerate(out3):
                for k, x in enumerate(row):
                    if x > .50:
                        out3[j][k] = 1
                    else:
                        out3[j][k] = 0
            '''
            out3 = threshold_vec(out3,thresh,len(dataset.anomalies))

            print('batch results: ')
            #perf_measure(truth, out3)

            gc.collect()

        # need to threshold before perf measure
        #print('results on whole test set: ')
        #perf_measure(np.transpose(y_test), np.transpose(y_score))

        print('beginning roc auc computation...')
        print('vector shapes: ', y_test.shape, y_score.shape)
        for j, row in enumerate(y_test):
            print('shape of test, score input for roc_curve: ', y_test[j].shape, y_score[j].shape)
            print('teset vec: ', y_test[j])
            print('score vec: ', y_score[j])
            FPR[j], TPR[j], thresholds[j] = roc_curve(y_test[j], y_score[j])
            roc_auc[j] = auc(FPR[j], TPR[j])
            print('types: ', type(FPR[j]))
            print('shapes: ', FPR[j].shape, TPR[j].shape, thresholds[j].shape)
            print('ROC_AUC for class ', j, ': ', roc_auc[j])

        FPR['micro'], TPR['micro'], thresholds['micro'] = roc_curve(y_test.ravel(), y_score.ravel())
        roc_auc['micro'] = auc(FPR['micro'], TPR['micro'])

        print('micro average auc: ', roc_auc['micro'])

        '''
        Best threshold, gmeans for anomaly:  0 4.144769e-06 0.5621021313588725
Best threshold, gmeans for anomaly:  1 2.712897e-06 0.501925926420419
Best threshold, gmeans for anomaly:  2 9.672138e-05 0.5129155380532567
Best threshold, gmeans for anomaly:  3 3.797817e-05 0.5230316993726799
Best threshold, gmeans for anomaly:  4 9.542505e-09 0.7640558123061597
Best threshold, gmeans for anomaly:  5 8.689116e-05 0.5343100539002834
Best threshold, gmeans for anomaly:  6 9.7504126e-05 0.5319450904826384
Best threshold, gmeans for anomaly:  7 7.3409424e-06 0.575578380227562
Best threshold, gmeans for anomaly:  8 3.034343e-06 0.5824579156213765
Best threshold, gmeans for anomaly:  9 0.00011589018 0.5855186103329046
Best threshold, gmeans for anomaly:  10 2.0305235e-05 0.5218899407401184
        '''



        y_score_thresh = threshold(y_score, thresh)
        print('micro F1: ', f1_score(y_test, y_score_thresh, average='micro'))
        print('macro F1: ', f1_score(y_test, y_score_thresh, average='macro'))
        print('weighted F1: ', f1_score(y_test, y_score_thresh, average='weighted'))
        #perf_measure(truth.cpu().numpy(), out3.cpu().numpy())
        #perf_measure(np.transpose(y_test), np.transpose(y_score_thresh))

        for j in range(len(FPR)-1):
            plt.figure()
            lw = 2

            gmeans = np.sqrt(TPR[j] * (1 - FPR[j]))
            ix = np.argmax(gmeans)
            print('Best threshold, gmeans for anomaly: ', j, thresholds[j][ix], gmeans[ix])
            
            plt.plot(FPR[j], TPR[j], color='darkorange',
                     lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[j])
            #plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
            plt.scatter(FPR[j][ix], TPR[j][ix], marker = 'o', color = 'black', label = 'Best')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver operating characteristic for anomaly ' + dataset.anomalies[j])
            plt.legend(loc="lower right")
            plt.savefig('ROC_valid_'+str(j)+'.png')
            


def split(df, based_on='subject_id', cv=5):
    splits = []
    based_on_uniq = df[based_on]#set(df[based_on].tolist())
    based_on_uniq = np.array_split(based_on_uniq, cv)
    for fold in based_on_uniq:
        splits.append(df[df[based_on] == fold.tolist()[0]])
    return splits

def horizontal_concat(features):
    return np.concatenate(features, axis = 0)

def vertical_concat(features, statistic="mean"):
    stacked_features = np.vstack(features)
    if statistic == "mean":
        return np.mean(stacked_features, axis = 1)

    elif statistic == "max":
        return np.amax(stacked_features, axis = 1)

    elif statistic == "min":
        return np.amin(stacked_features, axis = 1)

    else:
        raise ValueError('Invalid string given to vertical concat!')
        return None

def get_ROC_PR(y_score, y_test, fold, clf, concat, anomalies):
    print('y_score shape: ', y_score.shape)
    print('y_test shape: ', y_test.shape)
    print('y_test: ', y_test)
    print('y_score: ', y_score)
    
    FPR, TPR = dict(), dict()
    roc_auc, thresholds, precisions, recalls, pr_thresh = dict(), dict(), dict(), dict(), dict()
    best_thresholds = dict()

    if y_score.shape[0] != len(anomalies):
        y_score = np.transpose(y_score)
    if y_test.shape[0] != len(anomalies):
        y_test = np.transpose(y_test)

    for j, row in enumerate(y_test):
        FPR[j], TPR[j], thresholds[j] = roc_curve(y_test[j], y_score[j])
        precisions[j], recalls[j], pr_thresh[j] = precision_recall_curve(y_test[j], y_score[j])
        roc_auc[j] = auc(FPR[j], TPR[j])
    

    n = y_test.shape[1]
    thresh_opt = []
    for j in range(len(FPR)):
        plt.figure()
        lw = 2
        gmeans = np.sqrt(TPR[j] * (1 - FPR[j]))
        ix = np.argmax(gmeans)
        print('Best thresh, gmeans for: ', anomalies[j], thresholds[j][ix], gmeans[ix])
        print('Best TPR, FPR for ', anomalies[j], TPR[j][ix], FPR[j][ix])
        #print('Best F1 for ', anomalies[j], ": ", TPR[j][ix]*n / (TPR[j][ix]*n + .5*(FPR[j][ix]*n + (1-TPR[j][ix])*n)))
        thresh_opt.append(thresholds[j][ix])

        plt.plot(FPR[j], TPR[j], color = 'darkorange',
                 lw = lw, label = 'ROC Curve (area = %0.2f)' % roc_auc[j])
        plt.plot([0,1], [0,1], color = 'navy', lw = lw, linestyle = '--')
        plt.scatter(FPR[j][ix], TPR[j][ix], marker = 'o', color = 'black', label = 'Best')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.title('ROC for ' + anomalies[j] + ' fold ' + str(fold + 1))
        plt.legend(loc = 'lower right')
        plt.savefig('ROC_fold_' + str(fold + 1) + '_' + str(j) + '_' + clf + '_' + concat +  '.png')
        plt.clf()

    print('Beginning PR curve generation')
    for j in range(len(precisions)):
        plt.figure()
        lw = 2
        area = auc(recalls[j], precisions[j])
        fscores = (2. * precisions[j] * recalls[j]) / (precisions[j] + recalls[j])
        ix = np.argmax(fscores)
        dists = np.abs(precisions[j] - recalls[j])
        ix2 = np.argmin(dists)
        print('Best precision-recall maximizing f-score for ', anomalies[j], ': ', precisions[j][ix], recalls[j][ix])
        print('Best p-r minimzing d(p,r) for anomalies[j] ', anomalies[j], ': ', precisions[j][ix2], recalls[j][ix2])
        plt.plot(recalls[j], precisions[j], color = 'darkorange',
                 lw = lw, label = 'PR curves (area = %0.2f)' % area)
        plt.plot(recalls[j][ix], precisions[j][ix], marker='o', color='black', label='Best')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.ylabel('Precision')
        plt.xlabel('Recall')
        plt.title('PR curve for ' + anomalies[j] + ' fold ' + str(fold + 1))
        plt.legend(loc = 'lower left')
        plt.savefig('PR_fold_' + str(fold + 1) + '_' + str(j)  + '_' + clf + '_' + concat +  '.png')
        plt.clf()
        

    return thresh_opt


def get_train_acc(y_score, y_test, fold, clf, concat, anomalies):
    FPR, TPR = dict(), dict()
    roc_auc, thresholds, precisions, recalls, pr_thresh = dict(), dict(), dict(), dict(), dict()
    best_thresholds = dict()

    if y_score.shape[0] != len(anomalies):
        y_score = np.transpose(y_score)
    if y_test.shape[0] != len(anomalies):
        y_test = np.transpose(y_test)

    print('y_score: ', y_score)
    print('y_test: ', y_test)

    for j, row in enumerate(y_test):
        FPR[j], TPR[j], thresholds[j] = roc_curve(y_test[j], y_score[j])
        precisions[j], recalls[j], pr_thresh[j] = precision_recall_curve(y_test[j], y_score[j])
        roc_auc[j] = auc(FPR[j], TPR[j])
    

    n = y_test.shape[1]
    thresh_opt = []
    for j in range(len(FPR)):
        plt.figure()
        lw = 2
        gmeans = np.sqrt(TPR[j] * (1 - FPR[j]))
        ix = np.argmax(gmeans)
        thresh_opt.append(thresholds[j][ix])

    y_thresh = np.zeros((y_score.shape[0], y_score.shape[1]))
    for j in range(y_score.shape[0]):
        for k in range(y_score.shape[1]):
            if not np.isnan(thresh_opt[j]):
                if y_score[j][k] > thresh_opt[j]:
                    y_thresh[j][k] = 1
                else:
                    y_thresh[j][k] = 0
            if y_score[j][k] > .5:
                y_thresh[j][k] =1 
            else:
                y_thresh[j][k] = 0

    print('Training performance: ')
    return perf_measure(y_test, y_thresh, anomalies,verbose=False)

def plot_grad_flow(named_parameters,epoch):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.cpu().abs().mean())
            max_grads.append(p.grad.cpu().abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])

    plt.savefig('grads_'+str(epoch)+'_.png')
    plt.ylim(bottom = -0.001, top=max(max_grads)) # zoom in on the lower gradient regions
    plt.savefig('grads_'+str(epoch)+'_zoom_.png')
    plt.ylim(bottom = -.001, top = .02)

'''
133    pos_weights = np.ones_like(dataset.label_proportions)
134    neg_counts = [dataset.nsamples - pos_count for pos_count in dataset.label_proportions]
135    for cdx, (pos_count, neg_count) in enumerate(zip(dataset.label_proportions, neg_counts)):
136        pos_weights[cdx] = neg_count / (pos_count + 1e-5)
137
138    pos_weights = torch.as_tensor(pos_weights, dtype=torch.float)
139    print('pos weights: ', pos_weights)
140
'''
def compute_pos_weights(train_label_cts):
    weight_set = []
    for i, label_cts in enumerate(train_label_cts):
        n = sum(label_cts)
        pos_weights = np.ones_like(label_cts)
        neg_counts = [n - count for count in label_cts]
        for cdx, (pos_count, neg_count) in enumerate(zip(label_cts, neg_counts)):
            pos_weights[cdx] = neg_count / (pos_count + 1e-5)

        pos_weights = torch.as_tensor(pos_weights, dtype=torch.float)
        weight_set.append(pos_weights)
    
    return weight_set
