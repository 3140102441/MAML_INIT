# This code is modified from https://github.com/dragen1860/MAML-Pytorch and https://github.com/katerakelly/pytorch-maml 

import backbone
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from methods.meta_template import MetaTemplate

class PreModel(nn.Module):
    def __init__(self, model_func,lr = 0.001, dropp = 0.1, num_train_class = 64):
        self.feature  = model_func()
        self.loss_fn = nn.CrossEntropyLoss().cuda()
        self.dropout = nn.Dropout(p=dropp) 
        self.classifier = backbone.Linear_fw(self.feature.final_feat_dim, num_train_class)
        self.classifier.bias.data.fill_(0)
        
        self.train_lr = lr

    def forward(self,x):
        out  = self.feature.forward(x)
        out = self.dropout(out)
        scores  = self.classifier.forward(out)
        return scores

    def train_loop(self, epoch, train_loader, optimizer): #overwrite parrent function
        print_freq = 10
        correct_all = 0
        loss_all = 0
        optimizer.zero_grad()

        #train
        for i, (input, target) in enumerate(train_loader):
            input = input.cuda()
            target = target.cuda()

            # compute output
            output = self.forward(input)
            loss = self.loss_fn(output, target)

            # measure accuracy and record loss
            pred_q = F.softmax(output, dim=1).argmax(dim=1)
            correct = torch.eq(pred_q, target).sum().item()  # convert to numpy
            correct_all += correct
            loss_all += loss.data[0]

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % print_freq==0:
                print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f} | Acc {:f}%'.format(epoch, i, len(train_loader), loss_all/float(i+1),correct_all/float(i+1)/input.size(0)*100))
                      
    def test_loop(self, test_loader, return_std = False): #overwrite parrent function
        correct =0
        count = 0
        acc_all = []
        
        iter_num = len(test_loader) 
        for i, (x,_) in enumerate(test_loader):
            self.n_query = x.size(1) - self.n_support
            assert self.n_way  ==  x.size(0), "MAML do not support way change"
            correct_this, count_this = self.correct(x)
            acc_all.append(correct_this/ count_this *100 )

        acc_all  = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)
        acc_std  = np.std(acc_all)
        print('%d Test Acc = %4.2f%% +- %4.2f%%' %(iter_num,  acc_mean, 1.96* acc_std/np.sqrt(iter_num)))
        if return_std:
            return acc_mean, acc_std
        else:
            return acc_mean

