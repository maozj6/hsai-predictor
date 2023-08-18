import math
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset
import numpy as np
from bisect import bisect
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from learning import EarlyStopping
from vae import VAE
from torchvision.utils import save_image
import argparse

import os
from os.path import join, exists
from os import mkdir
from sklearn.metrics import confusion_matrix
from PIL import Image
import cv2
import warnings
import matplotlib.pyplot as plt
import pandas as pd
from bisect import bisect
from myloader import RolloutObservationDataset
def loss_function(recon_x, x, mu, logsigma):
    """ VAE loss function """
    BCE = F.mse_loss(recon_x, x, reduction='sum' )

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + 2 * logsigma - mu.pow(2) - (2 * logsigma).exp())
    return BCE + KLD

def save_checkpoint(state, is_best, filename, best_filename):
    """ Save state in filename. Also save in best_filename if is_best. """
    torch.save(state, filename)
    if is_best:
        torch.save(state, best_filename)

class EvaNet(nn.Module):
    def __init__(self):
        super(EvaNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(2704, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256,2)
        self.soft=nn.Softmax(dim=1)

    def forward(self, x):
        input_size = x.size(0)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(input_size,-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x=self.soft(x)
        return x
def main(logdir,task):

    device="cuda"
    parser = argparse.ArgumentParser(description='VAE Trainer')
    #--train="/home/UFAD/z.mao/01dataset/thread_2/" --test="/home/UFAD/z.mao/013dataset/thread_2/"
    # parser.add_argument('--batch-size', type=int, default=32, metavar='N',
    #                     help='input batch size for training (default: 32)')
    # parser.add_argument('--epochs', type=int, default=1000, metavar='N',
    #                     help='number of epochs to train (default: 1000)')
    # parser.add_argument('--logdir', type=str, default='log3', help='Directory where results are logged')
    # parser.add_argument('--noreload', action='store_true',
    #                     help='Best model is not reloaded if specified')
    # parser.add_argument('--nosamples', action='store_true',
    #                     help='Does not save samples during training if specified')
    #--train="/home/UFAD/z.mao/01dataset/thread_2/" --test="/home/UFAD/z.mao/013dataset/thread_2/"

    parser.add_argument('--train', default="/home/mao/23Summer/code/Cali-predictors/RacingCar/data/train",
                        help='Best model is not reloaded if specified')
    parser.add_argument('--test',default="/home/mao/23Summer/code/Cali-predictors/RacingCar/data/train",
                        help='Does not save samples during training if specified')
    parser.add_argument('--log',default="logs/"+"safeVae" + "/",
                        help='log path')
    parser.add_argument('--eva',default="/home/mao/23Summer/code/Cali-predictors/RacingCar/models/eva.tar",
                        help='evaluator path')

    args = parser.parse_args()

    train_path=args.train
    test_path=args.test
    batch_size=64
    cur_best = None
    vae_dir = join(logdir, task)
    if not exists(vae_dir):
        mkdir(vae_dir)
        mkdir(join(vae_dir, 'samples'))
    device = 'cuda:0'
    print(device)
    trainacc=[]
    trainloss=[]
    train_celoss=[]
    train_reconloss=[]
    model = VAE(1, 32).to(device)
    net = EvaNet()

    net = net.to(device)
    best = torch.load("/home/mao/23Summer/code/Cali-predictors/RacingCar/models/eva.tar")
    net.load_state_dict(best["state_dict"])

    # net = Net()

    model = model.to(device)
    noreload=False
    reload_file = join(vae_dir, 'safe_vae_best.tar')
    optimizer = optim.Adam(model.parameters())

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
    earlystopping =EarlyStopping('min', patience=15)  # 关于 EarlyStopping 的代码可先看博客后面的内容
    if not noreload and exists(reload_file):
        state = torch.load(reload_file)
        print("Reloading model at epoch {}"
              ", with test error {}".format(
            state['epoch'],
            state['test_acc']))
        model.load_state_dict(state['state_dict'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        earlystopping.load_state_dict(state['earlystopping'])


    train_dataset=RolloutObservationDataset(train_path,leng=0)
    test_dataset=RolloutObservationDataset(test_path,leng=0)
    test_dataset.load_next_buffer()
    train_dataset.load_next_buffer()

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,drop_last = True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True,drop_last = True)

    #
    # train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True,drop_last = True)
    # test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True,drop_last = True)

    lf2=nn.CrossEntropyLoss()

    for epoch in range(401):
        running_loss = 0.0
        recon_loss=0
        ce_loss=0
        correct = 0
        total = 0
        losssum = 0
        model.train()
        # safe_monotor=0
        pbar = tqdm(total=len(train_loader),
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} {postfix}')
        for i, data in enumerate(train_loader, 0):
            total = total + len(data[0])
            inputs, labels,actions = data
            inputs, labels = Variable(inputs), Variable(labels)

            optimizer.zero_grad()  # 优化器清零
            inputs = inputs.float()
            inputs = inputs.to(device)
            labels = labels.to(device)

            recon_batch, mu, logvar = model(inputs.unsqueeze(1))
            predict_label=net(recon_batch)
            loss = loss_function(recon_batch, inputs.unsqueeze(1), mu, logvar)
            loss2 = lf2(predict_label, labels)
            total_loss = loss + 4096*loss2

            _, predicted = torch.max(predict_label.data, 1)
            correct += (predicted == labels).sum().item()

            total_loss.backward()
            optimizer.step()  # 优化
            running_loss += total_loss.item()
            losssum += total_loss.item()
            recon_loss+=loss.item()
            ce_loss+=loss2.item()
            if i % 200 == 199:
                pbar.set_description('training [%d %5d]  recon-loss: %.6f ce_loss:  %.6f acc: %.6f  lr: {:%.6f}:' % (epoch + 1, i + 1,  recon_loss / total,ce_loss / total,correct / total, optimizer.state_dict()['param_groups'][0]['lr']))
                pbar.update(200)

        trainloss.append(losssum/total)
        train_celoss.append(ce_loss/total)
        train_reconloss.append(recon_loss/total)
        trainacc.append(correct/total)
        pbar.close()

        model.eval()
        test_loss = 0

        test_preds = []
        test_trues = []
        test_reconloss=[]
        test_celoss=[]
        total = 0
        correct=0
        recon_loss=0
        ce_loss=0
        with torch.no_grad():
            pbar = tqdm(total=len(test_loader),
                        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} {postfix}')
            for i, data in enumerate(test_loader, 0):
                total = total + len(data[0])
                inputs, labels ,actions= data
                inputs, labels = Variable(inputs.unsqueeze(1)), Variable(labels)
                inputs = inputs.float()
                inputs = inputs.to(device)
                labels = labels.to(device)
                recon_batch, mu, logvar = model(inputs)
                predict_label=net(recon_batch)
                _, predicted = torch.max(predict_label.data, 1)
                correct += (predicted == labels).sum().item()
                test_preds.extend(predicted.detach().cpu().numpy())
                test_trues.extend(labels.detach().cpu().numpy())
                loss = loss_function(recon_batch, inputs, mu, logvar)
                loss2 = lf2(predict_label, labels)
                total_loss = loss + loss2
                recon_loss=recon_loss+loss
                ce_loss+=loss2
                test_loss += total_loss

                if i % 20 == 19:
                    pbar.set_description(
                        'test [%d %5d] recon-loss:%.7f celoss:%.7f  acc: %.6f  ' % (epoch + 1, i + 1, recon_loss/total,ce_loss/total,correct / total))
                    pbar.update(20)
        test_loss=test_loss/total
        pbar.close()
        test_celoss.append(ce_loss)
        test_reconloss.append(recon_loss)
        testacc=(correct/total)
        scheduler.step(test_loss)
        earlystopping.step(test_loss)
        conf_matrix = confusion_matrix(test_trues, test_preds)
        print(conf_matrix)

        best_filename = join(vae_dir, 'safe_vae_best.tar')
        filename = join(vae_dir, 'checkpoint.tar')
        is_best = not cur_best or testacc > cur_best
        if is_best:
            cur_best = testacc

        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'train_loss':trainloss,
            "train_acc":trainacc,
            'test_loss': test_loss,
            "test_acc":testacc,
            "test_recon_loss":test_reconloss,
            "test_ce_loss":test_celoss,
            "train_recon_loss":train_reconloss,
            "train_ce_loss":train_celoss,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'earlystopping': earlystopping.state_dict(),
            'matrix':conf_matrix
        }, is_best, filename, best_filename)

        with torch.no_grad():
            ASIZE, LSIZE, RSIZE, RED_SIZE, SIZE = \
                3, 32, 256, 64, 64
            sample = torch.randn(64, 32).to(device)
            sample = model.decoder(sample).cpu()
            save_image(sample.view(64, 1, RED_SIZE, RED_SIZE),
                       join(vae_dir, 'samples/sample_' + str(epoch) + '.png'))
            model.eval()
            with torch.no_grad():
                for i, data in enumerate(test_loader):
                    inputs, labels ,actions= data
                    inputs = Variable(inputs.unsqueeze(1))
                    inputs = inputs.float()
                    inputs = inputs.to(device)
                    recon_batch, mu, logvar = model(inputs)
                    #
                    # data = data.to(device)
                    # recon_batch, mu, logvar = model(data.float())
                    break
            save_image(inputs.view(64, 1, RED_SIZE, RED_SIZE),
                       join(vae_dir, 'samples/origin_' + str(epoch) + '.png'))
            save_image(recon_batch.view(64, 1, RED_SIZE, RED_SIZE),
                       join(vae_dir, 'samples/recon_' + str(epoch) + '.png'))
    print('finished training!')
    print("end")
if __name__ == '__main__':


    main("vaelog/",  "safeVae" + "/")

