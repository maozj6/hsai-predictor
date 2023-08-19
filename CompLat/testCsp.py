# This is a sample Python script.
from vae import VAE
from sklearn.metrics import precision_score, recall_score, f1_score,accuracy_score, confusion_matrix
from tqdm import tqdm
from CarRacingDQNAgent import CarRacingDQNAgent
from torch.optim.lr_scheduler import ReduceLROnPlateau
from learning import EarlyStopping
import torch
import torch.nn as nn
import cv2

import torch.nn.functional as f
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from randomloader import RolloutSequenceDataset
import argparse
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
class latent_lstm(nn.Module):
    """ MDRNN model for multi steps forward """
    def __init__(self, latents=64, actions=3, hiddens=256):
        super().__init__()

        self.rnn = nn.LSTM(latents + actions, hiddens)
        self.fc = nn.Linear(
            hiddens, 64)

    def forward(self, actions, latents): # pylint: disable=arguments-differ
        seq_len, bs = actions.size(0), actions.size(1)

        ins = torch.cat([actions, latents], dim=-1)
        outs, _ = self.rnn(ins)
        out=self.fc(outs)
        return out


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

def main(step,test_path,evapath,rnnpath,vaepath):


    agent = CarRacingDQNAgent(epsilon=0)  # Set epsilon to 0 to ensure all actions are instructed by the agent
    agent.load("trial_400.h5")


    best = torch.load(evapath)
    correct = 0
    total = 0
    net = EvaNet()

    net = net.to(device)
    net.load_state_dict(best["state_dict"])
    test_preds = []
    test_trues = []

    vae = VAE(1, 32).to(device)
    best = torch.load(vaepath)
    vae.load_state_dict(best["state_dict"])

    vae.eval()

    decoder = vae.decoder
    #
    # train_path = args.train
    # test_path = args.test
    rnn = latent_lstm()
    bestrnn = torch.load(rnnpath)
    rnn.load_state_dict(bestrnn["state_dict"])


    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(rnn.parameters(), lr=0.001)
    # Training loop
    num_epochs = 301

    test_dataset = RolloutSequenceDataset(test_path, seq_len=15, leng=step)
    test_dataset.load_next_buffer()

    test_loader = DataLoader(test_dataset, batch_size=batchsize, shuffle=True, drop_last=True)
    rnn = rnn.to(device)
    train_loss = []
    test_loss = []

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
    earlystopping = EarlyStopping('min', patience=15)
    out_result = np.zeros((22))
    tp = np.zeros((22))
    tn = np.zeros((22))
    fp = np.zeros((22))
    fn = np.zeros((22))
    positive = 0
    dist = np.zeros((22))
    p22 = np.zeros((22))
    count = 0
    rnn.eval()
    for epoch in range(1):
        y_test=[]
        y_pred=[]
        y_sft=[]
        pbar = tqdm(total=len(test_loader),
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} {postfix}')
        pbar.set_description("Testing on step "+str(step) )

        for data in test_loader:  # Iterate over your training dataset
            save = []
            inputs, acts, next_obs, safes = data
            safesnp=safes.cpu().detach().numpy()
            inputs = inputs.to(device)
            obs = inputs.float()
            next_obs = next_obs.to(device)
            next_obs = next_obs.float()
            obs, next_obs = [
                f.upsample(x.view(-1, 1, 64, 64), size=64,
                           mode='bilinear', align_corners=True)
                for x in (obs, next_obs)]

            obs_z, next_obs_z = [
                vae(x)[0] for x in (obs, next_obs)]
            # zs=vae(inputs)
            zoutputs = rnn(acts.view((batchsize, 15, 3)).transpose(1, 0).to(device),
                           obs_z.view(batchsize, 15, 64).transpose(1, 0))
            save.append(zoutputs.cpu().detach().numpy()[14][0])
            test_img = zoutputs.transpose(1, 0).detach()[0]
            for savei in range(step*10-1):
                tensor_a = obs_z.view(15, 64)[1:15]
                tensor_b = zoutputs.view(15, 64)[14]
                next = torch.cat((tensor_a, tensor_b.view(1, 64)), 0)
                mu = next[-4:-1, 0:32]
                logsigma = next[-4:-1, 32:64]
                sigma = logsigma.exp()
                eps = torch.randn_like(sigma)
                z = eps.mul(sigma).add_(mu)
                recon_x_single = decoder(z.to(device))
                act_input=recon_x_single.cpu().detach().numpy()
                img1= cv2.resize(act_input[0][0],(96,96))
                img2 = cv2.resize(act_input[1][0], (96, 96))
                img3 = cv2.resize(act_input[2][0], (96, 96))
                act_input2=np.array([img1,img2,img3])
                act_input2=np.transpose(act_input2, [2,1,0])
                action = agent.act(act_input2)
                acts=torch.zeros((15, 1, 3))
                acts[:,0,0]=action[0]
                acts[:, 0, 1] = action[1]
                acts[:, 0, 2] = action[2]
                zoutputs = rnn(acts.to(device), next.view(15, 1, 64))
                obs_z = next
                save.append(zoutputs.cpu().detach().numpy()[14][0])
                test_img = zoutputs
            save_test = torch.tensor(np.array(save))
            mu = save_test[:, 0:32]
            logsigma = save_test[:, 32:64]

            sigma = logsigma.exp()
            eps = torch.randn_like(sigma)
            z = eps.mul(sigma).add_(mu)

            recon_x = decoder(z.to(device))
            # save_test = decoder(save_test.to(device))
            cats = inputs[0]
            saved_pred = torch.cat((cats.view(15, 1, 64, 64), recon_x), 0)
            # save_image(saved_pred.view(215, 1, 64, 64),
            #            'lat_pred_thr2/origin_' + str(epoch) + '.png')

            out_labels = net(recon_x)

            _, predicted = torch.max(out_labels.data, 1)

            predicted_np = predicted.cpu().detach().numpy()
            # print(safesnp[0])
            safe_pos = np.where(predicted_np == 0)
            if len(safe_pos[0]) > 0:
                if safesnp[0]==1:
                    y_test.append(1)
                    y_pred.append(0)
                    y_sft.append(out_labels.cpu().detach().numpy()[safe_pos[0]])

                if safesnp[0]==0:
                    y_test.append(0)
                    y_pred.append(0)
                    y_sft.append(out_labels.cpu().detach().numpy()[safe_pos[0]])

                # y_test.append(safesnp[0])
                # y_pred.append(safesnp[0])

            else:
                y_sft.append(out_labels.cpu().detach().numpy()[-1])
                if safesnp[0]==1:
                    y_test.append(1)
                    y_pred.append(1)
                if safesnp[0]==0:
                    y_test.append(0)
                    y_pred.append(1)
            pbar.update(1)

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    conf = confusion_matrix(y_test, y_pred)
    pbar.close()

    return acc, f1, precision, recall, y_sft, y_test,conf


if __name__ == '__main__':

    device="cuda"
    print(device)
    batchsize=1
    parser = argparse.ArgumentParser(description='test')

    parser.add_argument('--test',)
    parser.add_argument('--eva',)
    parser.add_argument('--vae',)
    parser.add_argument('--rnn',)
    args = parser.parse_args()

    test_path=args.test
    eva=args.eva
    vae=args.vae
    rnn=args.rnn

    accs=[]
    f1s=[]
    precs=[]
    recs=[]

    y_sft=[]
    y_lbl=[]
    confs = []

    args = parser.parse_args()
    for i in range(1,21):
        acc,f1,precision,recall,sfts,lbls,conf=main(i,test_path,eva,vae,rnn)
        confs.append(conf)
        accs.append(acc)
        f1s.append(f1)
        precs.append(precision)
        recs.append(recall)
        y_sft.append(sfts)
        y_lbl.append(lbls)
        print(accs)
        print(f1s)
        print(precs)
        print(recs)

    np.savez_compressed("test-compo-lstm-action-con3.npz", accs=accs, f1s=f1s, precs=precs, recs=recs, sft=y_sft, lbl=y_lbl,conf=confs)

