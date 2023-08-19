# This is a sample Python script.
from vae import VAE

from torch.optim.lr_scheduler import ReduceLROnPlateau
from learning import EarlyStopping
import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from myloader import RolloutSequenceDataset
from torchvision.utils import save_image
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
    args = parser.parse_args()

    test_path=args.test
    evapath=args.eva
    vaepath=args.vae
    rnnpath=args.rnn





    best=torch.load(evapath)
    correct=0
    total=0
    net = EvaNet()

    net = net.to(device)
    net.load_state_dict(best["state_dict"])
    test_preds = []
    test_trues = []



    vae = VAE(1, 32).to(device)
    best = torch.load(vaepath)
    vae.load_state_dict(best["state_dict"])


    vae.eval()

    decoder=vae.decoder

    train_path=args.train
    test_path=args.test
    rnn=latent_lstm()
    bestrnn = torch.load(rnnpath)
    rnn.load_state_dict(bestrnn["state_dict"])

    # ins=torch.rand((15,64,32))
    # acts=torch.rand((15,64,3))
    #
    # out=rnn(ins,acts)
    #
    # print("end")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(rnn.parameters(), lr=0.001)
    # Training loop
    num_epochs = 301

    test_dataset = RolloutSequenceDataset(test_path, seq_len=15,leng=0)
    test_dataset.load_next_buffer()

    test_loader = DataLoader(test_dataset, batch_size=batchsize, shuffle=True, drop_last=True)
    rnn = rnn.to(device)
    train_loss = []
    test_loss = []

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
    earlystopping = EarlyStopping('min', patience=15)
    out_result=np.zeros((22))
    tp=np.zeros((22))
    tn=np.zeros((22))
    fp=np.zeros((22))
    fn=np.zeros((22))
    positive=0
    dist=np.zeros((22))
    p22=np.zeros((22))
    count=0
    rnn.eval()
    for epoch in range(1):



        for data in test_loader:  # Iterate over your training dataset
            save = []
            inputs, acts, next_obs, safes = data
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
            zoutputs = rnn(acts.transpose(1, 0).to(device),
                           obs_z.view(batchsize, 15, 64).transpose(1, 0))
            save.append(zoutputs.cpu().detach().numpy()[14][0])
            test_img = zoutputs.transpose(1, 0).detach()[0]
            for savei in range(199):
                tensor_a=obs_z.view( 15, 64)[1:15]
                tensor_b=zoutputs.view( 15, 64)[14]
                next=torch.cat((tensor_a,tensor_b.view(1,64)),0)
                zoutputs = rnn(torch.zeros((15, 1, 3)).to(device),next.view(15, 1, 64))
                obs_z=next
                save.append(zoutputs.cpu().detach().numpy()[14][0])
                test_img = zoutputs
            save_test = torch.tensor(np.array(save))
            mu=save_test[:,0:32]
            logsigma=save_test[:,32:64]

            sigma = logsigma.exp()
            eps = torch.randn_like(sigma)
            z = eps.mul(sigma).add_(mu)

            recon_x = decoder(z.to(device))
            # save_test = decoder(save_test.to(device))
            cats=inputs[0]
            saved_pred=torch.cat((cats.view(15,1,64,64),recon_x),0)
            # save_image(saved_pred.view(215, 1, 64, 64),
            #            'lat_pred_thr2/origin_' + str(epoch) + '.png')

            out_labels=net(recon_x)

            _, predicted = torch.max(out_labels.data, 1)

            predicted_np=predicted.cpu().detach().numpy()
            safe_pos = np.where(predicted_np == 0)
            if len(safe_pos[0])>0:
                first=safe_pos[0][0]
            else:
                first=200
            # first=safe_pos[0][0]
            twentyth=int(first/10)
            predicted_labels=[]
            for ii in range(20):
                if ii<twentyth:
                    predicted_labels.append(1)
                else:
                    predicted_labels.append(0)
            real_label_matrix=safes.cpu().detach().numpy()

            for jindex in range(20):
                # print(jindex)
                # print(label_matrix)
                # print(pred_matrix)
                if real_label_matrix[0][jindex+1]==predicted_labels[jindex]:
                    out_result[jindex]=out_result[jindex]+1
                if real_label_matrix[0][jindex+1]==1 and predicted_labels[jindex]==1:
                    tp[jindex]=tp[jindex]+1
                if real_label_matrix[0][jindex+1]==0 and predicted_labels[jindex]==0:
                    tn[jindex]=tn[jindex]+1
                if real_label_matrix[0][jindex+1]==0 and predicted_labels[jindex]==1:
                    fp[jindex]=fp[jindex]+1
                if real_label_matrix[0][jindex+1]==1 and predicted_labels[jindex]==0:
                    fn[jindex]=fn[jindex]+1
            print(tp)
            print(tn)
            print(fp)
            print(fn)

            print(count)
            print(len(test_loader))
            count=count+1

            # save_image(outputsn.view(batchsize, 1, 64, 64),
            #            'samples/recon_' + str(epoch) + '.png')
    #
    # np.savez_compressed("19mayae.npz", train_loss=train_loss, test_loss=test_loss)
    # torch.save(rnn.state_dict(), "latrnn.pth")

    print("tp")
    print(tp)
    print(tn)
    print(fp)
    print(fn)
    print("dist")
    print(dist)
    print("posi_rate")
    print(p22)
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
