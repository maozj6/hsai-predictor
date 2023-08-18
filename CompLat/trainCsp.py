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

if __name__ == '__main__':

    device="cuda"
    print(device)
    batchsize=128
    parser = argparse.ArgumentParser(description='VAE Trainer')


    parser.add_argument('--train',
                       )
    parser.add_argument('--test',)

    args = parser.parse_args()

    vae = VAE(1, 32).to(device)
    best = torch.load("safe_vae_best.tar")
    vae.load_state_dict(best["state_dict"])


    vae.eval()

    decoder=vae.decoder

    train_path=args.train
    test_path=args.test
    rnn=latent_lstm()
    cur_best = None

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

    train_dataset = RolloutSequenceDataset(train_path,seq_len=15, leng=0)
    test_dataset = RolloutSequenceDataset(test_path, seq_len=15,leng=0)
    test_dataset.load_next_buffer()
    train_dataset.load_next_buffer()

    train_loader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batchsize, shuffle=True, drop_last=True)
    rnn = rnn.to(device)
    train_loss = []
    test_loss = []

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
    earlystopping = EarlyStopping('min', patience=15)

    for epoch in range(num_epochs):

        total_loss = 0
        total_num = 0
        pbar = tqdm(total=len(train_loader),
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} {postfix}')
        rnn.train()
        for data in train_loader:  # Iterate over your training dataset
            inputs, acts,next_obs,safes = data
            total_num = total_num + len(inputs)
            inputs = inputs.to(device)
            obs = inputs.float()
            next_obs = next_obs.to(device)
            next_obs = next_obs.float()
            obs, next_obs = [
                f.upsample(x.view(-1, 1, 64, 64), size=64,
                           mode='bilinear', align_corners=True)
                for x in (obs, next_obs)]

            obs_z,next_obs_z = [
                vae(x)[0] for x in (obs, next_obs)]
            # zs=vae(inputs)
            zoutputs = rnn(torch.zeros((batchsize,15,3)).transpose(1, 0).to(device),obs_z.view(batchsize, 15, 64).transpose(1, 0))
            # Forward pass
            loss_z=zoutputs[-1]
            loss_target= next_obs_z.view(batchsize, 15, 64).transpose(1, 0)
            # Compute loss
            loss = criterion(loss_z,loss_target[-1])
            total_loss = total_loss + loss.item()
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.update(1)

        # Print the epoch loss
        print(f"Training Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / total_num:.10f}")
        train_loss.append(total_loss / total_num)
        pbar.close()
        rnn.eval()
        total_loss = 0
        total_num = 0
        pbar = tqdm(total=len(test_loader),
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} {postfix}')
        for data in test_loader:  # Iterate over your training dataset
            inputs, acts, next_obs, safes = data
            total_num = total_num + len(inputs)
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
            zoutputs = rnn(torch.zeros((batchsize,15,3)).transpose(1, 0).to(device),
                           obs_z.view(batchsize, 15, 64).transpose(1, 0))
            # Forward pass

            # Compute loss
            loss = criterion(zoutputs, next_obs_z.view(batchsize, 15, 64).transpose(1, 0))
            total_loss = total_loss + loss.item()

            pbar.update(1)

        print(f"Test Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / total_num:.10f}")
        test_loss.append(total_loss / total_num)
        pbar.close()
        val_loss=total_loss/total_num
        scheduler.step(val_loss)
        earlystopping.step(val_loss)

        is_best = not cur_best or val_loss < cur_best
        if is_best:
            cur_best = val_loss

        save_checkpoint({
            'epoch': epoch,
            'state_dict': rnn.state_dict(),
            'train_loss': train_loss,
            'test_loss': val_loss,

            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'earlystopping': earlystopping.state_dict(),
        }, is_best, 'thr2_best.tar', 'thr2_checkpoint.tar')

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
                out = rnn(torch.zeros((15, 1, 3)).to(device),test_img.view(15, 1, 64))

                save.append(out.cpu().detach().numpy()[14][0])
                test_img = out
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
            save_image(saved_pred.view(215, 1, 64, 64),
                       'lat_pred_thr2/origin_' + str(epoch) + '.png')

            # save_image(outputsn.view(batchsize, 1, 64, 64),
            #            'samples/recon_' + str(epoch) + '.png')
            break
    #
    # np.savez_compressed("19mayae.npz", train_loss=train_loss, test_loss=test_loss)
    # torch.save(rnn.state_dict(), "latrnn.pth")


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
