# This is a sample Python script.
from vae import VAE
import sys
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
from randomloader import RolloutSequenceDataset
import argparse
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
class latent_lstm2(nn.Module):
    """ MDRNN model for multi steps forward """
    def __init__(self, latents=64, actions=1, hiddens=256):
        super().__init__()

        self.rnn = nn.LSTM(latents + actions, hiddens)
        self.fc = nn.Linear(
            1280, 2)
        self.sft=nn.Softmax()

    def forward(self, actions, latents): # pylint: disable=arguments-differ
        seq_len, bs = actions.size(0), actions.size(1)

        ins = torch.cat([actions.unsqueeze(2), latents], dim=-1)
        outs, _ = self.rnn(ins)
        outs=outs.transpose(1, 0)
        outs=torch.flatten(outs,start_dim=1)
        out=self.fc(outs)
        out=self.sft(out)

        return out
class latent_lstm(nn.Module):
    """ MDRNN model for multi steps forward """
    def __init__(self, latents=64, actions=3, hiddens=256):
        super().__init__()

        self.rnn = nn.LSTM(latents + actions, hiddens)
        self.fc = nn.Linear(
            15*256, 2)
        self.sft=nn.Softmax()

    def forward(self, actions, latents): # pylint: disable=arguments-differ
        seq_len, bs = actions.size(0), actions.size(1)

        ins = torch.cat([actions, latents], dim=-1)
        outs, _ = self.rnn(ins)
        outs=outs.transpose(1, 0)
        outs=torch.flatten(outs,start_dim=1)
        out=self.fc(outs)
        out=self.sft(out)

        return out

def save_checkpoint(state, is_best, filename, best_filename):
    """ Save state in filename. Also save in best_filename if is_best. """
    torch.save(state, filename)
    if is_best:
        torch.save(state, best_filename)

def main(train_path,test_path,step_save_name,step,epochs,task,vae_path):

    if task==1:
        rnn = latent_lstm()
        seq_len=15
    elif task==2:
        rnn = latent_lstm2()
        seq_len=5
    else:
        print("Invalid task number! `", sys.exc_info()[0])

    vae = VAE(1, 32).to(device)



    best = torch.load(vae_path)
    vae.load_state_dict(best["state_dict"])

    vae.eval()

    decoder = vae.decoder

    cur_best = None


    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(rnn.parameters(), lr=0.001)
    # Training loop
    num_epochs = epochs

    train_dataset = RolloutSequenceDataset(train_path, seq_len=seq_len, leng=step)
    test_dataset = RolloutSequenceDataset(test_path, seq_len=seq_len, leng=step)
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
        correct=0
        pbar = tqdm(total=len(train_loader),
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} {postfix}')
        rnn.train()
        for data in train_loader:  # Iterate over your training dataset
            inputs, acts, next_obs, safes = data
            total_num = total_num + len(inputs)
            safes=safes.to(device)
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
            zoutputs = rnn(acts.transpose(1, 0).to(device), obs_z.view(batchsize, seq_len, 64).transpose(1, 0))
            # Forward pass
            _, predicted = torch.max(zoutputs.data, 1)
            correct += (predicted == safes).sum().item()
            # Compute loss
            loss = criterion(zoutputs, safes.to(device))
            total_loss = total_loss + loss.item()
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.update(1)

        # Print the epoch loss
        print(f"Training Epoch [{epoch + 1}/{num_epochs}],,Acc:{correct/total_num } , Loss: {total_loss / total_num:.10f}")
        train_loss.append(total_loss / total_num)
        pbar.close()
        rnn.eval()
        total_loss = 0
        total_num = 0
        correct=0
        pbar = tqdm(total=len(test_loader),
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} {postfix}')
        for data in test_loader:  # Iterate over your training dataset
            inputs, acts, next_obs, safes = data
            total_num = total_num + len(inputs)
            inputs = inputs.to(device)
            obs = inputs.float()
            next_obs = next_obs.to(device)
            next_obs = next_obs.float()
            safes=safes.to(device)

            obs, next_obs = [
                f.upsample(x.view(-1, 1, 64, 64), size=64,
                           mode='bilinear', align_corners=True)
                for x in (obs, next_obs)]

            obs_z, next_obs_z = [
                vae(x)[0] for x in (obs, next_obs)]
            # zs=vae(inputs)
            zoutputs = rnn(acts.transpose(1, 0).to(device),
                           obs_z.view(batchsize, seq_len, 64).transpose(1, 0))

            _, predicted = torch.max(zoutputs.data, 1)
            correct += (predicted == safes).sum().item()
            # Forward pass

            # Compute loss
            loss = criterion(zoutputs, safes.to(device))
            total_loss = total_loss + loss.item()

            pbar.update(1)

        print(f"Test Epoch [{epoch + 1}/{num_epochs}],Acc:{correct/total_num } Loss: {total_loss / total_num:.10f}")
        test_loss.append(total_loss / total_num)
        pbar.close()
        val_loss = total_loss / total_num
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
        }, is_best, step_save_name+'-best.tar', step_save_name+'-checkpooint.tar')

if __name__ == '__main__':

    device="cuda"
    print(device)
    batchsize=128
    parser = argparse.ArgumentParser(description='VAE Trainer')
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
    parser = argparse.ArgumentParser(description='predictor trainer')
    parser.add_argument('--train', default="",
                      )
    parser.add_argument('--test',default="",
                      )
    parser.add_argument('--save',default="",
                      )
    parser.add_argument('--epochs',default=101,
                       )
    parser.add_argument('--steps',default=9,
                      )
    parser.add_argument('--task',default=1,
                      )
    parser.add_argument('--vae',
                      )

    args = parser.parse_args()


    train_path=args.train
    test_path=args.test
    save_path=args.save
    steps = int(args.steps)
    epochs=int(args.epochs)
    task=int(args.task)
    # print(epochs)
    # print(task)
    vae_path=args.vae
    for i in range(0,steps):
        main(train_path,test_path,save_path+"step"+str(i),i,epochs,task,vae_path)


