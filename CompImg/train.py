import argparse
from Seq2Seq import Seq2Seq
from torchvision.utils import save_image

from torch.optim.lr_scheduler import ReduceLROnPlateau
from learning import EarlyStopping
import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from myloader import RolloutSequenceDataset2

def save_checkpoint(state, is_best, filename, best_filename):
    """ Save state in filename. Also save in best_filename if is_best. """
    torch.save(state, filename)
    if is_best:
        torch.save(state, best_filename)
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    batch_size=16
    parser = argparse.ArgumentParser(description='VAE Trainer')

    parser.add_argument('--train',
                    )
    parser.add_argument('--test',
                      )
    args = parser.parse_args()
    train_path=args.train
    test_path=args.test

    train_dataset=RolloutSequenceDataset2(train_path,seq_len=15)
    test_dataset=RolloutSequenceDataset2(test_path,seq_len=15)
    test_dataset.load_next_buffer()
    train_dataset.load_next_buffer()


    def collate(batch):
        # Add channel dim, scale pixels between 0 and 1, send to GPU
        batch=np.array(batch)
        #
        batch = torch.tensor(batch)
        # batch = batch / 255.0
        batch = batch.view(batch_size, 15, 64, 64)
        batch = batch.to(device)

        # Randomly pick 10 frames as input, 11th frame is target
        rand = np.random.randint(10, 15)
        return batch[:, rand - 10:rand].float(), batch[:, rand].float()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,drop_last = True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True,drop_last = True)


        # Training Data Loader




    cur_best = None

    # The input video frames are grayscale, thus single channel
    model = Seq2Seq(num_channels=1, num_kernels=64,
                    kernel_size=(3, 3), padding=(1, 1), activation="relu",
                    frame_size=(64, 64), num_layers=3).to(device)
    optim = Adam(model.parameters(), lr=1e-4)

    # Binary Cross Entropy, target pixel values either 0 or 1
    criterion = nn.MSELoss()
    scheduler = ReduceLROnPlateau(optim, mode='min', factor=0.1, patience=5, verbose=True)
    earlystopping =EarlyStopping('min', patience=15)  # 关于 EarlyStopping 的代码可先看博客后面的内容
    num_epochs = 61

    for epoch in range(1, num_epochs + 1):
        print(epoch)
        starttime = datetime.datetime.now()

        train_loss = 0
        model.train()


        for batch_num,data  in enumerate(train_loader, 1):
            input,target=data
            input=input.to(device).float()
            target=target[:,-1].to(device).float()

            if batch_num%100==0:
                print(str(batch_num)+"/"+str(len(train_loader))+"  "+str((datetime.datetime.now()-starttime).seconds))
            output = model(input.unsqueeze(1))
            loss = criterion(output.flatten(), target.flatten())
            loss.backward()
            optim.step()
            optim.zero_grad()
            train_loss += loss.item()
        train_loss /= len(train_loader.dataset)

        val_loss = 0
        model.eval()
        with torch.no_grad():
            for batch_num, data in enumerate(test_loader, 1):

                input, target = data
                input = input.to(device).float()
                target = target[:, -1].to(device).float()

                if batch_num % 100 == 0:
                    print(str(batch_num) + "/" + str(len(train_loader)) + "  " + str(
                        (datetime.datetime.now() - starttime).seconds))
                output = model(input.unsqueeze(1))
                loss = criterion(output.flatten(), target.flatten())
                train_loss += loss.item()

                val_loss += loss.item()
        val_loss /= len(test_loader.dataset)

        print("Epoch:{} Training Loss:{:.2f} Validation Loss:{:.2f}\n".format(
            epoch, train_loss, val_loss))

        # torch.save(model, "./model/RCNN" + str(epoch) + ".pth")  # 保存整个模型
        # torch.save(model.state_dict(), "RCNN" +  ".pth")
        endtime = datetime.datetime.now()
        print(endtime - starttime)
        scheduler.step(val_loss)
        earlystopping.step(val_loss)

        is_best = not cur_best or val_loss < cur_best
        if is_best:
            cur_best = val_loss

        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'train_loss': train_loss,
            'test_loss': val_loss,

            'optimizer': optim.state_dict(),
            'scheduler': scheduler.state_dict(),
            'earlystopping': earlystopping.state_dict(),
        }, is_best, 'thr2-action-best0712.tar', 'thr2-action-checkpoint0712.tar')


        with torch.no_grad():

                for i, data in enumerate(test_loader):
                    input, target = data
                    input = input.to(device).float()
                    target = target[:, -1].to(device).float()
                    output = model(input.unsqueeze(1))
                    save=torch.cat((target.view(batch_size,1,64,64),output),0)

                    newinput = input[0]
                    newinput=newinput.unsqueeze(0)
                    pred=[]
                    for j in range(200):
                        tmp = model(newinput.unsqueeze(1))
                        pred.append(tmp.cpu().detach().numpy())
                        newinput=torch.cat((newinput[:,0:9],tmp),1)
                        print("end")
                    break
                pred_tensor=torch.tensor(np.array(pred))
                save_image(pred_tensor.view(200, 1, 64, 64),
                       ( 'conv-lstm/thr2/sequence' + str(epoch) + '.png'))
                save_image(save.view(32, 1, 64, 64),
                       ( 'conv-lstm/thr2/both' + str(epoch) + '.png'))
                save_image(target.view(batch_size, 1, 64, 64),
                       ( 'conv-lstm/thr2/target' + str(epoch) + '.png'))
                save_image(output.view(batch_size, 1, 64, 64),
                   ('conv-lstm/thr2/pred' + str(epoch) + '.png'))
