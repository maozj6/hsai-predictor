import argparse
from Seq2Seq import Seq2Seq

from torch.optim.lr_scheduler import ReduceLROnPlateau
from learning import EarlyStopping
import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from myloader import RolloutSequenceDataset
from torch.nn import functional as F
import cv2
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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    batch_size=1
    parser = argparse.ArgumentParser(description='VAE Trainer')

    parser.add_argument('--test', )
    args = parser.parse_args()
    test_path=args.test
    test_dataset=RolloutSequenceDataset(test_path,seq_len=10)
    test_dataset.load_next_buffer()

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True,drop_last = True)
    best=torch.load("models/best.tar")
    correct=0
    total=0
    net = EvaNet()

    net = net.to(device)
    net.load_state_dict(best["state_dict"])

    cur_best = None

    # The input video frames are grayscale, thus single channel
    model = Seq2Seq(num_channels=1, num_kernels=64,
                    kernel_size=(3, 3), padding=(1, 1), activation="relu",
                    frame_size=(64, 64), num_layers=3).to(device)
    state=torch.load("thr2-checkpoint.tar")
    model.load_state_dict(state['state_dict'])
    optim = Adam(model.parameters(), lr=1e-4)



    # Binary Cross Entropy, target pixel values either 0 or 1
    out_result=np.zeros((22))
    tp=np.zeros((22))
    tn=np.zeros((22))
    fp=np.zeros((22))
    fn=np.zeros((22))
    positive=0
    dist=np.zeros((22))
    p22=np.zeros((22))
    count=0
    model.eval()
    tmp_gurad=0
    for epoch in range(0, 1):
        print(epoch)
        starttime = datetime.datetime.now()

        train_loss = 0




        with torch.no_grad():
                guard=0
                for i, data in enumerate(test_loader):
                    # guard=guard+1
                    # if guard>51:
                    #     break
                    input, action, target, safes=data
                    input = input.to(device).float()
                    initI=input.cpu().detach().numpy()
                    # inimg.append(initI)
                    target = target[:, -1].to(device).float()
                    output = model(input.unsqueeze(1))
                    save=torch.cat((target.view(1,1,64,64),output),0)

                    newinput = input[0]
                    newinput=newinput.unsqueeze(0)
                    pred=[]
                    for j in range(200):
                        tmp = model(newinput.unsqueeze(1))
                        tmp2=tmp.cpu().detach().numpy()
                        pred.append(tmp2)
                        tmp_gurad=tmp_gurad+1
                        newinput=torch.cat((newinput[:,0:9],tmp),1)
                        # print("end")

                    pred_tensor=torch.tensor(np.array(pred))
                    pred_labels=net(pred_tensor.squeeze(1).float().to(device))
                    save_single=np.array(pred)
                    save_single=np.squeeze(save_single,axis=1)
                    # allsave.append(save_single)
                    print("")
                    _, predicted = torch.max(pred_labels.data, 1)

                    predicted_np = predicted.cpu().detach().numpy()
                    safe_pos = np.where(predicted_np == 0)
                    # allpred.append(predicted_np)
                    if len(safe_pos[0]) > 0:
                        first = safe_pos[0][0]
                    else:
                        first = 200
                    # first=safe_pos[0][0]
                    # allfirst.append(first)
                    twentyth = int(first / 10)
                    predicted_labels = []
                    for ii in range(20):
                        if ii < twentyth:
                            predicted_labels.append(1)
                        else:
                            predicted_labels.append(0)
                    real_label_matrix = safes.cpu().detach().numpy()

                    for jindex in range(20):
                        # print(jindex)
                        # print(label_matrix)
                        # print(pred_matrix)
                        if real_label_matrix[0][jindex + 1] == predicted_labels[jindex]:
                            out_result[jindex] = out_result[jindex] + 1
                        if real_label_matrix[0][jindex + 1] == 1 and predicted_labels[jindex] == 1:
                            tp[jindex] = tp[jindex] + 1
                        if real_label_matrix[0][jindex + 1] == 0 and predicted_labels[jindex] == 0:
                            tn[jindex] = tn[jindex] + 1
                        if real_label_matrix[0][jindex + 1] == 0 and predicted_labels[jindex] == 1:
                            fp[jindex] = fp[jindex] + 1
                        if real_label_matrix[0][jindex + 1] == 1 and predicted_labels[jindex] == 0:
                            fn[jindex] = fn[jindex] + 1
                    print(tp)
                    print(tn)
                    print(fp)
                    print(fn)

                    print(count)
                    print(len(test_loader))
                    count = count + 1


