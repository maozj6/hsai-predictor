import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from myloader import RolloutObservationDataset
from os.path import join, exists
from os import mkdir
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
import os
batch_size=128
seq_length=15

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
def save_checkpoint(state, is_best, filename, best_filename):
    """ Save state in filename. Also save in best_filename if is_best. """
    torch.save(state, filename)
    if is_best:
        torch.save(state, best_filename)
#

def main(logdir,leng,trainpath,testpath):
    cur_best = None
    vae_dir = logdir
    print(os.getcwd())

    if not exists(vae_dir):
        mkdir(vae_dir)
    vae_dir=vae_dir
    device =  "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    trainacc=[]
    trainloss=[]
    testacclist=[]
    testlosslist=[]
    mpreclist=[]
    mfnlist=[]
    fone_list=[]
    net = EvaNet()

    net = net.to(device)

    optimizer = optim.Adam(net.parameters())


    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    cirterion = nn.CrossEntropyLoss()

    train_dataset=RolloutObservationDataset(trainpath,leng=leng)
    test_dataset=RolloutObservationDataset(testpath,leng=leng)
    test_dataset.load_next_buffer()
    train_dataset.load_next_buffer()

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,drop_last = True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True,drop_last = True)


    for epoch in range(101):
        running_loss = 0.0
        correct = 0
        total = 0
        losssum = 0
        net.train()
        # safe_monotor=0
        pbar = tqdm(total=len(train_loader),
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} {postfix}')
        for i, data in enumerate(train_loader, 0):
            total = total + len(data[0])
            inputs, labels,actions = data
            inputs, labels = Variable(inputs), Variable(labels)

            optimizer.zero_grad()
            inputs = inputs.float()

            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = net(inputs.view(batch_size,1,64,64))
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()

            loss = cirterion(outputs,labels )
            loss.backward()
            optimizer.step()  # 优化
            running_loss += loss.item()
            losssum += loss.item()
            if i % 200 == 199:
                pbar.set_description('training [%d %5d] acc: %.3f  loss: %.3f' % (epoch + 1, i + 1, correct / total, running_loss / 20))
                pbar.update(200)

        trainloss.append(losssum/total)
        trainacc.append(correct/total)
        pbar.close()

        net.eval()

        test_preds = []
        test_trues = []
        losssum = 0

        with torch.no_grad():
            pbar = tqdm(total=len(test_loader),
                        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} {postfix}')
            for i, data in enumerate(test_loader, 0):
                total = total + len(data[0])
                inputs, labels,actions = data
                inputs, labels = Variable(inputs.view(batch_size,1,64,64)), Variable(labels)
                inputs = inputs.float()
                inputs = inputs.to(device)
                labels = labels.to(device)
                # labels = labels.float()
                # labels=labels.view(batch_size,1)
                outputs = net(inputs)
                loss = cirterion(outputs, labels)
                losssum += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()
                test_preds.extend(predicted.detach().cpu().numpy())
                test_trues.extend(labels.detach().cpu().numpy())
                if i % 20 == 19:
                    pbar.set_description(
                        'leng %d test [%d %5d] acc: %.5f lr: %.7f ' % (leng,epoch + 1, i + 1, correct / total,optimizer.state_dict()['param_groups'][0]['lr']))
                    pbar.update(20)

        pbar.close()
        conf_matrix = confusion_matrix(test_trues, test_preds)
        f1_micro = f1_score(test_trues, test_preds, average='micro')

        mprec=conf_matrix[0][1]/(conf_matrix[0][1]+conf_matrix[1][1])
        minusfn=conf_matrix[1][0]/(conf_matrix[1][0]+conf_matrix[1][1])
        testacc=(correct/total)
        testacclist.append(testacc)
        test_loss=losssum/total
        testlosslist.append(test_loss)
        scheduler.step(test_loss)
        # earlystopping.step(test_loss)
        # print(conf_matrix)
        print("len: "+str(leng)+"epoch: "+str(epoch)+"acc: "+str(testacc))
        mpreclist.append(mprec)
        mfnlist.append(minusfn)
        fone_list.append(f1_micro)
        best_filename = join(vae_dir, 'best.tar')
        filename = join(vae_dir, 'checkpoint.tar')
        is_best = not cur_best or test_loss < cur_best
        if is_best:
            cur_best = test_loss

        save_checkpoint({
            'epoch': epoch,
            'state_dict': net.state_dict(),
            'train_loss':trainloss,
            "train_acc":trainacc,
            'test_loss': test_loss,
            "test_acc":testacc,
            "testacclist":testacclist,
            "testlosslist":testlosslist,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            # 'earlystopping': earlystopping.state_dict(),
            'matrix':conf_matrix,
            "lr":optimizer.state_dict()['param_groups'][0]['lr'],
            "one_minus_prec":mpreclist,
            "one_minus_fn": mfnlist,
            "fone_list": fone_list,

        }, is_best, filename, best_filename)
        # if earlystopping.stop:
        #     print("End of Training because of early stopping at epoch {}".format(epoch))
        #     break
    print('finished training!')
    print("end")

    plt.subplot(1, 2, 1)
    plt.plot(trainacc, label="train_acc")
    plt.plot(testacclist, label="test_acc")
    plt.plot(mpreclist, label="test_fp")
    plt.plot(mfnlist, label="test_fn")
    plt.plot(fone_list, label="test_f1")
    plt.grid()

    plt.legend()  # 显示图例

    plt.title("plot 1")
    plt.subplot(1, 2, 2)
    plt.title("plot 2")
    plt.plot(trainloss, label="train_loss")
    plt.plot(testlosslist, label="test_loss")
    plt.grid()

    plt.legend()  # 显示图例

    plt.suptitle("Evaluator")
    plt.show()
    plt.savefig('./eva_result.jpg')

    # testacc = (correct / total)
if __name__ == '__main__':
    # Acc()
    import logging
    import argparse

    parser = argparse.ArgumentParser("1CNN training")

    parser.add_argument('--log', type=str,default="/logs/",
                        help="Where things are logged and models are loaded from.")
    parser.add_argument('--sub',type=str ,default="sp300/",
                        help="sub log dir, to distinguish different model structure or data source")
    parser.add_argument('--train', type=str ,default="../data/train/",
                        help="training data set")
    parser.add_argument('--test', type=str ,default="../data/train/",
                        help="test data set")
    args = parser.parse_args()

    main_log=args.log
    sublog=args.sub

    trainpath=args.train
    testpath=args.test

    for i in range(0,1):

        main(main_log,((i)),trainpath,testpath)
