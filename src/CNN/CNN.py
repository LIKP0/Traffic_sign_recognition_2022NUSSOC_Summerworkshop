from torch.utils.data import Dataset, DataLoader
import cv2
import os
import numpy as np
import csv
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as op

# return BGR image with size [25,25,3]
# done histogram equalization
class TrafficSignDataset(Dataset):
    def __init__(self, train):
        self.img_list = []
        self.label_list = []
        self.crop_list = []
        if train:
            csv_file = open('./Dataset_2_Train/Train.csv')
            reader = csv.reader(csv_file)
            for item in reader:
                if reader.line_num == 1:
                    continue
                self.label_list.append(int(item[6]))
                self.img_list.append(f'./Dataset_2_Train/{item[7]}')
                self.crop_list.append([int(item[2]), int(item[3]), int(item[4]), int(item[5])])
            # for i in range(43):
            #     path = f'./Dataset_2_Train/Train/{i}/'
            #     for file in os.listdir(path):
            #         self.img_list.append(path + file)
            #         self.label_list.append(i)
        else:
            csv_file = open('./Dataset_2_Test/Test.csv')
            reader = csv.reader(csv_file)
            for item in reader:
                if reader.line_num == 1:
                    continue
                self.label_list.append(int(item[6]))
                self.img_list.append(f'./Dataset_2_Test/{item[7]}')
                self.crop_list.append([int(item[2]), int(item[3]), int(item[4]), int(item[5])])


    def transform(self, path, idx):
        img = cv2.imread(path)
        cropped = img[self.crop_list[idx][1]:self.crop_list[idx][3], self.crop_list[idx][0]:self.crop_list[idx][2]]
        img = cv2.resize(cropped, (32,32))
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:,:,0])
        equalized = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
        RGB = cv2.split(equalized)
        np_res = np.stack(RGB, axis=0)
        return torch.FloatTensor(np_res)
        
    def __getitem__(self, index):
        return self.transform(self.img_list[index], index), self.label_list[index]
    
    def __len__(self):
        return len(self.img_list)
    

class CNN(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(CNN, self).__init__()
        self.layer_list = nn.ModuleList([])
        self.layer_list.append(nn.Conv2d(n_channels, 64, 3, 1, 1))
        self.layer_list.append(nn.BatchNorm2d(64))
        self.layer_list.append(nn.MaxPool2d(3, 2, 1))
        # 64*15*15
        self.layer_list.append(nn.Conv2d(64, 128, 3, 1, 1))
        self.layer_list.append(nn.BatchNorm2d(128))
        # 128*15*15
        self.layer_list.append(nn.MaxPool2d(3, 2, 1))
        # 128*8*8
        self.layer_list.append(nn.Conv2d(128, 256, 3, 1, 1))
        self.layer_list.append(nn.BatchNorm2d(256))
        # 256*8*8
        self.layer_list.append(nn.Conv2d(256, 256, 3, 1, 1))
        self.layer_list.append(nn.BatchNorm2d(256))
        # 256*8*8
        self.layer_list.append(nn.MaxPool2d(3, 2, 1))
        # 256*5*5
        self.layer_list.append(nn.Conv2d(256, 512, 3, 1, 1))
        self.layer_list.append(nn.BatchNorm2d(512))
        # 512*5*5
        self.layer_list.append(nn.Conv2d(512, 512, 3, 1, 1))
        self.layer_list.append(nn.BatchNorm2d(512))
        # 512*5*5
        self.layer_list.append(nn.MaxPool2d(3, 2, 1))
        # 512*3*3
        self.layer_list.append(nn.Conv2d(512, 512, 3, 1, 1))
        self.layer_list.append(nn.BatchNorm2d(512))
        # 512*3*3
        self.layer_list.append(nn.Conv2d(512, 512, 3, 1, 1))
        self.layer_list.append(nn.BatchNorm2d(512))
        # 512*3*3
        self.layer_list.append(nn.MaxPool2d(3, 2, 1))
        self.linear_layer = nn.Linear(512, n_classes)

        self.relu = nn.ReLU()
    
    def forward(self, x):
        for layer in self.layer_list:
            if isinstance(layer, nn.BatchNorm2d):
                # print("x:", x.shape)
                x = self.relu(layer(x))
            else:
                # print("x:", x.shape)
                x = layer(x)
        x = x.view(x.shape[0], -1)
        out = self.linear_layer(x)
        return out

        
def accuracy(predictions, targets):
    cnt = 0
    for idx in range(len(predictions)):
        if np.argmax(predictions[idx]) == targets[idx]:
            cnt += 1
    return cnt / len(predictions)

def evaluate(cnn, loader):
    with torch.no_grad():
        out = []
        labels = []
        for idx, data in enumerate(loader):
            inputs, label = data
            inputs = inputs.to(device)
            label = label.to(device)
            for l in label:
                labels.append(l)
            outputs = cnn(inputs)
            # print(outputs)
            # for o in outputs.tolist():
            #     print("pred:", np.argmax(o))
            # break
            for o in outputs.tolist():
                out.append(o)
        return accuracy(out, labels)

default_lr = 1e-4
max_epoch = 15
eval_step = 3
test_acc_list = []
train_acc_list = []
loss_list = []
epoch_list = []
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_set = TrafficSignDataset(train=True)
test_set = TrafficSignDataset(train=False)
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
test_loader = DataLoader(test_set, batch_size=32, shuffle=False)
cnn = CNN(3, 43)
cnn = cnn.to(device)
loss_func = nn.CrossEntropyLoss()
optimizer = op.Adam(params=cnn.parameters(), lr=default_lr)

for epoch in range(max_epoch):
    print(epoch, "/", max_epoch, "epoch finished")
    t = tqdm(train_loader)
    t.set_description("epoch: %s" % epoch)
    for data in t:
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = cnn(inputs)
        loss = loss_func(outputs, labels)
        loss.backward()
        t.set_postfix(loss=loss.item())
        optimizer.step()

    epoch_list.append(epoch + 1)
    loss_list.append(loss)
    train_acc = evaluate(cnn, train_loader)
    test_acc = evaluate(cnn, test_loader)
    print("Train Accuracy:", train_acc)
    print("Test Accuracy:", test_acc)
    train_acc_list.append(train_acc)
    test_acc_list.append(test_acc)
plt.plot(epoch_list, test_acc_list, 's-', color='r', label='test_accuracy')
plt.plot(epoch_list, loss_list, 's-', color='g', label='loss')
plt.plot(epoch_list, train_acc_list, 's-', color='b', label='train_accuracy')
plt.legend()
plt.savefig("Result.png")