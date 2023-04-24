from dvsc.model import RsNet
from dvsc.util import get_train_val_iter
from dvsc.util import get_test_iter
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as opt
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import pandas as pd
EPOCH = 1
LR = 1e-3

train_iter, val_iter = get_train_val_iter()
x, label = iter(train_iter).next()
test_iter = get_test_iter()
print('x:', x.shape, 'label:', label.shape)
lossf = nn.CrossEntropyLoss()
#lossf = nn.CrossEntropyLoss().to
model = RsNet()
optimizer = opt.SGD(model.parameters(), lr = LR)
model.train()

best_acc = 0
for epoch in range(EPOCH):
    losssum = 0
    for idx, (x, label) in enumerate(train_iter):
        #x, label = x.to(DEVICE), label.to(DEVICE)
        pred = model(x)
        loss =lossf(pred, label)
        losssum += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print("epoch: ", epoch, "loss: ", losssum)
    model.eval()
    with torch.no_grad():
        total_correct = 0
        total_num = 0
        for x, label in val_iter:
            y_val = model(x)
            y_pred = y_val.argmax(dim=1)
            total_correct += torch.eq(y_pred, label).float().sum().item()
            total_num += x.size(0)
        if total_correct/total_num > best_acc:
            best_acc = total_correct / total_num
            torch.save(model, "./best_model.pkl")
        print("epoch:", epoch, "acc: " ,total_correct / total_num)

modelx = torch.load("./best_model.pkl")
modelx.eval()
x_arr = torch.LongTensor()
y_arr = torch.LongTensor()
for x, label in test_iter:
    pred = modelx(x)
    y_pred = pred.argmax(dim=1)
    y_arr = torch.cat((y_arr, y_pred), dim = 0)
    x_arr = torch.cat((x_arr, label), dim = 0)

submission_df = pd.DataFrame({'id': [], 'label': []})
submission_df['id'] = x_arr
submission_df['label'] = y_arr
submission_df.to_csv("submisson.csv".index("False"))
print(submission_df.head())




