import glob

import torch
import torch.utils.data as Data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Dataset
import os
import shutil
from PIL import Image
VAL_RATE = 0.8
PATH = '..\\data\\dog_vs_cat'
BATCH_SIZE = 16

#首先要根据文件名建立文件夹并将文件放入文件夹中
def clsf(file_path):
    print(file_path)
    cat_path = file_path+"\\cat\\"
    dog_path = file_path+"\\dog\\"
    if not os.path.exists(cat_path):
        os.mkdir(cat_path)
    if not os.path.exists(dog_path):
        os.mkdir(dog_path)
    print(cat_path)
    print(dog_path)
    path_list = os.listdir(file_path)
    for item in path_list:
        item_name = item.split('.')
        print(item_name)
        p = os.path.join(file_path, item)
        print(p)
        if 'cat' in item_name and 'jpg' in item_name:
            shutil.move(os.path.join(file_path,item), cat_path)
        elif 'dog' in item_name and 'jpg' in item_name:
            shutil.move(os.path.join(file_path, item), dog_path)
def val(file_path):
    tar_cat = file_path + "\\cat\\"
    tar_dog = file_path + "\\dog\\"
    path_cat = os.listdir(tar_cat)
    path_dog = os.listdir(tar_dog)
    print(path_cat)
    cat_p = PATH + "\\val"
    dog_p = PATH + "\\val"
    if not os.path.exists(cat_p):
        os.mkdir(cat_p)
    if not os.path.exists(dog_p):
        os.mkdir(dog_p)
    cat_p += "\\cat\\"
    dog_p += "\\dog\\"
    if not os.path.exists(cat_p):
        os.mkdir(cat_p)
    if not os.path.exists(dog_p):
        os.mkdir(dog_p)
    lang = len(path_cat)
    cnt = lang * (1 - VAL_RATE)
    for item in path_cat:
        if cnt > 0:
            shutil.move(os.path.join(tar_cat, item), cat_p)
        else : break
        cnt -= 1
    cnt = lang * (1 - VAL_RATE)
    for item in path_dog:
        if cnt > 0:
            shutil.move(os.path.join(tar_dog, item), dog_p)
        else :break
        cnt -= 1



def get_train_val_iter():
    train_path = PATH + "\\train"
    val_path = PATH + "\\val"
    transform = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229, 0.224, 0.225])
                                    ])

    train_dataset = datasets.ImageFolder(train_path, transform)
    print(train_dataset.class_to_idx)
    val_dataset = datasets.ImageFolder(val_path, transform)
    print(val_dataset.class_to_idx)

    train_iter = Data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_iter = Data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    return train_iter, val_iter


class Test_dataset(Dataset):
    def __init__(self, img_path, transform = None):
        self.img_path = img_path
        self.transform = transform

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, idx):
        img_path = self.img_path[idx]
        file_name = os.path.basename(img_path)
        iid, _ = file_name.split(".")
        iid = int(iid)
        img_data = Image.open(img_path).convert("RGB")
        img = self.transform(img_data)
        return img, iid

def get_test_iter():
    file_path = PATH + "\\test"
    test_img_path_list = glob.glob(os.path.join(file_path, '*.jpg'))
    length = len(test_img_path_list)
    list_file = []
    for i in range(length):
        tar = file_path + "\\" + str(i + 1) + ".jpg"
        list_file.append(tar)
    transform = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229, 0.224, 0.225])
                                    ])
    test_data = Test_dataset(list_file, transform)
    test_iter = Data.DataLoader(test_data, shuffle=False)
    return test_iter
if __name__ == '__main__':
    test_iter = get_test_iter()
    cnt = 5
    for x, label in test_iter:
        print(x)
        print(label)
        cnt -= 1
        if cnt < 0 : break
    pass

