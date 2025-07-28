import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split


class HAR_Dataset(Dataset):
    def __init__(self, data, referent, labels):
        # Normalize data
        data = (data - referent[0]) / (referent[1] - referent[0])
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        x = x.unsqueeze(1)
        return x, self.labels[idx]

def load_data(skip_list):

    doc = np.load('./mmWaveHAR4/infocom24_dataset.npz')
    data, labels = doc['data'], doc['label']

    mapping = {'push': 0, 'pull': 1, 'clockwise': 2, 'anticlockwise': 3}


    filtered_data = []
    filtered_labels = []

    i = 0
    for d, l in zip(data, labels):
        #if i >= 1080:
        #    break
        if l in mapping and l not in skip_list:
            filtered_data.append(d)
            filtered_labels.append(mapping[l])
        i += 1

    filtered_data = np.array(filtered_data)
    filtered_labels = np.array(filtered_labels)




    x_temp, x_test, y_temp, y_test = train_test_split(
        filtered_data, filtered_labels, test_size=0.15, stratify=filtered_labels, random_state=42)


    x_train, x_val, y_train, y_val = train_test_split(
        x_temp, y_temp, test_size=0.176, stratify=y_temp, random_state=42)

    train_referent = (np.min(x_train), np.max(x_train))
    train_dataset = HAR_Dataset(x_train, train_referent, y_train)
    val_referent = (np.min(x_val), np.max(x_val))
    val_dataset = HAR_Dataset(x_val, val_referent, y_val)
    test_referent = (np.min(x_test), np.max(x_test))
    test_dataset = HAR_Dataset(x_test, test_referent, y_test)


    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)


    print(f"Dataset split:")
    print(f"  Train: {len(train_dataset)} samples ({len(train_dataset)/(len(train_dataset)+len(val_dataset)+len(test_dataset))*100:.1f}%)")
    print(f"  Validation: {len(val_dataset)} samples ({len(val_dataset)/(len(train_dataset)+len(val_dataset)+len(test_dataset))*100:.1f}%)")
    print(f"  Test: {len(test_dataset)} samples ({len(test_dataset)/(len(train_dataset)+len(val_dataset)+len(test_dataset))*100:.1f}%)")

    return train_loader, val_loader, test_loader, x_train, x_test, y_train,  y_test


class HAR_query(Dataset):
    def __init__(self, data, referent):
        # Normalize data
        # WITH RESPECT TO TRAINING SET
        data = (data - referent[0]) / (referent[1] - referent[0])
        self.data = torch.tensor(data, dtype=torch.float32)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = self.data[idx]
        x = x.unsqueeze(1)
        return x
    
def load_query(query, referent):
    _query = HAR_query(query, referent)
    data_loader = DataLoader(_query)
    return data_loader
