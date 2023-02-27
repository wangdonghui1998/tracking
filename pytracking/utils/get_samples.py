import torch

def divide_train_test(dataset):
    train, test = dataset[0:], dataset[0:]

    return train, test

def get_samples(dataset, in_seq_len, pred_len,device):
    step = 1
    traindata = []
    targetdata = []
    for i in range(0, dataset.shape[0] - in_seq_len, step):
        train_data = dataset[i:i + in_seq_len]
        target_data = dataset[i + in_seq_len:i + in_seq_len + pred_len]
        traindata.append(train_data)
        targetdata.append(target_data)

    traindata = torch.tensor([item.cpu().detach().numpy() for item in traindata],dtype=torch.float).to(device)
    targetdata = torch.tensor([item.cpu().detach().numpy() for item in targetdata],dtype=torch.float).to(device)
    return traindata,targetdata