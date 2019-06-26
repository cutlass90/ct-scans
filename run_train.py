import os
import torch
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from tensorboardX import SummaryWriter
from helpers import save_weights, load_weights

from data  import ScanDataset
from model import CTClassifier
from torch import nn
import torch.nn.functional as F

np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

###############################################################################
dataset_path = 'dataset/synthetic_v1'
lr = 0.0002
batch_size = 4
load_weights_path = None#'/home/nazar/faces/workshop_controlled_image_sunthesys/src/checkpoints/AgeGenderRaceEstimator/128/weights/latest.pth'
print_freq = 2
save_freq = 20
epochs = 50
checkpoint_dir = 'checkpoints/v1/'
device_ids = [0]  # or None if use cpu only
class_weight = torch.tensor([1, 35/65])
###############################################################################
if device_ids is not None:
    device = torch.device("cuda:" + ','.join(str(i) for i in device_ids))
else:
    device = torch.device("cpu")
writer = SummaryWriter(os.path.join(checkpoint_dir, 'tf_log'))


def objective(outs, targets):
    ce_loss = nn.CrossEntropyLoss(weight=class_weight.to(device))(outs, targets)
    return ce_loss


def get_accuracy(logits, targets):
    _, predicted = torch.max(logits.data, 1)
    total = targets.size(0)
    correct = predicted.eq(targets.data).cpu().sum()
    acc = (100.0 * correct / total)
    return acc


def train(epoch, net, gen, optimizer, criterion):
    net.train()
    for step, data in tqdm(enumerate(gen)):
        inputs = data['image'].to(device)
        targets = data['target'].to(device)

        outputs = net(inputs)
        optimizer.zero_grad()
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        if (step + 1) % print_freq == 0:
            acc = get_accuracy(outputs, targets)
            print(f'Epoch={epoch}, step={step}, loss={loss.item()}, acc={acc}')
            writer.add_scalar('loss', loss.item(), epoch * len(gen) + step)
            writer.add_scalar('acc', acc, epoch * len(gen) + step)

        if (step + 1) % save_freq == 0:
            save_weights(net, os.path.join(checkpoint_dir, 'weights/', 'latest.pth'))


def eval_model(net, dataset):
    net.eval()
    print('\n\nStart evaluation', end='')
    dataset.set_mode('val')
    gen = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    loss, acc = [], []
    for step, data in tqdm(enumerate(gen)):
        inputs = data['image'].to(device)
        targets = data['target'].to(device)
        with torch.no_grad():
            outputs = net(inputs)
        loss += [objective(outputs, targets).item()]
        acc += [get_accuracy(outputs, targets)]
    print(f'  loss={np.array(loss).mean()}')
    print(f'  acc={np.array(acc).mean()}')
    net.train()
    return np.array(loss).mean(), np.array(acc).mean()


def main():
    os.makedirs(os.path.join(checkpoint_dir, 'weights'), exist_ok=True)

    dataset = ScanDataset(test_size=0.1, resize=(1/8, 1/8, 48))
    net = CTClassifier().to(device)

    if load_weights_path:
        load_weights(net, load_weights_path, True)
    if device_ids is not None:
        net = torch.nn.DataParallel(net, device_ids)
    for epoch in range(epochs):
        optimizer = optim.Adam(net.parameters(), lr=lr)
        dataset.set_mode('train')
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=3)
        train(epoch, net, dataloader, optimizer, objective)
        loss, acc = eval_model(net, dataset)
        writer.add_scalar('loss_val', loss, epoch * len(dataloader))
        writer.add_scalar('acc_val', acc, epoch * len(dataloader))
        save_weights(net, os.path.join(checkpoint_dir, 'weights/', f'epoch{epoch}_{loss}_{acc}.pth'))


if __name__ == "__main__":
    main()
