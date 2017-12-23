from tqdm import tqdm
import torch
from torch.autograd import Variable

def get_set_accuracy(model, loader):
    correct, total = 0, 0
    for x, y in tqdm(iter(loader)):
        x, y = Variable(x), Variable(y)
        score = model(x)
        _, y_pred = torch.max(score, 1)
        correct += y_pred.eq(y).sum().data[0]
        total += x.size(0)
    return correct / total
