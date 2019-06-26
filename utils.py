import torch

def img2tensor(img):
    img = (img - 127.5) / 128
    return torch.from_numpy(img).float()


