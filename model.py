import torch
from torch import nn



class CTClassifier(nn.Module):

    def __init__(self, n_downsamples=5, in_channels=1, filters=24, activation=nn.LeakyReLU(0.1), maxfeatures=256):
        super().__init__()

        self.maxfeatures = maxfeatures
        self.n_downsamples = n_downsamples
        self.filters = filters
        self.intro = nn.Sequential(nn.ReplicationPad3d(3),
                                   nn.Conv3d(in_channels, filters, kernel_size=7, padding=0),
                                   nn.BatchNorm3d(filters),
                                   activation)
        model = []
        for i in range(1, self.n_downsamples):
            inp = self.filters * i
            inp = min(inp, self.maxfeatures)
            out = self.filters * (i + 1)
            out = min(out, self.maxfeatures)
            model += [
                nn.Conv3d(inp, out, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm3d(out),
                activation]
            print(f'{inp} -> {out}')
        self.model = nn.Sequential(*model)
        self.dense = nn.Sequential(nn.Linear(120*4*4*3, 512),
                                   activation,
                                   nn.Dropout(),
                                   nn.Linear(512, 2))
        print()

    def forward(self, x):
        x = self.intro(x)
        x = self.model(x)
        x = x.view(x.size(0), -1)
        emb = self.dense(x)
        return emb


if __name__ == "__main__":
    estimator = CTClassifier()
    torch.save(estimator.state_dict(), 'test.pth')
    latent = estimator(torch.ones([10, 1, 64, 64, 48]))
    print('nonno')