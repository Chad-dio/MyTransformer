import torch
from torch import nn


class EBD(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.word_ebd = nn.Embedding(28, 24)
        self.pos_ebd = nn.Embedding(12, 24)
        self.pos_t = torch.arange(0, 12)

    def forward(self, X):
        return self.word_ebd(X) + self.pos_ebd(self.pos_t)


def transpose(QKV: torch.Tensor):
    QKV = QKV.reshape(QKV.shape[0], QKV.shape[1], 4, 6)
    QKV = QKV.transpose(-2, -3)
    return QKV


def attention(Q, K, V):
    A = Q @ K.transpose(-1, -2) / (Q.shape[-1] ** 0.5)
    A = torch.softmax(A, dim=-1)
    O = A @ V
    return O


class Attention_Block(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.Wq = nn.Linear(24, 24, bias=False)
        self.Wk = nn.Linear(24, 24, bias=False)
        self.Wv = nn.Linear(24, 24, bias=False)
        self.Wo = nn.Linear(24, 24, bias=False)

    def forward(self, X):
        Q, K, V = self.Wq(X), self.Wk(X), self.Wv(X)


if __name__ == "__main__":
    a = torch.ones((2, 12)).long()
    ebd = EBD()
    b = ebd(a)
    print(b.shape)
    pass
