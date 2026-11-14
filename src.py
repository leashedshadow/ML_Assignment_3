import numpy as np
from torchvision import datasets

ROLL = "M26ECM510"
SEED = int("".join(c for c in ROLL if c.isdigit()))   # -> 26510
assert SEED == 26510

def make_split():
    ds = datasets.CIFAR10("data", train=True, download=True)
    y = np.array(ds.targets)
    rng = np.random.default_rng(SEED)
    val = np.concatenate([rng.choice(np.where(y == c)[0], 500, replace=False)
                          for c in range(10)])
    train = np.setdiff1d(np.arange(50000), val)

    # normalisation stats from the 45k TRAIN images only
    X = ds.data[train].astype(np.float32) / 255.0          # (45000,32,32,3)
    mean, std = X.mean(axis=(0, 1, 2)), X.std(axis=(0, 1, 2))

    np.savez("split.npz", train=train, val=val, seed=SEED, mean=mean, std=std)
    print(len(train), len(val), np.bincount(y[val]), mean, std)

if __name__ == "__main__":
    make_split()