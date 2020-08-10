import numpy as np
from torch.utils.data import Dataset, DataLoader

class gaussianGridDataset(Dataset):
    def __init__(self, n, n_data, sig):
        self.grid = np.linspace(0, 8, n)
        self.data = list()
        for i in range(n):
            mean_x = self.grid[i]
            for j in range(n):
                mean_y = self.grid[j]

                temp = np.repeat(np.array([[i, j]]), repeats=n_data, axis=0)
                self.data.append(temp)

        self.out_dim = 2
        self.n_data = len(self.data)
        self.name = "gaussian_grid"

    def __getitem__(self, index):
        return self.data

    def __len__(self):
        return self.n_data

class gaussianGrid1dDataset(Dataset):
    def __init__(self, n, n_data, sig):
        self.grid = np.linspace(0, 8, n)
        self.data = list()
        for i in range(n):
            mean_x = self.grid[i]

            temp = np.repeat(np.array([i]), repeats=n_data, axis=0)
            self.data.append(temp)

        self.n_data = len(self.data)
        self.name = "gaussian_grid"

    def __getitem__(self, index):
        return self.data

    def __len__(self):
        return self.n_data
if __name__ == '__main__':

    # ds = DataLoader(gaussianGridDataset(5, 50, 0.1), batch_size=128, shuffle=True)
    # data = next(iter(ds))
    # print(data[0].shape)
    # print(data[1])


    ds = DataLoader(gaussianGrid1dDataset(5, 50, 0.1), batch_size=128, shuffle=True)
    data = next(iter(ds))
    print(data[0].shape)
    print(data[1])
    import pdb;pdb.set_trace()
