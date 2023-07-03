import numpy as np

class DataSet:
    def __init__(self, file_name):
        with np.load(file_name) as f:
            self.data_ph = f["data_ph"]
            self.data_in = f["data_in"]
            self.data_out = f["data_out"]
        self.num_data = len(self.data_ph)
        assert(len(self.data_in) == self.num_data and len(self.data_out) == self.num_data)
    
    def __len__(self):
        return self.num_data
    
    def __getitem__(self, i):
        return self.data_ph[i], self.data_in[i], self.data_out[i]
    

class DataLoader:
    def __init__(self, dataset:DataSet, batch_size):
        self.dataset = dataset
        self.index = 0
        self.batch_size = batch_size
        self.rand_idx = None
    
    def __iter__(self):
        self.index = 0
        self.rand_idx = np.random.permutation(len(self.dataset))
        return self

    def __next__(self):
        if self.index >= len(self.dataset):
            raise StopIteration
        value = self.dataset[self.rand_idx[self.index:self.index + self.batch_size]]
        # value = self.dataset[self.rand_idx[345:345 + self.batch_size]]
        self.index += self.batch_size
        return value
    
    def __len__(self):
        return len(self.dataset) // self.batch_size


if __name__ == "__main__":
    dataset = DataSet("processed_data.npz")
    dataloader = DataLoader(dataset, batch_size=32)

    for data in dataloader:
        print(data)
        break
