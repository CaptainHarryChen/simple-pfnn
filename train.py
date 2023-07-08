import numpy as np
from model import SimplePFNN
from dataloader import DataLoader, DataSet

total_epoch = 50
learning_rate = 0.01
batch_size = 64


def SGD_update(model:SimplePFNN):
    model.linear1.weight -= model.linear1.grad_weight * learning_rate
    model.linear1.bais -= model.linear1.grad_bais * learning_rate
    model.linear2.weight -= model.linear2.grad_weight * learning_rate
    model.linear2.bais -= model.linear2.grad_bais * learning_rate
    model.linear3.weight -= model.linear3.grad_weight * learning_rate
    model.linear3.bais -= model.linear3.grad_bais * learning_rate


def train(start_epoch=None):
    dataset = DataSet("processed_data.npz")
    dataloader = DataLoader(dataset, batch_size=batch_size)

    input_dim = 120
    output_dim = 160
    model = SimplePFNN(input_dim, output_dim)
    if start_epoch != None:
        model.load(f"./checkpoints/model_{start_epoch}.npz")
        start_epoch += 1
    else:
        start_epoch = 0

    for epoch in range(start_epoch, total_epoch):
        print(f"Epoch {epoch}/{total_epoch} : ")
        n_iter = len(dataloader)
        for it, data in enumerate(dataloader):
            model.zero_grad()
            phase, x, y = data
            pred = model(x, phase)
            loss = np.sum(0.5 * (pred - y) ** 2) / batch_size
            print(f"Iter: {it}/{n_iter}    Loss: {loss}")
            model.backward((pred - y) / batch_size)
            SGD_update(model)
        model.save(f"./checkpoints/model_{epoch}.npz")


if __name__ == "__main__":
    train(start_epoch=None)
