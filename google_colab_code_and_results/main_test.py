"""
Double pendulum motion simulation

"""
from __future__ import print_function

import os
import sys
sys.path.append("./model")
sys.path.append("./dynamics")

from visualize import *
from ode_solver import *
from network import *
from dataloader import *
from lagrangian import *
import time

import numpy as np
import torch
import torch.utils.data

if __name__ == "__main__":
    print("--- Starting Main Training Loop! ---")
    # determine device
    print("--- Checking for CUDA Device... ---")
    use_cuda = torch.cuda.is_available()

    device = torch.device("cuda:0" if use_cuda else "cpu")

    # import data
    print("--- Loading validation data... ---")
    train_data = np.load('val_dataset.npz')
    train_inputs = train_data["input"]
    train_labels = train_data["labels"]

    # organize data
    input_size = train_inputs.shape[1]
    output_size = 1  # for all lagrangian systems, output should be just a scalar energy value

    # load into PyTorch Dataset and Dataloader
    train_dataset = DynamicsDataset(train_inputs, train_labels)


    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                     batch_size=1,
                                                     shuffle=True,
                                                     collate_fn=DynamicsDataset.collate_fn,
                                                     pin_memory=True,
                                                     num_workers=1)

    # build model
    print("--- Constructing Model... ---")
    D_in = input_size  # state size
    # hidden_list = [D_in, 256, 256, 256, 256, 256]
    hidden_list = [D_in, 32, 64, 128, 256, 512, 256, 128, 64, 32]
    D_out = output_size
    lnn_model = LagrangianNeuralNetwork(D_in, hidden_list, D_out)


    # set up training parameters
    learning_rate = 1e-4
    weight_decay = 1e-5
    momentum = 0.9
    num_epochs = 10
    optimizer = torch.optim.Adam(lnn_model.parameters(),
                                 lr=learning_rate,
                                 weight_decay=weight_decay)

    if os.path.isfile("model_weights.pth"):
        print("Re-loading existing weights!")
        checkpoint = torch.load("model_weights.pth")
        lnn_model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    lnn_model.to(device)
    start = time.time()
    Acc = []
    loss_combined = []
    print("--- Beginning Inference! ---")
    timing = []

    # Run 10 iterations
    for epoch in range(num_epochs):
        print("Epoch Start Time: {}".format(time.time()))
        # lnn_model.train()
        average_training_loss = 0

        print("Epoch #", epoch)
        Accuracy = []

        # print(len(train_dataloader))
        for batch_idx, (x, y) in enumerate(train_dataloader):
            x, y = x.to(device), y.to(device)
            x = torch.squeeze(x)
            with torch.cuda.amp.autocast():
                y_pred = solve_euler_lagrange(lnn_model.forward, x.float())
                acc = (y_pred.unsqueeze(0) == y.float()).sum().item()/y.shape[0]
                Accuracy.append(acc)

        Acc.append(np.mean(Accuracy)/4 * 100)
        end = time.time()
        timing.append((end - start))

    print("Total time taken for prediction {}".format(np.sum(timing)))
    print("Average time {}".format(np.mean(timing)))
    print("Accuracy {}".format(np.mean(Acc)))
    print('end')
