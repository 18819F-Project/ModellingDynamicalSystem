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
    print("--- Loading training data... ---")
    train_data = np.load('train_dataset.npz')
    train_inputs = train_data["input"]
    train_labels = train_data["labels"]

    # organize data
    input_size = train_inputs.shape[1]
    output_size = 1  # for all lagrangian systems, output should be just a scalar energy value

    # load into PyTorch Dataset and Dataloader
    train_dataset = DynamicsDataset(train_inputs, train_labels)


    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                     batch_size=8,
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
    num_epochs = 20
    optimizer = torch.optim.Adam(lnn_model.parameters(),
                                 lr=learning_rate,
                                 weight_decay=weight_decay)

    if os.path.isfile("model_weights.pth"):
        print("Re-loading existing weights!")
        checkpoint = torch.load("model_weights.pth")
        lnn_model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # ensure model is in train mode so gradients are properly calculated
    lnn_model.train()

    # load device to either GPU or CPU depending on hardware
    lnn_model.to(device)

    # set up loss function
    loss_fcn = torch.nn.MSELoss()

    # set up GradScaler to improve run speed
    scaler = torch.cuda.amp.GradScaler()
    start = time.time()
    print("--- Beginning Training! ---")
    for epoch in range(num_epochs):
        print("Epoch Start Time: {}".format(time.time()))
        lnn_model.train()
        average_training_loss = 0

        print("Epoch #", epoch)
        Accuracy = []
        print(len(train_dataloader))
        for batch_idx, (x, y) in enumerate(train_dataloader):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            for p in lnn_model.parameters(): p.grad = None

            # output from model is the energy calculated from the parameterized lagrangian
            x = torch.squeeze(x)
            with torch.cuda.amp.autocast():
                y_pred = solve_euler_lagrange(lnn_model.forward, x.float())
                loss = loss_fcn(y_pred.unsqueeze(0), y.float())
                # print(y_pred.unsqueeze(0))
                # print(y.float())
                # print(y_pred.size())

            # perform backwards pass
            scaler.scale(loss).backward()

            # run optimization step based on backwards pass
            scaler.step(optimizer)

            # update the scale for next iteration
            scaler.update()

            if batch_idx % 100 == 0:
                print("Iter Num: ", batch_idx)
                print("\t", loss)
            if batch_idx % 5000 == 0:
                print("--- Saving weights ---")
                # save weights after each epoch
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': lnn_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, "model_weights.pth")
    end = time.time()

    print("Total time taken for training {}".format(end - start))
    print('end')
