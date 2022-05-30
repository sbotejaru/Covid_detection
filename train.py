from comet_ml import Experiment
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
from Dataloaders.Dataloader import BaselineDataloader
from Model.Model import CNN
from datetime import datetime
from sklearn.metrics import plot_confusion_matrix
from torchvision import models

if __name__ == "__main__":
    # Experiment
    exp_name = "exp"
    exp_path = os.path.join("./", "experiments", exp_name + datetime.today().strftime('%Y-%m-%d-%H_%M_%S'))
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)

    # Create an experiment with your api key
    experiment = Experiment(api_key="", project_name="", workspace="", )

    # For reproductibility
    random_seed = 1
    torch.manual_seed(random_seed)

    # Hyperparameters
    epochs = 20
    batch_size_train = 32
    learning_rate = 0.003

    # Run on GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device used: {device}")

    # Dataloaders
    train_dataset = BaselineDataloader("Data/train_processed_fix", split='split.json', phase='train')
    validation_dataset = BaselineDataloader("Data/train_processed_fix", split='split.json', phase='val')

    # Dataloaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=1, shuffle=False)

    # Instatiate the neural network
    model = CNN().to(device)
    model = model.cuda()

    # Afisarea numarului de parametri antrenabili ai modelului
    model_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Numarul total de parametri antrenabili ai modelului: {model_total_params}")
    experiment.log_parameter("nr_of_model_params", model_total_params)

    # Definirea loss-ului, functia NegativeLogLikeliHood
    criterion = torch.nn.NLLLoss()
    criterion.cuda()

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    ###################### Training #######################
    errors_train = []
    errors_validation = []

    # Training loop
    for epoch in range(epochs):
        # O lista unde vom stoca erorile temporare epocii curente
        temporal_loss_train = []

        # Functia .train() trebuie apelata explicit inainte de antrenare
        model.train()

        # Iteram prin toate sample-urile generate de dataloader
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Clean the gradients
            optimizer.zero_grad()

            # Forward propagation
            output = model(images)

            # Compute the error
            loss = criterion(output, labels)
            temporal_loss_train.append(loss.item())

            # Backpropagation (computing the gradients for each weight)
            loss.backward()

            # Update the weights
            optimizer.step()

        # Now, after each epoch, we have to see how the model is performing on the validation set #
        # Before evaluation we have to explicitly call .eval() method
        model.eval()
        temporal_loss_valid = []
        for images, labels in validation_loader:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            output = model(images)

            # Compute the error
            loss = criterion(output, labels)
            temporal_loss_valid.append(loss.item())

        # Compute metrics after each epoch (mean value of loss) #
        medium_epoch_loss_train = sum(temporal_loss_train) / len(temporal_loss_train)
        medium_epoch_loss_valid = sum(temporal_loss_valid) / len(temporal_loss_valid)

        errors_train.append(medium_epoch_loss_train)
        errors_validation.append(medium_epoch_loss_valid)

        print(f"Epoch {epoch}. Training loss: {medium_epoch_loss_train}. Validation loss: {medium_epoch_loss_valid}")

        # Log metrics
        experiment.log_metric("train_loss", medium_epoch_loss_train, step=epoch)
        experiment.log_metric("val_loss", medium_epoch_loss_valid, step=epoch)

        # Saving the model of the current epoch
        torch.save(model.state_dict(), os.path.join(exp_path, f"Epoch{epoch}_Error{round(medium_epoch_loss_valid, 3)}"))

    plt.title("Learning curves")
    plt.plot(errors_train, label='Training loss')
    plt.plot(errors_validation, label='Validation loss')
    plt.xlabel("Epoch")
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(exp_path, "losses.png"))
