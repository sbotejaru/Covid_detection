import numpy as np
import torch
import os
import matplotlib.pyplot as plt
from Dataloaders.Dataloader import BaselineDataloader
from Model.Model import CNN
from datetime import datetime
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix
from timeit import default_timer as timer


if __name__ == "__main__":

    # Experiment to evaluate
    exp_name = "exp2022-05-05-17_51_12"
    exp_path = os.path.join("experiments", exp_name)

    # Save some results
    res_path = os.path.join(exp_path, "test_results")
    if not os.path.exists(res_path):
        os.mkdir(res_path)

    # For reproductibility
    random_seed = 1
    torch.backends.cudnn.enabled = False
    torch.manual_seed(random_seed)

    # Run on GPU if available
    device = torch.device("cpu")
    print(f"Device used: {device}")

    # Dataloaders
    test_dataset = BaselineDataloader("Data/train_processed_fix", split='split.json', phase='test')

    # Dataloaders
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Instatiate the neural network
    model = CNN().to(device)
    model.load_state_dict(torch.load(os.path.join(exp_path, "Epoch19_Error0.355")))
    model.eval() 

    ###################### Evaluating #######################
    y_true = []
    y_pred = []

    bad_items = []
    with torch.no_grad():
        for images, labels in test_loader:   
            images, labels = images.to(device), labels.to(device)
            
            start = timer()
            
            # Forward propagation
            output = model(images)
            end = timer()
            # print(end - start) 
            
            # Results
            pred_y = torch.max(output, 1)[1].data.squeeze()
            y_true.append(labels.item())
            y_pred.append(pred_y.item())
            
            # Identify hard cases
            if labels.item() != pred_y.item():
                bad_items.append((images.squeeze().numpy(), labels.item(), pred_y.item()))
        
    # F1 scores
    scores = f1_score(y_true, y_pred, average=None)
    print(f'f1 scores for every class: {scores}')
    
    # Precision
    scores = precision_score(y_true, y_pred, average=None)
    print(f'precision scores for every class: {scores}')

    # recall
    scores = recall_score(y_true, y_pred, average=None)
    print(f'recall scores for every class: {scores}')


    # Plot confusion matrix
    conf_matrix = confusion_matrix(y_true=y_true, y_pred=y_pred)
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
    
    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actuals', fontsize=18)
    plt.title('Confusion Matrix', fontsize=18)
    # plt.show()
    plt.savefig(os.path.join(res_path, "confusion_matrix.png"))
    plt.clf()

    # Bad predictions
    res_path_images = os.path.join(exp_path, "test_results", "bad_guys")
    if not os.path.exists(res_path_images):
        os.mkdir(res_path_images) 
    for i, (img, label, pred) in enumerate(bad_items): 
        plt.imshow(img, cmap='gray')
        plt.savefig(os.path.join(res_path_images, f"Image{i}_Actual{label}.Pred{pred}.png"))

        
