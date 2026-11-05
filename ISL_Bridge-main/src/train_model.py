import csv 
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import time
from model import ISLModel



def main():
    #X
    dataset = []
    #y (needed to be converted to integer later for network training)
    labels = []
    with open("../data/landmarks_own.csv", "r", newline="") as file:
        reader = csv.reader(file)
        for row in reader:
            if row[-1] != 'sign':
                #126 coordinates and 1 sign, total 127 elements in each row
                dataset.append([float(x) for x in row[0:126]])
                labels.append(row[126])
    
    #dictionary with keys as unique labels and integers as values
    labels_integers = {label : i for i, label in enumerate(sorted(set(labels)))}

    torch.save(labels_integers, "../models/labels_integers.pth")

    #y (integer list)
    labels_ml = [int(labels_integers[x]) for x in labels]
        
    #dataset as numpy array , size (dataset, 126)
    X = np.array(dataset, dtype = np.float32)

    #labels(as integers) as numpy array , size  (dataset,)
    y = np.array(labels_ml, dtype = np.int64)
    cw = len(y)/ (33 * np.bincount(y))
    class_weights = torch.tensor(cw,dtype = torch.float32)

    #data augmentation
    left = X[:,0:63]
    right = X[:,63:126]
    X_aug = np.hstack((right,left))

    mask = X_aug[:, 0::3] != 0
    X_aug[:, 0::3][mask] = 1 - X_aug[:, 0::3][mask]
    y_aug = y.copy()

    X_final = np.vstack((X,X_aug))
    y_final = np.concatenate((y,y_aug))

    #split dataset for training and testing (80/20)
    X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size = 0.2, random_state = 42)
    X_train = torch.tensor(X_train, dtype = torch.float32)
    X_test = torch.tensor(X_test, dtype = torch.float32)
    y_train = torch.tensor(y_train, dtype = torch.long)
    y_test =  torch.tensor(y_test, dtype = torch.long)

    #Datasets
    dataset_train = TensorDataset(X_train,y_train)
    dataset_test = TensorDataset(X_test,y_test)

    #provides data in batches
    train_loader = DataLoader(dataset_train, batch_size = 32, shuffle = True)
    test_loader = DataLoader(dataset_test, batch_size = 32, shuffle = False)

    #initializing model, loss fxn and optimizer
    model = ISLModel()
    loss_fxn = nn.CrossEntropyLoss(class_weights)
    optimizer = optim.Adam(model.parameters(), lr = 0.001)

    best_accuracy = 0
    start = time.time()
    model.train()
    #training loop of 100 epoch
    for _ in range(400):
        total_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            predictions = model(X_batch) #forward pass  
            loss = loss_fxn(predictions, y_batch)
            total_loss += loss.item()
            loss.backward()
            optimizer.step() 

        #average loss per 50 epoch
        avg_loss = total_loss / len(train_loader)
        
        if _ % 25 == 0 :
            #testing during training to keep a check on overfitting
            model.eval()
            with torch.no_grad():
                correct = 0
                for X_batch , y_batch in test_loader:
                    output = model(X_batch) # size - (batch,33) , this calls forward fxn of the class
                    predictions = torch.argmax(output, dim = 1) # size - (batch,)
                    correct += torch.sum(predictions == y_batch).item()

            accuracy = correct / len(y_test)
            
            if accuracy > best_accuracy:
                torch.save(model.state_dict(), "../models/isl_model.pth")
                best_accuracy = accuracy

            print(f"{_} epochs, Loss : {avg_loss}, Test Accuracy : {accuracy * 100 :.2f} %" )
            model.train()

    print(f"Training time: {(time.time() - start)/ 60 :.2f} minutes") 
    print(f"Best Test Accuracy : {best_accuracy * 100 :.2f} %")


if __name__=="__main__":
    main()