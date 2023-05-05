import torch 
import os 


def train(device, model, loader, optimizer, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.train()
    
    for text, labels in loader:
        text = text.to(device)
        labels = labels.to(device)
        # reset the optimiser
        optimizer.zero_grad()

        # TODO: make predictions
        outputs = model(text)
        # TODO: calculate loss and accuracy
        loss = criterion(outputs, labels) 
        acc = binary_accuracy(outputs, labels)
        # backprop
        loss.backward()
        # update params
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc

    return epoch_loss / len(loader), epoch_acc / len(loader)

def evaluate(device, model, loader, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.eval()
    with torch.no_grad():
        for text, labels in loader:
            text = text.to(device)
            labels = labels.to(device)
            # make predictions
            outputs = model(text)
            # calculate loss and accuracy
            loss = criterion(outputs, labels) 
            acc = binary_accuracy(outputs, labels)

            epoch_loss += loss.item()
            epoch_acc += acc
            
        
    return epoch_loss / len(loader), epoch_acc / len(loader)


def binary_accuracy(outputs, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    outputs = torch.sigmoid(outputs)
    accuracy = torch.mean((outputs > 0.5).eq(y).float()).item()
    return accuracy


def train_model(device, model, train_loader, valid_loader, test_loader, filename, N_EPOCHS = 5, learning_rate=0.001):
    SAVE_DIR = 'models'

    MODEL_SAVE_PATH = os.path.join(SAVE_DIR, filename)
    if not os.path.isdir(f'{SAVE_DIR}'):
        os.makedirs(f'{SAVE_DIR}')

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_valid_loss = float('inf')

    for epoch in range(N_EPOCHS):
        
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, device)
        valid_loss, valid_acc = evaluate(model, valid_loader, criterion, device)
            
        # we keep track of the best model, and save it
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
        
        print(f'Epoch: {epoch}')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')
    
    # Restore the best model and evaluate. The test accuracy should be around 55%.
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    test_loss, test_acc = evaluate(model, test_loader, criterion)

    print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')