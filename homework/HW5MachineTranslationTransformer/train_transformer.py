import os 
import torch 
import time 
import math 



def train(device, model, train_loader, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    
    for src, _, trg in train_loader:
        src, trg = src.to(device), trg.to(device)
        
        optimizer.optimizer.zero_grad()
        output = model(src, trg)
                
        # output = [batch size, trg sent len - 1, output dim]
        # trg = [batch size, trg sent len]
        output = output[:,1:].contiguous().view(-1, output.shape[-1])
        trg = trg[:,1:].contiguous().view(-1)
        # output = [batch size * trg sent len - 1, output dim]
        # trg = [batch size * trg sent len - 1]
            
        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
        
    return epoch_loss / len(list(train_loader))



def evaluate(device, model, test_loader, criterion):
    model.eval()
    epoch_loss = 0
    
    with torch.no_grad():
        for src, _, trg in test_loader:
            src, trg = src.to(device), trg.to(device)

            output = model(src, trg)
            # output = [batch size, trg sent len - 1, output dim]
            # trg = [batch size, trg sent len]
            output = output[:,1:].contiguous().view(-1, output.shape[-1])
            trg = trg[:,1:].contiguous().view(-1)
            # output = [batch size * trg sent len - 1, output dim]
            # trg = [batch size * trg sent len - 1]
            
            loss = criterion(output, trg)
            epoch_loss += loss.item()
        
    return epoch_loss / len(list(test_loader))


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def train_model(device, model, train_loader, valid_loader, test_loader,
                optimizer, criterion, CLIP=1, N_EPOCHS=5, filename='transformer.pt'):


    SAVE_DIR = 'models'
    MODEL_SAVE_PATH = os.path.join(SAVE_DIR, filename)
    if not os.path.isdir(f'{SAVE_DIR}'):
            os.makedirs(f'{SAVE_DIR}')
    best_valid_loss = float('inf')

    if not os.path.isdir(f'{SAVE_DIR}'):
        os.makedirs(f'{SAVE_DIR}')

    for epoch in range(N_EPOCHS):
        start_time = time.perf_counter()
        
        train_loss = train(device, model, train_loader, optimizer, criterion, CLIP)
        valid_loss = evaluate(device, model, valid_loader, criterion)
        
        end_time = time.perf_counter()
        
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
        
        print(f'| Epoch: {epoch+1:03} | Time: {epoch_mins}m {epoch_secs}s| Train Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f} | Val. Loss: {valid_loss:.3f} | Val. PPL: {math.exp(valid_loss):7.3f} |')
    
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    test_loss = evaluate(device, model, test_loader, criterion)
    print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')