
import torch 
import os 
import time 
import math 



def train(device, model, train_loader, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    
    for src, src_len, trg in train_loader:
        src, trg = src.to(device), trg.to(device)

        optimizer.zero_grad()
        # src: (src_sent_len, batch_size), trg: (trg_sent_len, batch_size)
        output = model(src, src_len, trg)
        # output: (trg_sent_len, batch_size, output_dim)
        # remove first <eos> token from output and target tensor, so loss won't be calculated on them 
        output = output[1:].view(-1, output.shape[-1]) # output = [(trg sent len - 1) * batch size, output dim]
        trg = trg[1:].view(-1)  # trg = [(trg sent len - 1) * batch size]
        
        loss = criterion(output, trg)
        loss.backward()
        # gradient clipping 
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
        
    return epoch_loss / len(list(train_loader))


def evaluate(device, model, test_loader, criterion):
    model.eval()
    epoch_loss = 0
    
    with torch.no_grad():
        for src, src_len, trg in test_loader:
            src, trg = src.to(device), trg.to(device)
            output = model(src, src_len, trg, 0) # turn off teacher forcing
            output = output[1:].view(-1, output.shape[-1])
            trg = trg[1:].view(-1)
            loss = criterion(output, trg)
            epoch_loss += loss.item()
        
    return epoch_loss / len(list(test_loader))


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def train_model(device, model, train_loader, valid_loader, test_loader, 
                optimizer, criterion, N_EPOCHS = 50, CLIP = 1, filename='seq2seq.pt'):

    
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
        
        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    test_loss = evaluate(model, test_loader, criterion)
    print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')