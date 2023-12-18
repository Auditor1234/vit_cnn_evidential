import torch
import clip
from tqdm import tqdm
from utils import template, prompts, save_results



# train signal and text jointly
def train_one_epoch_signal_text(model, epoch, epochs, device, train_loader, loss_func, optimizer, scheduler, classification, model_dim=2):
    model.train() 
    total_loss = 0.0
    loop = tqdm(train_loader, desc='Train', ncols=150)
    for _, (window_data, window_labels) in enumerate(loop): # shape(B,400,8)
        if model_dim == 1:
            window_data = window_data.transpose(1, 2).unsqueeze(-1) # shape(B,8,400,1)
        else:
            window_data = window_data.unsqueeze(1) # shape(B,1,400,8)
        window_data = window_data.to(device).type(torch.float32)

        text = clip.tokenize([template + prompts[mov_idx + 12] for mov_idx in window_labels]).to(device)
        
        if classification:
            evidences = model(window_data, text)
            labels = window_labels.to(device).type(torch.long) - 1
            evidence_a, loss = loss_func(evidences, labels, classes=10)
        else:
            logits_per_image, logits_per_text = model(window_data, text)
            labels = torch.LongTensor(range(len(window_labels))).to(device)
            loss_I = loss_func(logits_per_image, labels)
            loss_T = loss_func(logits_per_text, labels)
            loss = (loss_I + loss_T) / 2
        total_loss += loss

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loop.set_description(f'Epoch [{epoch+1}/{epochs}]')
        loop.set_postfix(loss = '%.6f' % loss.item())
    scheduler.step()
    print("[%d/%d] epoch's total loss = %f" % (epoch + 1, epochs, total_loss))
    save_results('res/results.csv', '%d, %12.6f\n' % (epoch + 1, total_loss))


def validate_signal_text(model, device, val_loader, loss_func, classification, model_dim=2):
    model.eval()
    total_loss, correct_nums, total_nums = 0.0, 0, 0
    
    print("Validating...")
    loop = tqdm(val_loader, desc='Validation', ncols=100)
    for i, (window_data, window_labels) in enumerate(loop): # shape(B,400,8)
        if model_dim == 1:
            window_data = window_data.transpose(1, 2).unsqueeze(-1) # shape(B,8,400,1)
        else:
            window_data = window_data.unsqueeze(1) # shape(B,1,400,8)
        window_data = window_data.to(device).type(torch.float32)

        text = clip.tokenize([template + prompts[mov_idx + 12] for mov_idx in window_labels]).to(device)
                
        if classification:
            evidences = model(window_data, text)
            labels = window_labels.to(device).type(torch.long) - 1
            evidence_a, loss = loss_func(evidences, labels, classes=10)
        else:
            logits_per_image, logits_per_text = model(window_data, text)
            labels = torch.LongTensor(range(len(window_labels))).to(device)
            loss_I = loss_func(logits_per_image, labels)
            loss_T = loss_func(logits_per_text, labels)
            loss = (loss_I + loss_T) / 2
        total_loss += loss

        predict_idx = evidence_a.argmax(dim=-1)
        correct_nums += torch.sum(predict_idx == labels)
        total_nums += len(predict_idx)

        loop.set_description(f'Validating [{i + 1}/{len(val_loader)}]')
        loop.set_postfix(loss = '%.6f' % loss.item())

    precision = '%.4f' % (100 * correct_nums / total_nums) + '%'
    print("Total loss: %.6f" % (total_loss))
    print("Correct/Total: {}/{}".format(correct_nums, total_nums))
    print("Precision:", precision)
    return correct_nums.item() / total_nums


def evaluate_signal_text(model, device, eval_loader, loss_func, classification, model_dim=2):
    model.eval() # 精度在64%
    total_loss, correct_nums, total_nums = 0.0, 0, 0
    
    print("Evaluating...")
    loop = tqdm(eval_loader, desc='Evaluation')
    for i, (window_data, window_labels) in enumerate(loop): # shape(B,400,8)
        if model_dim == 1:
            window_data = window_data.transpose(1, 2).unsqueeze(-1) # shape(B,8,400,1)
        else:
            window_data = window_data.unsqueeze(1) # shape(B,1,400,8)
        window_data = window_data.to(device).type(torch.float32)

        text = clip.tokenize([template + prompts[mov_idx + 12] for mov_idx in window_labels]).to(device)
                
        if classification:
            evidences = model(window_data, text)
            labels = window_labels.to(device).type(torch.long) - 1
            evidence_a, loss = loss_func(evidences, labels, classes=10)
        else:
            logits_per_image, logits_per_text = model(window_data, text)
            labels = torch.LongTensor(range(len(window_labels))).to(device)
            loss_I = loss_func(logits_per_image, labels)
            loss_T = loss_func(logits_per_text, labels)
            loss = (loss_I + loss_T) / 2
        total_loss += loss

        predict_idx = evidence_a.argmax(dim=-1)
        correct_nums += torch.sum(predict_idx == labels)
        total_nums += len(predict_idx)

        loop.set_description(f'Evaluating [{i + 1}/{len(eval_loader)}]')
        loop.set_postfix(loss = '%.6f' % loss.item())
    
    precision = '%.4f' % (100 * correct_nums / total_nums) + '%'
    print("Total loss: %.6f" % (total_loss))
    print("Correct/Total: {}/{}".format(correct_nums, total_nums))
    print("Precision:", precision)
    return precision
