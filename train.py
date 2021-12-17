import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from data_loader import get_loader

#from models.vgg_model import VqaModel
#from models.mutan_model import VqaModel
#from models.san_model import VqaModel
from models.resnet_model import VqaModel

# setting up the device (cpu or gpu)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# loading the data
data_loader = get_loader(
    input_dir='./datasets',
    input_vqa_train='train.npy',
    input_vqa_valid='valid.npy',
    max_qst_length=30,
    max_num_ans=10,
    batch_size=256,
    num_workers=8)

qst_vocab_size = data_loader['train'].dataset.qst_vocab.vocab_size
ans_vocab_size = data_loader['train'].dataset.ans_vocab.vocab_size

# defining the model
model = VqaModel(
    feature_size=1024,
    qst_vocab_size=qst_vocab_size,
    ans_vocab_size=ans_vocab_size,
    qst_embed_size=300,
    num_layers=2,
    hidden_size=512).to(device)

# defining the loss function
criterion = nn.CrossEntropyLoss()

# storing the trainable parameters of the vqa model in trainable_params
trainable_params = list(model.img_encoder.fc.parameters()) \
    + list(model.qst_encoder.parameters()) \
    + list(model.fc1.parameters()) \
    + list(model.fc2.parameters())

# defining the optimizer and lr scheduler
optimizer = optim.Adam(trainable_params, lr=1e-3)
scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# run the model for 25 epochs
for epoch in range(25):
    
    train_loss = 0.0
    val_loss = 0.0
    n_train_correct = 0
    n_train_samples = 0
    n_val_correct = 0
    n_val_samples = 0
    
    step_size = len(data_loader['train'].dataset) / 256
    # training loop
    for idx, sample in enumerate(data_loader['train']):
        inp_images = sample['image'].to(device)
        inp_questions = sample['question'].to(device)
        inp_labels = sample['answer_label'].to(device)
        optimizer.zero_grad()
        output = model(inp_images, inp_questions) 
        loss = criterion(output, inp_labels.long())
        loss.backward()
        optimizer.step()
        # print the mini-batch loss
        if idx % 100 == 0:
            print('Training: Epoch [{:02d}/{:02d}], Step [{:04d}/{:04d}], Loss: {:.4f}'.format(epoch+1, 25, idx, int(step_size), loss.item())) 
        scheduler.step()
        
        # caculating training statistics
        _, predicted = torch.max(output, 1)
        n_train_samples += inp_labels.size(0)
        n_train_correct += (predicted == inp_labels).sum().item()
        train_loss += loss
    
    # validation loop
    model.eval()
    with torch.no_grad():
        step_size = len(data_loader['valid'].dataset) / 256
        for idx, sample in enumerate(data_loader['valid']):
            inp_images = sample['image'].to(device)
            inp_questions = sample['question'].to(device)
            inp_labels = sample['answer_label'].to(device)
            output = model(inp_images, inp_questions) 
            loss = criterion(output, inp_labels.long())
            # print the mini-batch loss
            if idx % 100 == 0:
                print('Validation: Epoch [{:02d}/{:02d}], Step [{:04d}/{:04d}], Loss: {:.4f}'.format(epoch+1, 25, idx, int(step_size), loss.item()))

            # calculating validation statistics
            _, predicted = torch.max(output, 1)
            n_val_samples += inp_labels.size(0)
            n_val_correct += (predicted == inp_labels).sum().item()
            val_loss += loss
    model.train()
    
    # calculating training and validation accuracy
    train_acc = n_train_correct / n_train_samples
    val_acc = n_val_correct / n_val_samples

    print('Epoch Statistics: Epoch [{:02d}/{:02d}], Train_Loss: {:.4f}, Val_Loss: {:.4f}, Train_Acc: {:.4f}, Val_Acc: {:.4f}\n'.format(epoch+1, 25, train_loss, val_loss, train_acc, val_acc))

    # logging the training loss and accuracy
    with open(os.path.join('./logs', '{}-log-epoch-{:02}.txt').format('train', epoch+1), 'w') as f:
        f.write(str(epoch+1) + '\t'+ str(train_loss) + '\t' + str(train_acc))
    
    # logging the validation loss and accuracy
    with open(os.path.join('./logs', '{}-log-epoch-{:02}.txt').format('valid', epoch+1), 'w') as f:
        f.write(str(epoch+1) + '\t'+ str(val_loss) + '\t' + str(val_acc))

    # saving the model checkpoint
    if (epoch+1) % 5 == 0:
        torch.save({'epoch': epoch+1, 'state_dict': model.state_dict()}, os.path.join('./models', 'model-epoch-{:02d}.ckpt'.format(epoch+1)))