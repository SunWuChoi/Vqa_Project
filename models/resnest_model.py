import torch
import torch.nn as nn
import torchvision.models as models

class ImgEncoder(nn.Module):

    def __init__(self, feature_size):
        super().__init__()
        # load the resnest model
        self.model = torch.hub.load('zhanghang1989/ResNeSt', 'resnest50', pretrained=True)
        # get input size of fc layer
        in_features = self.model.fc.in_features           
        # replace last fc with identity connection
        self.model.fc = nn.Identity()
        # add new last fc layer
        self.fc = nn.Linear(in_features, feature_size, bias=True)

    # forward pass of the image encoder
    def forward(self, x):
        with torch.no_grad():
            x = self.model(x)
        img_feat = self.fc(x)      
        return img_feat


class QstEncoder(nn.Module):

    def __init__(self, qst_vocab_size, qst_embed_size, feature_size, num_layers, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(qst_vocab_size, qst_embed_size)
        self.lstm = nn.LSTM(qst_embed_size, hidden_size, num_layers)
        self.tanh = nn.Tanh()
        self.fc = nn.Linear(2*num_layers*hidden_size, feature_size)

    def forward(self, x):
        # embed the input question
        x = self.embedding(x)                           
        x = self.tanh(x)
        x = x.transpose(0, 1) 
        # 2 lstm cells
        _, (hidden, cell) = self.lstm(x)                      
        qst_feat = torch.cat((hidden, cell), 2)                    
        qst_feat = qst_feat.transpose(0, 1)                     
        qst_feat = qst_feat.reshape(qst_feat.size()[0], -1)
        # tanh non linearity
        qst_feat = self.tanh(qst_feat)
        # final fc layer
        qst_feat = self.fc(qst_feat)                         
        return qst_feat


class VqaModel(nn.Module):
    def __init__(self, embed_size, qst_vocab_size, ans_vocab_size, word_embed_size, num_layers, hidden_size): 
        super(VqaModel, self).__init__()
        self.img_encoder = ImgEncoder(embed_size)
        self.qst_encoder = QstEncoder(qst_vocab_size, word_embed_size, embed_size, num_layers, hidden_size)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(0.5)
        # defining the 2 fc layers of the answer decoder
        self.fc1 = nn.Linear(embed_size, ans_vocab_size)
        self.fc2 = nn.Linear(ans_vocab_size, ans_vocab_size)

    def forward(self, image, question):
        img_feat = self.img_encoder(image)                     
        qst_feat = self.qst_encoder(question)  
        # fuse the image and question features
        fused_feat = torch.mul(img_feat, qst_feat)
        fused_feat = self.tanh(fused_feat)
        fused_feat = self.dropout(fused_feat)
        # pass through two fc layers
        fused_feat = self.fc1(fused_feat)
        fused_feat = self.tanh(fused_feat)
        fused_feat = self.dropout(fused_feat)
        fused_feat = self.fc2(fused_feat)
        return fused_feat