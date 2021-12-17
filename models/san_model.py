import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class QstEncoder(nn.Module):

    def __init__(self, qst_vocab_size, qst_embed_size, feature_size, num_layers, hidden_size):
        super().__init__()
        self.word2vec = nn.Embedding(qst_vocab_size, qst_embed_size)
        self.tanh = nn.Tanh()
        self.lstm = nn.LSTM(qst_embed_size, hidden_size, num_layers)
        self.fc = nn.Linear(2*num_layers*hidden_size, feature_size)

    def forward(self, question):
        qst_vec = self.word2vec(question)                             
        qst_vec = self.tanh(qst_vec)
        qst_vec = qst_vec.transpose(0, 1)                             
        _, (hidden, cell) = self.lstm(qst_vec)                        
        qst_feat = torch.cat((hidden, cell), 2)                    
        qst_feat = qst_feat.transpose(0, 1)                     
        qst_feat = qst_feat.reshape(qst_feat.size()[0], -1)  
        qst_feat = self.tanh(qst_feat)
        qst_feat = self.fc(qst_feat)                            

        return qst_feat

class ImgAttentionEncoder(nn.Module):

    def __init__(self, feature_size):
        super().__init__()
        vggnet_feat = models.vgg19(pretrained=True).features
        modules = list(vggnet_feat.children())[:-2]
        self.cnn = nn.Sequential(*modules)
        self.fc = nn.Sequential(nn.Linear(self.cnn[-3].out_channels, feature_size), nn.Tanh())

    def forward(self, image):
        with torch.no_grad():
            img_feat = self.cnn(image)                           
        img_feat = img_feat.view(-1, 512, 196).transpose(1,2) 
        img_feat = self.fc(img_feat)                          

        return img_feat


class Attention(nn.Module):
    
    def __init__(self, num_channels, feature_size, dropout=True):
        super().__init__()
        self.ff_image = nn.Linear(feature_size, num_channels)
        self.ff_questions = nn.Linear(feature_size, num_channels)
        self.dropout = nn.Dropout(p=0.5)
        self.ff_attention = nn.Linear(num_channels, 1)

    def forward(self, vi, vq):
        hi = self.ff_image(vi)
        hq = self.ff_questions(vq).unsqueeze(dim=1)
        ha = torch.tanh(hi+hq)
        if self.dropout:
            ha = self.dropout(ha)
        ha = self.ff_attention(ha)
        pi = torch.softmax(ha, dim=1)
        self.pi = pi
        vi_attended = (pi * vi).sum(dim=1)
        u = vi_attended + vq
        return u

class VqaModel(nn.Module):
    
    def __init__(self, feature_size, qst_vocab_size, ans_vocab_size, qst_embed_size, num_layers, hidden_size): 
        super().__init__()
        self.num_attention_layer = 2
        self.num_mlp_layer = 1
        self.img_encoder = ImgAttentionEncoder(feature_size)
        self.qst_encoder = QstEncoder(qst_vocab_size, qst_embed_size, feature_size, num_layers, hidden_size)
        self.san = nn.ModuleList([Attention(512, feature_size)]*self.num_attention_layer)
        self.tanh = nn.Tanh()
        self.mlp = nn.Sequential(nn.Dropout(p=0.5), nn.Linear(feature_size, ans_vocab_size))
        self.attn_features = []

    def forward(self, image, question):
        img_feat = self.img_encoder(image)
        qst_feat = self.qst_encoder(question)
        vi = img_feat
        u = qst_feat
        for attn_layer in self.san:
            u = attn_layer(vi, u)
            
        fused_feat = self.mlp(u)
        return fused_feat
