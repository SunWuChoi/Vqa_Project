import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class MutanFusion(nn.Module):
    
    def __init__(self, input_dim, out_dim, num_layers):
        super().__init__()
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.num_layers = num_layers

        hv = []
        for i in range(self.num_layers):
            do = nn.Dropout(p=0.5)
            lin = nn.Linear(input_dim, out_dim)

            hv.append(nn.Sequential(do, lin, nn.Tanh()))
        
        self.image_transformation_layers = nn.ModuleList(hv)
        
        hq = []
        for i in range(self.num_layers):
            do = nn.Dropout(p=0.5)
            lin = nn.Linear(input_dim, out_dim)
            hq.append(nn.Sequential(do, lin, nn.Tanh()))
            
        self.ques_transformation_layers = nn.ModuleList(hq)

    def forward(self, ques_emb, img_emb):
        batch_size = img_emb.size()[0]
        x_mm = []
        for i in range(self.num_layers):
            x_hv = img_emb
            x_hv = self.image_transformation_layers[i](x_hv)

            x_hq = ques_emb
            x_hq = self.ques_transformation_layers[i](x_hq)
            x_mm.append(torch.mul(x_hq, x_hv))
        
        x_mm = torch.stack(x_mm, dim=1)
        x_mm = x_mm.sum(1).view(batch_size, self.out_dim)
        x_mm = torch.tanh(x_mm)
        return x_mm
    

class ImgEncoder(nn.Module):

    def __init__(self, feature_size):
        super().__init__()
        # load the vgg19 model
        self.model = models.vgg19(pretrained=True)
        # get input size of fc layer
        in_features = self.model.classifier[-1].in_features
        # remove the last fc layer
        self.model.classifier = nn.Sequential(*list(self.model.classifier.children())[:-1])
        # add new last fc layer
        self.fc = nn.Linear(in_features, feature_size)
        
    # forward pass of the image encoder
    def forward(self, image):
        with torch.no_grad():
            img_feat = self.model(image)
        img_feat = self.fc(img_feat)

        l2_norm = img_feat.norm(p=2, dim=1, keepdim=True).detach()
        img_feat = img_feat.div(l2_norm)

        return img_feat
    

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


class VqaModel(nn.Module):

    def __init__(self, feature_size, qst_vocab_size, ans_vocab_size, qst_embed_size, num_layers, hidden_size):
        super().__init__()
        self.img_encoder = ImgEncoder(feature_size)
        self.qst_encoder = QstEncoder(qst_vocab_size, qst_embed_size, feature_size, num_layers, hidden_size)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(feature_size, ans_vocab_size)
        self.fc2 = nn.Linear(ans_vocab_size, ans_vocab_size)
        
        self.mutan = MutanFusion(feature_size, feature_size, 5)
        self.mlp = nn.Sequential(nn.Linear(feature_size, ans_vocab_size))
        
    def forward(self, image, question):
        img_feat = self.img_encoder(image)                    
        qst_feat = self.qst_encoder(question)                    
        fused_feat = self.mlp(self.mutan(qst_feat, img_feat))

        return fused_feat