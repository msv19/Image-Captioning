import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        #Embedding Vector - words into vectors
        self.word_embeddings = nn.Embedding(vocab_size, embed_size)
        
        #LSTM Input-Embedding Vector, Output - Hidden States
        self.lstm = nn.LSTM(embed_size,hidden_size,num_layers,batch_first= True)
        
        # FC Layer
        self.fc = nn.Linear(hidden_size,vocab_size)
        

    
    def forward(self, features, captions):
        
        captions = captions[:, :-1]   
        
        #Vector for each word  
        captions = self.word_embeddings(captions)
        
        #Concat the features for image and caption
        inputs = torch.cat((features.unsqueeze(dim=1),captions), dim=1) 
        
        # Gets the output and hidden state from the LSTM afer passing through the word embeddings
        lstm_out, hidden = self.lstm(inputs)
        
        #FC layer
        outputs=self.fc(lstm_out)  
        
        return outputs 
        

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        outputs = []
        output_length = 0
        hidden = (torch.randn(self.num_layers, 1, self.hidden_size).to(inputs.device),
                  torch.randn(self.num_layers, 1, self.hidden_size).to(inputs.device))
        
        for ii in range(max_len):
            #LSTM layer
            output, hidden = self.lstm(inputs,hidden)
            # FC Layer
            output = self.fc(output.squeeze(dim=1))
            _,index = torch.max(output,1)
            
            #CUDA Tensor to CPU and then to Numpy
            outputs.append(index.cpu().numpy()[0].item())
            
            if (index ==1) :
                break
            
            # Embed the predicted word as the new input to LSTM
            inputs = self.word_embeddings(index)   
            inputs = inputs.unsqueeze(1)  
            
            output_length +=1
         
        return outputs
               