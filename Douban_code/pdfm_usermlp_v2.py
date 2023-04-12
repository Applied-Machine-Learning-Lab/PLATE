import torch
#import torch.nn as nn
import numpy as np

from layer import FactorizationMachine, FeaturesEmbedding, FeaturesEmbedding2, FeaturesLinear, MultiLayerPerceptron, MultiLayerPerceptron_normal


class PromptDeepFactorizationMachineModel_usermlp(torch.nn.Module):
    """
    A pytorch implementation of DeepFM.

    Reference:
        H Guo, et al. DeepFM: A Factorization-Machine based Neural Network for CTR Prediction, 2017.
    """

    def __init__(self, field_dims, embed_dim, mlp_dims, dropout, domain_id):
        super().__init__()
        
        #self.prompt_dim = prompt_dim
        #self.prompt = nn.Parameter(torch.zeros(1, prompt_dim))  #initialize prompt = 0
        
        #fdim = np.array([prompt_dim]) #the dimension of feature field for prompt should be 1, add it to field_dims
        
        pdim = np.array([4]) #the dimension of feature field for prompt should be 1
        
        '''
        fdim = np.array([1]) #add the dimension of feature field for prompt
        
        for i in range(field_dims.shape[0]):
            fdim = np.append(fdim,[field_dims[i]])
        field_dims = fdim
        '''
        fdim = np.array([4])
        for i in range(field_dims.shape[0]):
            fdim = np.append(fdim,[field_dims[i]])
        self.embed_dim = embed_dim
        self.domain_id = domain_id
        self.linear = FeaturesLinear(fdim)
        self.fm = FactorizationMachine(reduce_sum=True)
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.embedding_prompt = FeaturesEmbedding(pdim, embed_dim)
        self.embed_output_dim = (len(field_dims)+2) * embed_dim
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, mlp_dims, dropout)
        self.mlp_2 = MultiLayerPerceptron_normal(embed_dim, (32, 32), dropout)
        

        
    def Freeze1(self):
        
        '''
        for param in self.parameters():
            param.requires_grad = True
            #if param=='promt' param.requires_grad = False
        '''

        #self.embedding_prompt.embedding.weight.requires_grad = False
        for param in self.parameters():
            param.requires_grad = True
        '''
        try:
            for param in self.head.parameters():
                param.requires_grad = True
        except:
            pass
        '''
        
    def Freeze2(self):#tune prompt
        for param in self.parameters():
            param.requires_grad = False
            #print(param)
            
        for name, param in self.named_parameters():
            #if 'output_l' in name:
            #print(name)
            if "mlp_2.mlp" in name:
                param.requires_grad = True

        self.embedding_prompt.embedding.weight.requires_grad = True
        
    def Freeze3(self):#tune prompt + head
        for param in self.parameters():
            param.requires_grad = False
            
        for name, param in self.named_parameters():
            #print(name)
            if 'output_l' in name:
            #if "mlp.mlp.8" in name:
                param.requires_grad = True
            if "mlp_2.mlp" in name:
                param.requires_grad = True

        self.embedding_prompt.embedding.weight.requires_grad = True
        
        #for i,p in enumerate(net.parameters()):
        #if i < 165:
        #    p.requires_grad = False

    def Freeze4(self):#tune prompt + linear + head
        for param in self.parameters():
            param.requires_grad = False
            
        for name, param in self.named_parameters():
            #print(name)
            if "mlp_2.mlp" in name:
                param.requires_grad = True
            if 'output_l' in name:
            #if "mlp.mlp.8" in name:
                param.requires_grad = True
            if 'linear' in name:
                param.requires_grad = True

        self.embedding_prompt.embedding.weight.requires_grad = True
        #self.linear.requires_grad = True
        #print(self.linear)
    
    def Freeze5(self):#tune prompt + linear
        for param in self.parameters():
            param.requires_grad = False
            
        for name, param in self.named_parameters():
            #print(name)
            if "mlp_2.mlp" in name:
                param.requires_grad = True
            if 'linear' in name:
                param.requires_grad = True

        self.embedding_prompt.embedding.weight.requires_grad = True
        #self.linear.requires_grad = True
        #print(self.linear)
        
    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        #print(x.shape)
        '''
        prompt = self.prompt.expand(x.shape[0], -1)
        x = torch.cat((prompt, x), dim=1)
        '''
        domain_id_l = x[:,0]
        domain_id_l = domain_id_l.view(domain_id_l.shape[0],1)
        lin = self.linear(x)

        x=x[:,1:]
        embed_x = self.embedding(x)
        
        prompt = self.embedding_prompt(domain_id_l)#domain prompt, first fix to 0, then train
        
        user_prompt = self.mlp_2(embed_x[:,0,:self.embed_dim])
        user_prompt = user_prompt.view(user_prompt.shape[0],1,user_prompt.shape[1])
        
        embed_x = torch.cat((prompt, user_prompt, embed_x), dim=1)

        x = lin + self.fm(embed_x) + self.mlp(embed_x.view(-1, self.embed_output_dim))
        return torch.sigmoid(x.squeeze(1))
