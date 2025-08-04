import torch
import torch.fft
import torch.nn as nn
from typing import List, Optional
from torch.autograd import Function
from torch.utils.data import Dataset, DataLoader


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def uniform_sampling(data, num):
    indices = torch.linspace(0, data.shape[3] - 1, steps=num).long()
    mask = torch.ones_like(data)  
    mask[..., indices] = 0 
    return data * mask, mask
    
def binary_distance(M1, M2):
    receivers_coordinates = torch.arange(1,129).to(device)
    fila_M1 = M1[0, 0, 0, :]
    fila_M2 = M2[0, 0, 0, :]
    
    coord_M1 = fila_M1*receivers_coordinates
    coord_M2 = fila_M2*receivers_coordinates
      
    return coord_M1, coord_M2

class SeismicDataset(torch.utils.data.Dataset):
    def __init__(self, T1, S, M1):
        self.T1 = T1
        self.S = S
        self.M1 = M1

    def __len__(self):
        return self.T1.shape[0]

    def __getitem__(self, idx):
        return self.T1[idx], self.S[idx], self.M1[idx]

# Paso 1 : definir la clase de binarización 
class BinaryQuantize(Function):
    @staticmethod
    def forward(ctx, input):
        out = torch.sign(input) 
        out = (out+1)/2
        return out

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input


class BinaryQuantize_a(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        out = torch.sign(input)
        out = (out+1)/2
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors[0]
        grad_input = (2 - torch.abs(2*input))
        grad_input = grad_input.clamp(min=0) * grad_output.clone()
        return grad_input
        
        
# Paso 2:  Definir clase subsampling y forward de lo que se quiere hacer 
# Esta clase recibe datos de [M,N] es decir [128,128]
class MaskLayer(nn.Module):
    def __init__(self, im_size: tuple):
        super(MaskLayer, self).__init__()

        H, W = im_size
        self.H = H
        self.W = W
        
        mask = torch.normal(0, 1, size=(1, W)) #trazas
        mask = mask / torch.sqrt(torch.tensor(W).float())
        self.mask = nn.Parameter(mask, requires_grad=True)

    def forward(self, inputs):
        
        mask = self.get_mask()
        x = mask*inputs
        primera_fila = mask[0,0,0,:]
        
        # Contar los ceros
        num_ceros = torch.sum(primera_fila == 0).item()
        
        print(f"Columnas removidas: {num_ceros}")
        
        return mask

    def get_mask(self):
        mask = self.mask
        mask = torch.tile(mask, (self.H, 1))
        mask = mask.unsqueeze(0).unsqueeze(0)
        mask = BinaryQuantize.apply(mask)
        
        return mask
        
        
# Paso 3: Clase de regularización 

class SubsamplingRate(torch.nn.Module):
    def __init__(self, parameter, compression):
        super(SubsamplingRate, self).__init__()
        self.parameter = parameter
        self.compression = compression

    def forward(self, mask):
        regularization = torch.square((torch.sum(mask) / mask.numel()) - self.compression)
        regularization = self.parameter * regularization

        return regularization
