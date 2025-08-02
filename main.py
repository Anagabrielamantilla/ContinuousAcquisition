
import time
import wandb
import torch
import numpy as np
from utils import *
from tqdm import trange
from NetworkPaul import *
import torch.optim as optim
from matplotlib import gridspec
import matplotlib.pyplot as plt

wandb.login(key="29302aca9a6946fea3f9a038c6f03dce10af7b91")
wandb.init(project="geofisica", name="E1")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load S
S = np.load("Data/seismic_diffusion_11.npy")
S = torch.from_numpy(S)
S = S-S.min()
S = S/S.max()
S = S*2 - 1
S = S.to(device)

print('Maximum value of S', S.max())
print('Minimum value of S', S.min())

T1, M1 = uniform_sampling(S, 50)
T1 = T1.to(device)
M1 = M1.to(device)

batch_size = 4
dataset = dataset = SeismicDataset(T1, S, M1)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

network = AttU_Net(img_ch=10,output_ch=10).to(device)

binary_layer = MaskLayer((1024,128)).to(device)
reg =  SubsamplingRate(parameter=1, compression=0.9)  

optimizer = optim.Adam(list(network.parameters()) + list(binary_layer.parameters()), lr=1e-3)
criterion = nn.MSELoss()

start = time.time()

epochs = 100

list_loss1, list_loss2, list_loss3, list_loss_total = [], [], [], []

for epoch in trange(epochs, desc="Entrenamiento"):
    network.train()
    binary_layer.train()
    
    for batch_T1, batch_S, batch_M1 in dataloader:
        batch_T1 = batch_T1.to(device)
        batch_S = batch_S.to(device)
        batch_M1 = batch_M1.to(device)

        optimizer.zero_grad()
        
        M2 = binary_layer(batch_T1)
        M2 = M2*(1-batch_M1)
        T2 = M2*batch_S
        
        first_term = M2*network(batch_T1)
        second_term = T2

        loss1 = criterion(first_term, second_term)
        
        distance_m1, distance_m2 = binary_distance(batch_M1, M2)
        #loss2 = 1e-1*criterion(distance_m1/128, distance_m2/128)
        loss2 = torch.exp(-criterion(distance_m1/128, distance_m2/128))

        
        #loss3 = reg(M2)
        #loss3 = torch.sum((1-M1[0,0,0,:])*M2[0,0,0,:])
        loss3 = torch.mean(M2*batch_M1)

        loss = loss1 - loss2 + loss3

        list_loss1.append(loss1.item())
        list_loss2.append(loss2.item())
        list_loss3.append(loss3.item())
        list_loss_total.append(loss.item())

        loss.backward()
        optimizer.step()
    
    #loss_mean = list_loss1.mean()
    # Logging
    wandb.log({
        "loss": loss1.mean().item(),
        "loss1": loss1.item(),
        "loss2": loss2.item()
        }, step=epoch)

        # Visualización
    if (epoch + 1) % 1 == 0:
        with torch.no_grad():
            fig, ax = plt.subplots(3, 3, figsize=(12, 10))

            def imshow_with_colorbar(ax, data, title):
                im = ax.imshow(data, aspect='auto', cmap='gray')
                ax.set_title(title)
                cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.ax.tick_params(labelsize=8)

            # Usar primer elemento del batch
            b0 = 0
            imshow_with_colorbar(ax[0, 0], batch_S[b0, 0].cpu().numpy(), 'S')
            imshow_with_colorbar(ax[0, 1], batch_M1[b0, 0, :25, :25].cpu().numpy(), 'M1')
            imshow_with_colorbar(ax[0, 2], batch_T1[b0, 0].cpu().numpy(), 'T1')
            imshow_with_colorbar(ax[1, 0], first_term[b0, 0].detach().cpu().numpy(), 'M2*G(T1)')
            imshow_with_colorbar(ax[1, 1], second_term[b0, 0].detach().cpu().numpy(), 'M2*(I-M1)S')
            imshow_with_colorbar(ax[2, 0], M2[b0, 0, :25, :25].detach().cpu().numpy(), 'M2')
            #imshow_with_colorbar(ax[2, 0], M2[b0, 0].detach().cpu().numpy(), 'M2')
            
            
            # Distancia entre máscaras
            ax[1, 2].plot(distance_m1.detach().cpu(), label='Distance M1', marker='o', color='blue')
            ax[1, 2].plot(distance_m2.detach().cpu(), label='Distance M2', marker='x', color='green')
            #ax[1, 2].plot(abs(distance_m1.detach().cpu() - distance_m2.detach().cpu()), color='purple', label='|Dist M1 - M2|')
            ax[1, 2].set_title('B(M1,r1) - B(M2,r2)')
            ax[1, 2].legend()

            # Gráfico de pérdidas
            ax[2, 1].plot(list_loss1, marker='o', color='orange', label='Loss1')
            ax[2, 1].plot(list_loss2, marker='x', color='cyan', label='Loss2')
            ax[2, 1].plot(list_loss3, marker='x', color='blue', label='Loss3')
            ax[2, 1].plot(list_loss_total, linestyle='-', color='red', label='Loss Total')
            ax[2, 1].set_title('Loss')
            ax[2, 1].legend()
            
            ax[2, 2].axis('off')  # espacio libre
            
            plt.tight_layout()
            wandb.log({"visualización_epoch": wandb.Image(fig)}, step=epoch)
            plt.close(fig)

end = time.time()
print('Total time of computing [s]', end-start)

'''
fig, ax = plt.subplots(1, 3, figsize=(12, 10))
ax[0].imshow(1-batch_M1[0, 0, :50, :50].detach().cpu().numpy(),aspect='auto', cmap='gray')
ax[1].imshow(M2[0, 0, :50, :50].detach().cpu().numpy(),aspect='auto', cmap='gray')
ax[2].imshow(batch_M1[0, 0, :50, :50].detach().cpu().numpy() - M2[0, 0, :50, :50].detach().cpu().numpy(),aspect='auto', cmap='gray')
'''