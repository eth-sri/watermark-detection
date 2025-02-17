import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from src.utils import logit


class DeltaModel(nn.Module):
    def __init__(self,num_classes=3):
        super(DeltaModel, self).__init__()
        self.delta = nn.Parameter(torch.tensor(0.0)) 
        self.x = nn.Parameter(torch.randn(10, num_classes))  
        self.factor = 1.96
        
    def categorize_logits(self, logits):
        
        median = torch.median(logits, dim=1).values
        std = torch.median(torch.std(logits, dim = 0))
        delta = self.delta 
        
        classification = torch.zeros(logits.shape, dtype=torch.int)
        above = logits >= median[:,None] + self.factor*std
        above *= logits >= median[:,None] + delta / 2 
        
        classification[above] = 1
        
        classification = max_pooling(classification)
        return classification
    

    def forward(self, logits):
        
        x = torch.zeros((10,4))
        for i in range(4):
            x[:,i] = torch.median(logits[:,:,i], dim = 1).values
        
        x = self.x
        
        delta = self.delta

        c = torch.zeros_like(logits)
        for i in range(4):
            c[:, :, i] = self.categorize_logits(logits[:, :, i])
                       
        logits = torch.zeros_like(c)        
                        
        delta_x = torch.zeros_like(c)
        for t1 in range(delta_x.shape[0]):
            for t0 in range(delta_x.shape[2]):
                if t0 == 0:
                    delta_x[t1,:,t0] = delta*c[t1,:,t0]+ x[t1,t0]*0
                else:
                    delta_x[t1,:,t0] = delta*c[t1,:,t0] + x[t1,t0-1]
                    
        for i in range(4):
            logits[:, :, i] = delta_x[:, :, i] - torch.logsumexp(delta_x[:, :, [j for j in range(4) if j != i]], dim=2)

        return logits, c


    
def max_pooling(data):
    result = torch.zeros(data.shape[1], dtype=torch.int)
    
    t1_size = data.shape[0]
    
    data = data[:t1_size//2]
    
    for i in range(data.shape[1]):
        column = data[:, i]
        counts = torch.bincount(column, minlength=3)
        max_count = torch.max(counts)
        max_indices = torch.where(counts == max_count)[0]
        
        # If there's a tie or the max index has multiple entries
        if len(max_indices) > 1:
            result[i] = 1 
        else:
            result[i] = max_indices[0]
    
    return result


def estimate_delta_grad(data: np.array, verbose: bool = False, return_loss: bool = False, num_epochs: int = 250):

    randomize_logits = False

    logits = logit(data)    
    logits = logits.reshape(-1, 9, 4)

    if randomize_logits:
        logits = np.random.permutation(logits.flatten()).reshape(logits.shape)

    logits = torch.tensor(logits, dtype=torch.float32)

    model = DeltaModel()
    model.factor = 1.96

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.1)

    # Training loop
    loss_list = []
    delta_list = []
    
    for epoch in range(num_epochs):
        model.train()
        
        optimizer.zero_grad()
        outputs, _ = model(logits)
        loss = criterion(outputs, logits.squeeze(0))
        loss.backward()
        optimizer.step()
        
        loss_list.append(loss.item())
        delta_list.append(model.delta.item())
        
        if (epoch + 1) % 100 == 0:
            if verbose:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')


    delta_list = np.array(delta_list)
    delta_learned = delta_list[np.argmin(loss_list)]

    if return_loss:
        return delta_learned, loss_list, delta_list, outputs.cpu().detach().numpy(), model
    else:
        return delta_learned
    