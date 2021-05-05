#%%
import numpy as np
import torch
import torch.optim as optim
from torch.autograd import grad
import matplotlib.pyplot as plt
import time
import copy
from scipy.integrate import odeint
from tqdm import tqdm
dtype=torch.float
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

%matplotlib inline

#%% function definitions
# Define the sin() activation function
class sinActivation(torch.nn.Module):
    @staticmethod
    def forward(input):
        return torch.sin(input)
        
# Define some more general functions
def gradient(x,f):
    # Calculate the derivative with auto-differention
    return grad([f], [x], grad_outputs=torch.ones(x.shape, dtype=dtype), create_graph=True)[0]

def perturbPoints(grid,xL,xR,sig=0.5):
#   stochastic perturbation of the evaluation points
#   force t[0]=xL  & force points to be in the t-interval
    delta_x = grid[1] - grid[0]  
    noise = delta_x * torch.randn_like(grid)*sig
    x = grid + noise
    x.data[2] = torch.ones(1,1)*(-1)
    x.data[x<xL]=xL - x.data[x<xL]
    x.data[x>xR]=2*xR - x.data[x>xR]
    x.data[0] = torch.ones(1,1)*xL

    x.data[-1] = torch.ones(1,1)*xR
    x.requires_grad = False
    return x

def parametricSolutions(x, nn, xL=0., xR=1., bc=0.):
    # parametric solutions 
    N1,N2 = nn(x)
    f = (1-torch.exp(-(x-xL)))*(1-torch.exp(x-xR))
    psi_hat  = bc  + f*N1
    return psi_hat


def hamEqs_Loss(x,psi, E):
    psi_dx = gradient(x,psi)
    psi_ddx= gradient(x,psi_dx)
    f = psi_ddx/2 + E*psi
    L  = (f.pow(2)).mean()
    return L
    
class qNN1(torch.nn.Module):
    def __init__(self, D_hid=10):
        super(qNN1,self).__init__()

        # Define the Activation
        # self.actF = torch.nn.Sigmoid()   
        self.actF = sinActivation()
        
        # define layers
        self.Ein    = torch.nn.Linear(1,1)
        self.Lin_1  = torch.nn.Linear(2, D_hid)
        self.Lin_2  = torch.nn.Linear(D_hid, D_hid)
        self.out    = torch.nn.Linear(D_hid, 1)

    def forward(self,t):
        In1 = self.Ein(torch.ones_like(t))
        L1 = self.Lin_1(torch.cat((t,In1),1))
        h1 = self.actF(L1)
        L2 = self.Lin_2(h1)
        h2 = self.actF(L2)
        out = self.out(h2)
        return out, In1

# Train the NN
def run_Scan_finitewell(xL, xR, bc, neurons, epochs, n_train,lr, minibatch_number = 1):
    fc0 = qNN1(neurons)
    fc1=0
    betas = [0.999, 0.9999]
    optimizer = optim.Adam(fc0.parameters(), lr=lr, betas=betas)
    Loss_history = []
    Llim =  1e+20
    En_loss_history = []
    boundary_loss_history = []
    nontriv_loss_history = []
    SE_loss_history = []
    Ennontriv_loss_history = []
    criteria_loss_history = []
    En_history = []
    EWall_history = []
    di = (None, 1e+20)
    dic = {0:di, 1:di, 2:di, 3:di, 4:di, 5:di, 6:di, 7:di, 8:di, 9:di}
    
    grid = torch.linspace(xL, xR, n_train).reshape(-1,1)
    
    ## TRAINING ITERATION    
    TeP0 = time.time()
    walle = -4
    last_psi_L = 0
    for tt in tqdm(range(epochs)):
# Perturbing the evaluation points & forcing t[0]=xL
        x=perturbPoints(grid,xL,xR,sig=.03*xR)
            
# BATCHING
        batch_size = int(n_train/minibatch_number)
        batch_start, batch_end = 0, batch_size

        idx = np.random.permutation(n_train)
        x_b = x[idx]
        x_b.requires_grad = True
        loss=0.0


        for nbatch in range(minibatch_number): 
# batch time set
            x_mb = x_b[batch_start:batch_end]

#  Network solutions 
            nn, En = fc0(x_mb)

            En_history.append(En[0].data.tolist()[0])

            psi  = parametricSolutions(x_mb, fc0, xL, xR, bc) 
            Ltot = hamEqs_Loss(x_mb, psi, En)
            SE_loss_history.append(Ltot) #
            
            criteria_loss = Ltot #(psi_f.pow(2)).mean() +

            if tt%2500 == 0:
              walle += 1
            Ltot += 1/((psi.pow(2)).mean()+1e-6) + 1/(En.pow(2).mean()+1e-6) + torch.exp(-1*En+walle).mean() #(psi_f.pow(2)).mean()+
            En_loss_history.append(torch.exp(-1*En+walle).mean()) #
            EWall_history.append(walle)

            
            
            #boundary_loss_history.append((psi_f.pow(2)).mean()) # 
            nontriv_loss_history.append(1/((psi.pow(2)).mean()+1e-6)) #
            Ennontriv_loss_history.append(1/(En.pow(2).mean()+1e-6)) #
            criteria_loss_history.append(criteria_loss)
# OPTIMIZER
            Ltot.backward(retain_graph=False) #True
            optimizer.step()
            loss += Ltot.data.numpy()
            optimizer.zero_grad()

            batch_start +=batch_size
            batch_end +=batch_size

# keep the loss function history
        Loss_history.append(loss)       

#Keep the best model (lowest loss) by using a deep copy
        if  criteria_loss < Llim:
            fc1 =  copy.deepcopy(fc0)
            Llim=criteria_loss

        E_bin = abs(En[0].data.tolist()[0]//10) 
        if criteria_loss < dic[E_bin][1]:
          dic[E_bin] = (copy.deepcopy(fc0), criteria_loss)

    TePf = time.time()
    runTime = TePf - TeP0  
    loss_histories = (Loss_history, boundary_loss_history, nontriv_loss_history, SE_loss_history, Ennontriv_loss_history, En_loss_history, criteria_loss_history, fc0, En_history, EWall_history, dic)
    return fc1, loss_histories, runTime

## Train the model 

xL = 0.
xR = 1.
xBC1=0.

n_train, neurons, epochs, lr,mb = 100, 50, int(125e3), 8e-3, 1 
model1,loss_hists1,runTime1 = run_Scan_finitewell(xL, xR, xBC1, neurons, epochs, n_train, lr, mb)

# %%

plt.figure(figsize = (8,6))
plt.plot(np.arange(len(loss_hists1[8]))/1000, loss_hists1[8])
plt.tight_layout()
plt.ylabel('Model Energy History')
plt.xlabel('Thousand Epochs')
# %%
# Loss function
print('Training time (minutes):', runTime1/60)
plt.loglog(loss_hists1[0],'-b',alpha=0.975)
#plt.axvline(x = aarg)

plt.tight_layout()
plt.ylabel('Total Loss')
plt.xlabel('Epochs')

# %%
# Loss function
# print('Training time (minutes):', runTime1/60)
plt.loglog(loss_hists1[3],'-b',alpha=0.975, label='SE loss')
plt.loglog(loss_hists1[0],'-r',alpha=0.975, label='Total loss')
#plt.axvline(x = aarg)
plt.legend()

plt.tight_layout()
plt.ylabel('Loss')
plt.xlabel('Epochs')

# %%
# TEST THE PREDICTED SOLUTIONS
nTest = n_train; tTest = torch.linspace(xL-.1,xR+.1,nTest)
tTest = tTest.reshape(-1,1);
tTest.requires_grad=True
t_net = tTest.detach().numpy()
psi =parametricSolutions(tTest,model1,xL,xBC1) 
psi=psi.data.numpy();

plt.figure(figsize = (8,6))
psi_0to10 = parametricSolutions(tTest,loss_hists1[10][0][0],xL,xR,xBC1)
psi_10to20 = parametricSolutions(tTest,loss_hists1[10][1][0],xL,xR,xBC1)
# psi_40to50 = parametricSolutions(tTest,loss_hists1[10][4][0],xL,xR,xBC1)

plt.plot(t_net, -1*psi_0to10.data.numpy()/np.max(np.abs(psi_0to10.data.numpy())), '-r', linewidth=1, label = 'n = 1')
plt.plot(t_net, -1*psi_10to20.data.numpy()/np.max(np.abs(psi_10to20.data.numpy())), '-b', linewidth=1, label = 'n = 2')
# plt.plot(t_net, 1*psi_40to50.data.numpy()/np.max(np.abs(psi_40to50.data.numpy())), '-g', linewidth=1, label = 'n = 3')
plt.ylabel('$\psi(x)$')
plt.xlabel('x')
plt.legend()
# %%
