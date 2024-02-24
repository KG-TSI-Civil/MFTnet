"""
@Reference: https://arxiv.org/abs/1705.07115
"""

import torch
import torch.nn as nn
import torch.utils.data as Data
import numpy as np
import matplotlib.pyplot as plt
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm


class MultiTaskLoss(nn.Module):
    def __init__(self):
        super(MultiTaskLoss, self).__init__()
        self.mse_loss1 = nn.MSELoss()
        self.mse_loss2 = nn.MSELoss()
        self.mse_loss3 = nn.MSELoss()
        self.mse_loss4 = nn.MSELoss()
        self.log_vars = nn.Parameter(torch.tensor([-3.6,2.,-1.4,3.], requires_grad = True))
        
    def forward(self, output1, output2, output3,output4,target1, target2,target3,target4):
        loss1 = self.mse_loss1(output1, target1)
        loss2 = self.mse_loss2(output2, target2)
        loss3 = self.mse_loss3(output3, target3)
        loss4 = self.mse_loss4(output4, target4)
        loss =  torch.exp(-self.log_vars[0]) * loss1 + torch.exp(-self.log_vars[1]) * loss2 + torch.exp(-self.log_vars[2]) * loss3 + torch.exp(-self.log_vars[3]) * loss4 + \
            self.log_vars[0] + self.log_vars[1] + self.log_vars[2] + self.log_vars[3]

        # log_var is the log value of variance

        return loss1,loss2,loss3,loss4,loss


def plotloss(epoch,tloss,vloss,name,folder):
    fig, ax = plt.subplots()
    plt.plot(epoch,tloss, label= name+' Train loss')
    plt.plot(epoch,vloss, label = name + ' Validation loss')
    plt.legend()
    plt.savefig(folder + "/" +name + ' loss.png')
    plt.close(fig)


def Train(train_dataset,valid_dataset,model,folder,args):
    
    torch.set_float32_matmul_precision('high') 

    train_loader=Data.DataLoader(dataset = train_dataset,batch_size = args.batch_size_train,shuffle = True,num_workers = 0,pin_memory = True)
    valid_loader=Data.DataLoader(dataset = valid_dataset,batch_size = args.batch_size_valid,shuffle = True,num_workers = 0,pin_memory = True)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    loss_fn = MultiTaskLoss()  


    if torch.cuda.is_available():
        model = model.cuda()
    loss_fn = loss_fn.to(device)

    model.double()


    params = ([p for p in model.parameters()] + [pp for pp in loss_fn.parameters()])

    optimizer = torch.optim.Adam(params,lr = args.initial_lr,weight_decay = 5e-5)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor = args.factor, patience = args.patience_val, min_lr = 0.5e-6)

    num_epochs = args.maximun_epoch
    num_epochs_without_improvement = 0
    patiences = args.patience
    train_losses,valid_losses,epoch_list = [],[],[]
    t_mag_losses,t_epi_losses,t_ptime_losses,t_dep_losses = [],[],[],[]
    v_mag_losses,v_epi_losses,v_ptime_losses,v_dep_losses = [],[],[],[]

    if args.pretrained is not None:
        model=torch.load(args.pretrained)
        print("Load pretrained checkpoints")

    best_valid_loss = float('inf') 
    for epoch in range(num_epochs):
        model.train()
        trainlen=len(train_loader)
        train_loss,t_mag_loss,t_epi_loss,t_ptime_loss,t_dep_loss = 0.0,0.0,0.0,0.0,0.0
        with tqdm(total = trainlen, desc = 'Training') as pbar:
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                inputs = inputs.to(device)
                labels = labels.to(device)
                output_pred = model(inputs)
                magpred,epipred,ptimepred,deppred = output_pred[0],output_pred[1],output_pred[2],output_pred[3]
                loss1,loss2,loss3,loss4,loss = loss_fn(magpred,epipred,ptimepred,deppred, labels[:,0:1],labels[:,1:2],labels[:,2:3],labels[:,3:4])
                train_loss += loss.item()
                t_mag_loss += loss1.item()
                t_epi_loss += loss2.item()
                t_ptime_loss += loss3.item()
                t_dep_loss += loss4.item()
                #print(train_loss)
                loss.backward()
                optimizer.step()
                pbar.update(1)
        train_loss /= trainlen
        t_mag_loss /= trainlen
        t_epi_loss /= trainlen
        t_ptime_loss /= trainlen
        t_dep_loss /= trainlen
        train_losses.append(train_loss)
        t_mag_losses.append(t_mag_loss)
        t_epi_losses.append(t_epi_loss)
        t_ptime_losses.append(t_ptime_loss)
        t_dep_losses.append(t_dep_loss)


        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }, folder + '/' +'epoch{}.pt'.format(epoch))
        print(f'Epoch {epoch}: Train loss = {train_loss:.4f}')
        print(f'Epoch {epoch}: Current learning rate = {optimizer.param_groups[0]["lr"]:.8f}')

        model.eval()
        with torch.no_grad():
            valid_loss,v_mag_loss,v_epi_loss,v_ptime_loss,v_dep_loss = 0.0,0.0,0.0,0.0,0.0
            validlen=len(valid_loader)
            with tqdm(total = validlen, desc = 'Validation') as pbar:
                for inputs, labels in valid_loader:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    output_pred = model(inputs)
                    magpred,epipred,ptimepred,deppred = output_pred[0],output_pred[1],output_pred[2],output_pred[3]
                    loss11,loss22,loss33,loss44,v_loss = loss_fn(magpred,epipred,ptimepred,deppred, labels[:,0:1],labels[:,1:2],labels[:,2:3],labels[:,3:4])
                    valid_loss  += v_loss.item()
                    v_mag_loss += loss11.item()
                    v_epi_loss += loss22.item()
                    v_ptime_loss += loss33.item()
                    v_dep_loss += loss44.item()
                    pbar.update(1)
                    # print(valid_loss)
            valid_loss /= validlen
            v_mag_loss /= validlen
            v_epi_loss /= validlen
            v_ptime_loss /= validlen
            v_dep_loss /= validlen
            epoch_list.append(epoch)
            valid_losses.append(valid_loss)
            v_mag_losses.append(v_mag_loss)
            v_epi_losses.append(v_epi_loss)
            v_ptime_losses.append(v_ptime_loss)
            v_dep_losses.append(v_dep_loss)
            print(f'Epoch {epoch}: Validation loss = {valid_loss:.4f}, Mag loss = {v_mag_loss:.4f}, Epi loss = {v_epi_loss:.4f}, Ptravel loss = {v_ptime_loss:.4f}, Dep loss = {v_dep_loss:.4f}')

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                num_epochs_without_improvement = 0
                torch.save(model.state_dict(), folder + "/" +'best_model.pt')
                torch.save(model,folder + "/" +"bestmodel.pth")
            else:
                num_epochs_without_improvement += 1
            if num_epochs_without_improvement >= patiences:
                print('Early stopping!')
                break
            scheduler.step(valid_loss)

    print("learned weights:(log_variance)")
    print(loss_fn.log_vars.clone().detach())

    np.savetxt(folder + '/' +'logvar and var.txt', loss_fn.log_vars.clone().detach().cpu().numpy())

    plotloss(epoch_list,train_losses,valid_losses,"",folder)
    plotloss(epoch_list,t_mag_losses,v_mag_losses,"Magnitude",folder)
    plotloss(epoch_list,t_epi_losses,v_epi_losses,"Epicentral Distance",folder)
    plotloss(epoch_list,t_ptime_losses,v_ptime_losses,"P Travel Time",folder)
    plotloss(epoch_list,t_dep_losses,v_dep_losses,"Depth",folder)


