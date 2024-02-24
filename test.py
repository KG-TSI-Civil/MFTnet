import torch
import torch.utils.data as Data
import numpy as np


def results(origin,pred,name,folder):  

    origin = np.array(origin,dtype="float64")
    pred = np.array(pred,dtype="float64")
    origin=np.round(origin,4)
    pred=np.round(pred,4)

    mea = np.array([np.mean(origin-pred)])
    absmae = np.array([np.mean(abs(origin-pred))])
    std = np.array([np.std(origin-pred)])
    writetxt = np.concatenate((mea,absmae,std)) 
    np.savetxt(folder + "/" +name+'_pred_results.txt', writetxt)

    print(name+"test: mean    abs mean    std")
    print(mea)
    print(absmae)
    print(std)

    return writetxt


def Test(test_dataset,model,folder):

    torch.set_float32_matmul_precision('high') 

    test_loader=Data.DataLoader(dataset=test_dataset,batch_size=128,shuffle=False,num_workers=0,pin_memory=True)

    if torch.cuda.is_available():
        model = model.cuda()

    model.double()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)


    mag_origin,mag_pred  = [],[]
    epi_origin,epi_pred = [],[]
    ptime_origin,ptime_pred=[],[]
    dep_origin,dep_pred = [],[]
    model.eval()
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            model=torch.load(folder + "/" +'bestmodel.pth')
            output_pred = model(inputs)
            magpred,epipred,ptimepred,deppred = output_pred[0],output_pred[1],output_pred[2],output_pred[3]
            mag_origin.extend(labels[:,0:1].tolist())
            mag_pred.extend(magpred.tolist())
            epi_origin.extend(labels[:,1:2].tolist())
            epi_pred.extend(epipred.tolist())
            ptime_origin.extend(labels[:,2:3].tolist())
            ptime_pred.extend(ptimepred.tolist())
            dep_origin.extend(labels[:,3:4].tolist())
            dep_pred.extend(deppred.tolist())

    print("Number of test:")
    print(len(mag_origin))

    magresults=results(mag_origin,mag_pred,"Magnitude",folder)
    epiresults=results(epi_origin,epi_pred,"Distance",folder)
    ptravelresults=results(ptime_origin,ptime_pred,"P Travel Time",folder)
    depresults=results(dep_origin,dep_pred,"Depth",folder)
    allresults = np.concatenate((magresults,epiresults,ptravelresults,depresults)) 
    np.savetxt(folder + "/" +'all_results.txt', allresults)

