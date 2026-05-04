import torch

def get_optim_sched(network, optim, sched, lr, wd, T_max, lr_muon=None):
    if optim == 'adam':
        optimizer = torch.optim.Adam(network.parameters(), lr=lr, weight_decay=wd)  
    elif optim == 'adamw':
        optimizer = torch.optim.AdamW(network.parameters(), lr=lr, weight_decay=wd)
    elif optim == 'sgd':
        optimizer = torch.optim.SGD(network.parameters(), lr=lr, momentum=0.9, weight_decay=wd)   
    elif optim == 'sgd_nomom':
        optimizer = torch.optim.SGD(network.parameters(), lr=lr, weight_decay=wd)
    else:
        print("Invalid optimizer choice. Valid choices: [adam, sgd, adamw, sgd_nomom]")

    if sched == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = T_max)
    else:
        scheduler = None

    return optimizer, scheduler