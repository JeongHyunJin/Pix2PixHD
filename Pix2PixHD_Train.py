# -*- coding: utf-8 -*-
"""
Train code for the Pix2PixHD Model

Created on Wed 12 Nov 2020

@author: Hyun-Jin Jeong (https://jeonghyunjin.com, jeong_hj@khu.ac.kr)

Reference:
1) https://github.com/JeongHyunJin/Jeong2020_SolarFarsideMagnetograms
2) https://iopscience.iop.org/article/10.3847/2041-8213/abc255
3) https://iopscience.iop.org/article/10.3847/2041-8213/ab9085 
"""

#------------------------------------------------------------------------------

if __name__ == '__main__':

    import os
    import torch
    from torch.utils.data import DataLoader
    from Pix2PixHD_Networks import Discriminator, Generator, Loss
    from Pix2PixHD_Options import TrainOption
    from Pix2PixHD_Pipeline import CustomDataset
    from Pix2PixHD_Utils import Manager, update_lr, weights_init
    from tqdm import tqdm
    import numpy as np
    import datetime        

    start_time = datetime.datetime.now()

#------------------------------------------------------------------------------
# [1] Initial Conditions Setup
    
    torch.backends.cudnn.benchmark = True

    opt = TrainOption().parse()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.gpu_ids)
    device = torch.device('cuda:0' )
    dtype = torch.float16 if opt.data_type == 16 else torch.float32

    if opt.val_during_train:
        from options import TestOption
        test_opt = TestOption().parse()
        save_freq = opt.save_freq

    init_lr = opt.lr
    lr = opt.lr

    dataset = CustomDataset(opt)
    data_loader = DataLoader(dataset=dataset,
                             batch_size=opt.batch_size,
                             num_workers=opt.n_workers,
                             shuffle=not opt.no_shuffle)

    G = Generator(opt).apply(weights_init).to(device=device, dtype=dtype)
    D = Discriminator(opt).apply(weights_init).to(device=device, dtype=dtype)

    criterion = Loss(opt)

    G_optim = torch.optim.Adam(G.parameters(), lr=lr, betas=(opt.beta1, opt.beta2), eps=opt.eps)
    D_optim = torch.optim.Adam(D.parameters(), lr=lr, betas=(opt.beta1, opt.beta2), eps=opt.eps)

    if opt.latest and os.path.isfile(opt.model_dir + '/' + str(opt.latest) + '_dict.pt'):
        pt_file = torch.load(opt.model_dir + '/' + str(opt.latest) + '_dict.pt')
        init_epoch = pt_file['Epoch']
        print("Resume at epoch: ", init_epoch)
        G.load_state_dict(pt_file['G_state_dict'])
        D.load_state_dict(pt_file['D_state_dict'])
        G_optim.load_state_dict(pt_file['G_optim_state_dict'])
        D_optim.load_state_dict(pt_file['D_optim_state_dict'])
        current_step = init_epoch * len(dataset)

        for param_group in G_optim.param_groups:
            lr = param_group['lr']

    else:
        init_epoch = 1
        current_step = 0

    manager = Manager(opt)
    
#------------------------------------------------------------------------------
# [2] Model training
    
    total_step = opt.n_epochs * len(data_loader)

    for epoch in range(init_epoch, opt.n_epochs + 1):
        for input, target, _, _ in tqdm(data_loader):
            G.train()
         
            current_step += 1
            input, target = input.to(device=device, dtype=dtype), target.to(device, dtype=dtype)

            D_loss, G_loss, target_tensor, generated_tensor = criterion(D, G, input, target)

            G_optim.zero_grad()
            G_loss.backward()
            G_optim.step()

            D_optim.zero_grad()
            D_loss.backward()
            D_optim.step()

            package = {'Epoch': epoch,
                       'current_step': current_step,
                       'total_step': total_step,
                       'D_loss': D_loss.detach().item(),
                       'G_loss': G_loss.detach().item(),
                       'D_state_dict': D.state_dict(),
                       'G_state_dict': G.state_dict(),
                       'D_optim_state_dict': D_optim.state_dict(),
                       'G_optim_state_dict': G_optim.state_dict(),
                       'target_tensor': target_tensor,
                       'generated_tensor': generated_tensor.detach()}

            manager(package)

#------------------------------------------------------------------------------
# [2] Model Checking 

            if opt.val_during_train and (current_step % save_freq == 0):
                G.eval()
                test_image_dir = os.path.join(test_opt.image_dir, str(current_step))
                os.makedirs(test_image_dir, exist_ok=True)
                test_model_dir = test_opt.model_dir

                test_dataset = CustomDataset(test_opt)
                test_data_loader = DataLoader(dataset=test_dataset,
                                              batch_size=test_opt.batch_size,
                                              num_workers=test_opt.n_workers,
                                              shuffle=not test_opt.no_shuffle)

                for p in G.parameters():
                    p.requires_grad_(False)

                for input, target, _, name in tqdm(test_data_loader):
                    input, target = input.to(device=device, dtype=dtype), target.to(device, dtype=dtype)
                    fake = G(input)
                    
                    np_fake = fake.cpu().numpy().squeeze()
                    np_real = target.cpu().numpy().squeeze()
                    
                    if opt.display_scale != 1:
                        np_fake = np.clip(np_fake*np.float(opt.display_scale), -1, 1)
                        np_real = np.clip(np_real*np.float(opt.display_scale), -1, 1)
                        
                    manager.save_image(np_fake, path=os.path.join(test_image_dir, 'Check_{:d}_'.format(current_step)+ name[0] + '_fake.png'))
                    manager.save_image(np_real, path=os.path.join(test_image_dir, 'Check_{:d}_'.format(current_step)+ name[0] + '_real.png'))
                    

                for p in G.parameters():
                    p.requires_grad_(True)

#------------------------------------------------------------------------------

        if epoch > opt.epoch_decay :
            lr = update_lr(lr, init_lr, opt.n_epochs - opt.epoch_decay, D_optim, G_optim)
    
    end_time = datetime.datetime.now()
    
    print("Total time taken: ", end_time - start_time)
