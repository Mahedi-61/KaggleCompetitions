import utils
from network import Generator, Discriminator 
import torch
from torch import nn
from tqdm import tqdm  

class CycleGAN(object):
    def __init__(self, num_epochs, device, lr=2e-4, lambda_cycle=10):
        super().__init__()
        
        self.epochs = num_epochs
        self.device = device
        self.lr = lr
        self.lambda_cycle = 10
        self.is_save = True
        self.is_load = False
        
        #intialize models
        self.gen_mtp = Generator().to(device)
        self.gen_ptm = Generator().to(device)
        self.disc_p = Discriminator().to(device)
        self.disc_m = Discriminator().to(device)
        
        utils.init_weights(self.gen_mtp)
        utils.init_weights(self.gen_ptm)
        utils.init_weights(self.disc_p)
        utils.init_weights(self.disc_m)
        

        self.adam_gen = torch.optim.Adam(
            list(self.gen_mtp.parameters()) + list(self.gen_ptm.parameters()),
            lr=self.lr, betas=(0.5, 0.99)
        )
        
        self.adam_disc = torch.optim.Adam(
            list(self.disc_p.parameters()) + list(self.disc_m.parameters()),
            lr=self.lr, betas=(0.5, 0.99)
        )
        
        if self.is_load:
            model_number = "1"
            checkpoint_file = "save_e"+ model_number + ".ckpt"
            utils.loading_checkpoint(checkpoint_file, self.gen_mtp, self.gen_ptm,
                       self.disc_m, self.disc_p, self.adam_gen, self.adam_disc, self.device)
        
        self.gen_lr_sched = torch.optim.lr_scheduler.LambdaLR(self.adam_gen, 
                            utils.lr_scheduler(self.epochs).step)
        self.disc_lr_sched = torch.optim.lr_scheduler.LambdaLR(self.adam_disc, 
                            utils.lr_scheduler(self.epochs).step)
        
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()
        
    def train(self, d_loader):
        for epoch in range(self.epochs):
            loop = tqdm(d_loader, leave=False, total=d_loader.__len__())
            
            # iterating the dataset
            for p_img, m_img in loop:
                p_img, m_img = p_img.to(self.device), m_img.to(self.device)

                # training Discriminator
                self.adam_disc.zero_grad()
                D_monet_real = self.disc_m(m_img)
                fake_m = self.gen_ptm(p_img)
                D_monet_fake = self.disc_m(fake_m)
                fake_p = self.gen_mtp(m_img)
                D_photo_real = self.disc_p(p_img)
                D_photo_fake = self.disc_p(fake_p)

                D_monet_rloss = self.mse(D_monet_real, torch.ones_like(D_monet_real))
                D_monet_floss = self.mse(D_monet_fake, torch.zeros_like(D_monet_fake))
                D_photo_rloss = self.mse(D_photo_real, torch.ones_like(D_photo_real))
                D_photo_floss = self.mse(D_monet_fake, torch.zeros_like(D_photo_fake))

                total_disc_loss = (
                    D_monet_rloss + D_monet_floss + D_photo_rloss + D_photo_floss
                )/2

                #backward
                total_disc_loss.backward(retain_graph=True)
                self.adam_disc.step()
                
                # training generator
                self.adam_gen.zero_grad()
                # forward pass (round-1)
                #fake_p = self.gen_mtp(m_img)
                disc_photo = self.disc_p(fake_p)
                cycle_m = self.gen_ptm(fake_p)

                # forward pass (round-2)
                #fake_m = self.gen_ptm(p_img)
                disc_monet = self.disc_m(fake_m)
                cycle_p = self.gen_mtp(fake_m)

                #identity
                id_m = self.gen_ptm(m_img)
                id_p = self.gen_mtp(p_img)

                # round-1 loss
                adver_loss_photo = self.mse(disc_photo, torch.ones_like(disc_photo))
                cycle_loss_monet = self.l1(cycle_m, m_img)
                identity_loss_monet = self.l1(id_m, m_img)

                # round-2 loss
                adver_loss_monet= self.mse(disc_monet, torch.ones_like(disc_monet))
                cycle_loss_photo = self.l1(cycle_p, p_img)
                identity_loss_photo = self.l1(id_p, p_img)

                total_gan_loss = (
                    adver_loss_photo + adver_loss_monet +
                    (cycle_loss_monet * self.lambda_cycle) + 
                    (cycle_loss_photo * self.lambda_cycle) +
                    (identity_loss_monet * self.lambda_cycle * 0.5) + 
                    (identity_loss_photo * self.lambda_cycle * 0.5)
                )
                #backward
                total_gan_loss.backward()
                self.adam_gen.step()
                
            # final tasks in one epoch
            print("Epoch: %d | generator loss: %f | discriminator loss %f" %
                 (epoch+1, total_gan_loss.item(), total_disc_loss.item()))

            self.gen_lr_sched.step()
            self.disc_lr_sched.step()

            utils.show_images(d_loader, self.gen_ptm)
            
            if (self.is_save and (epoch+1) %5) == 0:
                checkpoint_file = "save_e"+ str(epoch+1) + ".ckpt"
                utils.saving_checkpoint(checkpoint_file, 
                                  self.gen_mtp, self.gen_ptm,
                                  self.disc_m, self.disc_p,
                                  self.adam_gen, self.adam_disc)