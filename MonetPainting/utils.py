from torch import nn 
import torch 
import matplotlib.pyplot as plt 

def unnorm(img, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
  for t,m, s in zip(img, mean, std):
    (t.mul_(s)).add_(s)
  
  return img

def init_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
    
def saving_checkpoint(checkpont_file, gen_mtp, gen_ptm,
                                      disc_m, disc_p, adam_gen, adam_disc):
    
    print("saving checkpoint " + checkpont_file)
    checkpoint = {
        "gen_mtp" : gen_mtp.state_dict(),
        "gen_ptm" : gen_ptm.state_dict(),
        "disc_m" : disc_m.state_dict(),
        "disc_p" : disc_p.state_dict(),
        "optimizer_gen" : adam_gen.state_dict(),
        "optimizer_disc" : adam_disc.state_dict(),
    }
    torch.save(checkpoint, checkpont_file)
    
    
def loading_checkpoint(checkpoint_file, gen_mtp, gen_ptm,
                       disc_m, disc_p, adam_gen, adam_disc, device):
    
    print("loading :" + checkpoint_file)
    checkpoint = torch.load(checkpoint_file, map_location=device)
    
    gen_mtp.load_state_dict(checkpoint["gen_mtp"])
    gen_ptm.load_state_dict(checkpoint["gen_ptm"])
    disc_m.load_state_dict(checkpoint["disc_m"])
    disc_p.load_state_dict(checkpoint["disc_p"])
    adam_gen.load_state_dict(checkpoint["optimizer_gen"])
    adam_disc.load_state_dict(checkpoint["optimizer_disc"])

    
class lr_scheduler():
    def __init__(self, total_epochs):
        self.total_epochs = total_epochs
        self.decay_epochs = int(total_epochs/2)
        
    def step(self, epoch_num):
        if (epoch_num < self.decay_epochs):
            return 1
        else:
            return 1-(epoch_num - self.decay_epochs)/(self.total_epochs - self.decay_epochs)
    
def show_images(d_loader, gen_ptm):
    _, ax = plt.subplots(4, 2, figsize=(8,12))
    d_iterator = iter(d_loader)
    for i in range(4):
        photo_img, _ = next(d_iterator)
        pred_monet_img = gen_ptm(photo_img.to("cuda")).cpu().detach() 
        ax[i, 0].imshow(unnorm(photo_img[0]).permute(1, 2, 0))
        ax[i, 0].set_title("photo")
        ax[i, 0].axis("off")
        ax[i, 1].imshow(unnorm(pred_monet_img[0]).permute(1, 2, 0))
        ax[i, 1].set_title("predicted monet")
        ax[i, 1].axis("off")