from torchvision import transforms
import os
import torch 
import utils 
from tqdm import tqdm 
from PIL import Image 
from torch.utils import DataLoader, Dataset  
from train import CycleGAN
from dataset import ImageDataset
import shutil 

# Code for submitting in kaggle 
class PhotoDataset(Dataset):
    def __init__(self, root_photo):
        super().__init__()
        self.root_photo = root_photo
        self.photo_images = os.listdir(self.root_photo)
        
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5), std=(0.5))
        ])

    def __len__(self):
        return len(self.photo_images)

    def __getitem__(self, idx):
        photo_img = self.photo_images[idx]
        p_img = Image.open(os.path.join(self.root_photo, photo_img)).convert("RGB")
        
        return self.transform(p_img)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
monet_dir = "../input/gan-getting-started/monet_jpg"
photo_dir = "../input/gan-getting-started/photo_jpg" 

dataset = ImageDataset(monet_dir, photo_dir)
dataloader = DataLoader(dataset, batch_size=1, pin_memory=True)
gan = CycleGAN(80, device)
gan.train(dataloader)

    
photo_dataset = PhotoDataset("../input/gan-getting-started/photo_jpg")
photo_loader = DataLoader(photo_dataset, batch_size=1, pin_memory=True)
#mkdir ../images
t = tqdm(photo_loader, leave=False, total=photo_loader.__len__())
trans = transforms.ToPILImage()

print("successfully completed !")
for i, photo in enumerate(t):
    with torch.no_grad():
        pred_monet = gan.gen_ptm(photo.to(gan.device)).cpu().detach()
        pred_monet = utils.unnorm(pred_monet)
        print(pred_monet.shape)
        img = trans(pred_monet[0]).convert("RGB")
        img.save("../images/" + str(i+1) + ".jpg")


#shutil.make_archive("/kaggle/working/images", 'zip', "/kaggle/images")