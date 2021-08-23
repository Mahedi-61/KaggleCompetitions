from torchvision import transforms
import os
from torch.utils.data import Dataset 
from PIL import Image 

class ImageDataset(Dataset):
    def __init__(self, root_monet, root_photo):
        super().__init__()
        self.root_monet = root_monet 
        self.root_photo = root_photo

        self.monet_images = os.listdir(self.root_monet)
        self.photo_images = os.listdir(self.root_photo)

        self.monet_len = len(self.monet_images)
        self.photo_len = len(self.photo_images)
        
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5), std=(0.5))
        ])

    def __len__(self):
        return min(self.monet_len, self.photo_len)

    def __getitem__(self, idx):
        monet_img = self.monet_images[idx % self.monet_len]
        photo_img = self.photo_images[idx % self.photo_len]

        m_img = Image.open(os.path.join(self.root_monet, monet_img)).convert("RGB")
        p_img = Image.open(os.path.join(self.root_photo, photo_img)).convert("RGB")
        
        return self.transform(p_img), self.transform(m_img)