from PIL import Image
from torchvision.transforms import Compose, ToTensor, ToPILImage, CenterCrop, transforms, Resize
from torch.utils.data.dataset import Dataset

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])

hr_target_size = (1020, 2040)

hr_transform = transforms.Compose([
    transforms.Lambda(lambda img: img.rotate(0) if img.size[0] > img.size[1] else img.rotate(90)),
    transforms.CenterCrop(hr_target_size),
    transforms.ToTensor()
])

lr_target_size = (1020 // 4, 2040 // 4)

lr_transform = transforms.Compose([
    transforms.Lambda(lambda img: img.rotate(0) if img.size[0] > img.size[1] else img.rotate(90)),
    transforms.CenterCrop(lr_target_size),
    transforms.ToTensor()
])


class Div2kTrainDataset(Dataset):
    def __init__(self, hr_base_dir, lr_base_dir):
        super(Div2kTrainDataset, self).__init__()

        self.hr_base_dir = hr_base_dir
        self.lr_base_dir = lr_base_dir

        self.hr_image_filenames = [f'{self.hr_base_dir}/{i:0>4}.png' for i in range(1, 801)]
        self.lr_image_filenames = [f'{self.lr_base_dir}/{i:0>4}x4d.png' for i in range(1, 801)]

        self.hr_transform = hr_transform
        self.lr_transform = lr_transform

    def __getitem__(self, index):
        hr_image = self.hr_transform(Image.open(self.hr_image_filenames[index]))
        lr_image = self.lr_transform(Image.open(self.lr_image_filenames[index]))
        return hr_image, lr_image

    def __len__(self):
        return len(self.hr_image_filenames)

class Div2kValDataset(Dataset):
    def __init__(self, hr_base_dir, lr_base_dir):
        super(Div2kValDataset, self).__init__()

        self.hr_base_dir = hr_base_dir
        self.lr_base_dir = lr_base_dir

        self.hr_image_filenames = [f'{self.hr_base_dir}/{i:0>4}.png' for i in range(801, 901)]
        self.lr_image_filenames = [f'{self.lr_base_dir}/{i:0>4}x4d.png' for i in range(801, 901)]

        self.hr_transform = hr_transform
        self.lr_transform = lr_transform

    def __getitem__(self, index):
        hr_image = self.hr_transform(Image.open(self.hr_image_filenames[index]))
        lr_image = self.lr_transform(Image.open(self.lr_image_filenames[index]))
        return hr_image, lr_image

    def __len__(self):
        return len(self.hr_image_filenames)
    
def display_transform():
    return Compose([
        ToPILImage(),
        Resize(400),
        CenterCrop(400),
        ToTensor()
    ])