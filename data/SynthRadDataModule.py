from data.datautils import SynthRAD2025findImages
from data.transforms import get_transforms



class MNISTDataModule(L.LightningDataModule):
    
    
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.train_image_paths,self.val_image_paths = SynthRAD2025findImages(self.opt.basic_information)
        self.train_transform = get_transforms(self.opt.train)
        self.val_transform = get_transforms(self.opt.validation)  ### validation and test transform are the same
    def prepare_data(self):
        train_image_paths, val_image_paths = SynthRAD2025findImages(self.opt.basic_information)

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            train_dataset = CacheDataset(
                data=self.train_image_paths,
                transform=self.train_transforms,
                cache_rate=self.opt.train.cache_rate,
                num_workers=self.opt.train.cacheDataset_num_workers)

            val_dataset = CacheDataset(
                data=self.val_image_paths,
                transform=self.val_transforms,
                cache_rate=self.opt.train.cache_rate,
                num_workers=self.opt.train.cacheDataset_num_workers,
            )


        # Assign test dataset for use in dataloader(s)
    if stage == "test":
            self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.self.opt.train.batch_size,
                        shuffle=True, num_workers=self.self.opt.train.dataLoader_num_workers,
                        pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.opt.validation.batch_size,
                        shuffle=False, num_workers=self.opt.validation.dataLoader_num_workers,
                        pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=32)

