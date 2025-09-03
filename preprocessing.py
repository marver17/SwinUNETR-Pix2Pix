import os
import torch
import multiprocessing
from monai.data import Dataset, DataLoader, NibabelWriter
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, EnsureTyped, Spacingd, Orientationd,Resized
from data.solid_head_transform import GenerateSolidHeadMaskd,ApplyMaskAsBackgroundd
import matplotlib.pyplot as plt
import  yaml

from data.solid_head_transform import GenerateSolidHeadMaskd, ApplyMaskAsBackgroundd
from utils import util
from data.datautils import MR2CTfindImages
from accelerate import Accelerator
from tqdm.auto import tqdm
import multiprocessing

def main():
    # Inizializza l'accelerator
    accelerator = Accelerator()
    multiprocessing.set_start_method('spawn')

    with open('experiments/MR2CT/options_1.yaml', 'r') as file:
        options = yaml.safe_load(file)

    opt = util.DotDict(options)
    dataset_option = util.DotDict(options['data'])
    training_option = util.DotDict(options["training"])


    device = accelerator.device
    pipeline = Compose([
        LoadImaged(keys=["CT", "MR"], reader="ITKReader",image_only=False),
        EnsureChannelFirstd(keys=["CT", "MR"]),
        
        # Sposta i dati sul dispositivo target PRIMA delle operazioni pesanti
        EnsureTyped(keys=["CT", "MR"], device=device, track_meta=True),
        
        Spacingd(keys=["CT", "MR"], pixdim=(1.5, 1.5, 1.5), mode="trilinear"),
        Orientationd(keys=["CT", "MR"], axcodes="RAS"),
        Resized(
        keys=["CT","MR"], spatial_size=[193, 193, 229],mode="trilinear"),


        GenerateSolidHeadMaskd(keys=("CT", "MR"), new_key="head_mask"),
        ApplyMaskAsBackgroundd(
            keys=["CT", "MR"], 
            mask_key="head_mask", 
            background_values={"CT": -1024, "MR": 0}
        ),
            EnsureTyped(keys=["CT","MR"] + ["head_mask"], device=torch.device("cpu")),
    ])
    train_image_paths, val_image_paths = MR2CTfindImages(dataset_option.basic_information,False)
    test_image_paths = MR2CTfindImages(dataset_option.basic_information,True)
    image_paths = train_image_paths + val_image_paths + test_image_paths
    dataset = Dataset(
        data=image_paths,
        transform=pipeline,
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=15, 
        pin_memory=True,
    )
    dataloader = accelerator.prepare(dataloader)
    
    writer = NibabelWriter()

    # TQDM "process-aware"
    progress_bar = tqdm(
        range(len(dataloader)), 
        disable=not accelerator.is_main_process
    )
    for batch_data in dataloader:
        # Il loop di salvataggio rimane quasi identico, perché ogni processo
        # riceve batch diversi e scrive su file diversi.
        for i in range(len(batch_data["CT"])):
            ct_tensor = batch_data["CT"][i]       
            mr_tensor = batch_data["MR"][i]
            
            ct_meta = {key: val[i] for key, val in batch_data["CT_meta_dict"].items()}
            mr_meta = {key: val[i] for key, val in batch_data["MR_meta_dict"].items()}

            ct_name = os.path.join(os.path.dirname(ct_meta["filename_or_obj"]),os.path.basename(ct_meta["filename_or_obj"]).replace("ctMRspace.nii.gz","ctMRspaceMRCTmasked.nii.gz"))
            mr_name = os.path.join(os.path.dirname(mr_meta["filename_or_obj"]),"processed","MR",os.path.basename(mr_meta["filename_or_obj"]).replace("_t1w.nii.gz","mrMRCTmasked.nii.gz"))

            os.makedirs(os.path.dirname(ct_name), exist_ok=True)
            writer.set_data_array(ct_tensor, channel_dim=0)
            writer.set_metadata(ct_meta)
            writer.write(ct_name, verbose=False)

            os.makedirs(os.path.dirname(mr_name), exist_ok=True)
            writer.set_data_array(mr_tensor, channel_dim=0)
            writer.set_metadata(mr_meta)
            writer.write(mr_name, verbose=False)
        
        progress_bar.update(1)

if __name__ == '__main__':
    main()