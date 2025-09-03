#### Inizializza repository settare in modon concorde il path dei dati, il path del repository e 
#### le altre variabili d'ambiente all'interno del file .env

from dotenv import load_dotenv
import os
import sys
import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint

load_dotenv()  
repo_path = os.getenv("REPO_PATH")
sys.path.append(repo_path)
os.chdir(repo_path)

import yaml
from utils import util
from data.transforms import get_transforms
from data.datautils import CT2MRfindImages,MR2CTfindImages
from monai.data import DataLoader, CacheDataset,PersistentDataset,Dataset
import matplotlib.pyplot as plt
import argparse
from models.pix2pix import Pix2Pix
from lightning.pytorch.loggers import TensorBoardLogger
import torch


def train_pix2pix(config_file,gpu_id = 0, parallelize_gpus= False,start_after_interuption=None,training_only = True):
    """
    Train a Pix2Pix model using the specified configuration file.

    Args:
        config_file (_type_): Path to the configuration file containing training options.
        gpu_id (int, optional): GPU ID to use for training. Defaults to 0.
        parallelize_gpus (bool, optional): Enable for parellize training across multiple GPUs. Defaults to False.
        start_after_interuption (_type_, optional): Restart training. Defaults to None.
        training_only (bool, optional): Adding testing phase to your model. Defaults to True.

    Raises:
        ValueError: _description_
        ValueError: _description_
    """
    
    print("Training Pix2Pix model...")
    
    with open(config_file, 'r') as file:
        options = yaml.safe_load(file)
        
    opt = util.DotDict(options)
    dataset_option = util.DotDict(options['data'])
    training_option = util.DotDict(options['training'])
    train_transforms  = get_transforms(dataset_option.train)
    validation_transforms = get_transforms(dataset_option.validation)
    torch.multiprocessing.set_start_method('spawn')

    if training_option.direction == "CTtoMR" : 
        train_image_paths, val_image_paths = CT2MRfindImages(dataset_option.basic_information,False)
    elif training_option.direction == "MRtoCT" : 
        train_image_paths, val_image_paths = MR2CTfindImages(dataset_option.basic_information,False)
    else :
        raise ValueError("Invalid direction specified. Use 'CTtoMR' or 'MRtoCT'.")
    
    # train_dataset = CacheDataset(
    #     data=train_image_paths,
    #     transform=train_transforms,
    #     cache_rate=dataset_option.train.cache_rate,
    #     num_workers=dataset_option.train.cacheDataset_num_workers,
    # )
    train_dataset = Dataset(
        data=train_image_paths,
        transform=train_transforms,
    )

    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=dataset_option.train.batch_size,
        shuffle=True,
        num_workers=dataset_option.train.dataLoader_num_workers,
        pin_memory=True,
        persistent_workers=True
    )

    # val_dataset = CacheDataset(
    #     data=val_image_paths,
    #     transform=validation_transforms,
    #     cache_rate=dataset_option.train.cache_rate,
    #     num_workers=dataset_option.train.cacheDataset_num_workers,
    # )
    val_dataset = Dataset(
            data=val_image_paths,
            transform=validation_transforms,)

    
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=dataset_option.validation.batch_size,
        shuffle=False,
        num_workers=dataset_option.train.dataLoader_num_workers,
        pin_memory=False,
    )




    model = Pix2Pix(opt)

    print(f"Numero di file per training : {train_dataset.__len__()}")
    print(f"Numero di file per validazione : {val_dataset.__len__()}")


    #### if i set the --parellelize_gpus flag, I will use all the available GPUs indipendently from the --gpu_id flag.
    if parallelize_gpus :
        from lightning.pytorch.strategies import DDPStrategy
        strategy = DDPStrategy(find_unused_parameters=True)
        accelerator_config = "gpu"
        # strategy = "ddp_spawn"
        devices_config = -1
    elif gpu_id >= 0:
        accelerator_config = "gpu"
        strategy = "auto"
        devices_config = [gpu_id]
    elif gpu_id == -1:
        accelerator_config = "cpu"
        devices_config = "auto" ### use automatic optimization for CPU training
        strategy = "auto"

    else :
        raise ValueError("Invalid GPU ID specified. Use a non-negative integer for a specific GPU or -1 for CPU or flag --parallelize_gpus to use all available GPUs.")

    checkpoint_callback_ssim = ModelCheckpoint(
        monitor='validation/ssim',        # Monitora la metrica SSIM
        filename=f'best',
        auto_insert_metric_name = True, 
        save_top_k=1,          # Salva i 3 migliori modelli basati 
        mode='max',            
        every_n_epochs=1,      
        save_on_train_epoch_end=False  # Assicura che venga valutato alla fine dell'epoca di validazione
    )

    checkpoint_callback_every_epoch = ModelCheckpoint(
        filename='last-epoch', # Nome del file solo con l'epoca
        save_top_k=-1,         # Salva tutti i checkpoint
        every_n_epochs=1,      # Salva ogni epoca
        save_on_train_epoch_end=True, # Salva alla fine di ogni epoca di training
        train_time_interval=None # Non salvare basandosi sul tempo
        )

    log_save_dir = os.path.join(opt.logging.save_dir,opt.training.direction)
    logger_callback = TensorBoardLogger(save_dir = log_save_dir,
                                        version  = opt.experiment_number)

    trainer = pl.Trainer(
        accelerator=accelerator_config,
        devices=devices_config,
        strategy = strategy , 
        max_epochs=opt.training.epochs,  
        check_val_every_n_epoch=5,
        callbacks=[
            checkpoint_callback_ssim,
            checkpoint_callback_every_epoch],
        logger = logger_callback
    )

    if start_after_interuption:
        try : 
            ckpt_path = ckpt_path = os.path.join(opt.logging.save_dir,opt.training.direction,f"version_{str(opt.experiment_number)}","checkpoints","last-epoch.ckpt")
            print(f"Resuming training from checkpoint: {ckpt_path}")
            trainer.fit(model, train_dataloader, val_dataloader,ckpt_path=ckpt_path)
        except Exception as e:
            print(f"Error resuming training: {e}")
    else:
        print("Starting training from scratch.")
        trainer.fit(model, train_dataloader, val_dataloader)




    if training_only == False:
        print("Starting testing phase...")
        if training_option.direction == "CTtoMR" : 
            test_image_paths = CT2MRfindImages(dataset_option.basic_information,True)
        elif training_option.direction == "MRtoCT" : 
            test_image_paths = MR2CTfindImages(dataset_option.basic_information,True)
        else :
            raise ValueError("Invalid direction specified. Use 'CTtoMR' or 'MRtoCT'.")
        
        
        test_dataset = CacheDataset(
        data=test_image_paths,
        transform=validation_transforms,
        cache_rate=dataset_option.train.cache_rate,
        num_workers=dataset_option.train.cacheDataset_num_workers)
        
        
        test_dataloader = DataLoader(
        val_dataset,
        batch_size=dataset_option.validation.batch_size,
        shuffle=False,
        num_workers=dataset_option.train.dataLoader_num_workers,
        pin_memory=False )
        
        trainer.test(model, dataloaders=test_dataloader, ckpt_path="best")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model with specified configuration.")
    parser.add_argument(
        "--config_file",
        type=str,
        default="options.yaml",
        help="Path to the training configuration file."
    )

    parser.add_argument(
        "--gpu_id",
        type=int,
        default=-1,
        help="Specify a single GPU ID to use (e.g., 0).Using -1 will use a cpu."
    )
    parser.add_argument(
        "--parallelize_gpus",
        action="store_true",
        help="Use all available GPUs for training. Ignored if --gpu_id is set."
    )

    parser.add_argument(
        "--start_after_interuption",
        action="store_true",
        help="Starting epoch for training. Using by default, last epoch"
    )
    parser.add_argument(
        "--training_only",
        action="store_false",
        help="Test the model on testing dataset"
    )
    args = parser.parse_args()

    print(args.start_after_interuption)
    train_pix2pix(config_file=args.config_file, 
                    gpu_id=args.gpu_id, 
                    parallelize_gpus= args.parallelize_gpus, 
                    start_after_interuption= args.start_after_interuption, 
                    training_only=args.training_only)
