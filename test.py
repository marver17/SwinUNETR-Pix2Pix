import argparse
import os
import yaml
import torch
from tqdm import tqdm

# Import MONAI components
from monai.inferers import SlidingWindowInferer
from monai.metrics import MAEMetric, PSNRMetric, MultiScaleSSIMMetric, SSIMMetric
from monai.data import CacheDataset, DataLoader

# Import accelerate
from accelerate import Accelerator

from models.pix2pix import Pix2Pix
from models.utils import save_image
from data.transforms import get_transforms
from data.datautils import CT2MRfindImages,MR2CTfindImages
from utils import util

def set_input(input_batch, direction):
    """Unpack input data from the dataloader and perform necessary pre-processing steps.
    With accelerate, tensors from the dataloader are already on the correct device.

    Parameters:
        input_batch (dict): include the data itself and its metadata information.
        direction (str): The direction of translation, e.g., 'CTtoMR'.

    The option 'direction' can be used to swap images in domain A and domain B.
    """
    AtoB = direction == 'CTtoMR'

    # With accelerate, input_batch tensors are already on the correct device
    # after the dataloader has been prepared.
    real_A = input_batch['CT' if AtoB else 'MR']
    real_B = input_batch['MR' if AtoB else 'CT']
    
    
    mask = input_batch['MASK'] if 'MASK' in input_batch else None

    if mask is not None and torch.numel(mask) == 0: # Check if tensor is empty
        mask = None 

    patient_information = [{"subject": x, "dataset": y} for x, y in zip(input_batch["subject"], input_batch["dataset"])]
    return real_A, real_B, mask, patient_information

def test_model(model, dataloader, opt, accelerator):
    """
    Tests the model using the provided dataloader and calculates metrics.
    Uses accelerate for distributed training setup.
    """
    sw_inferer = SlidingWindowInferer(
        roi_size=(128, 128, 128),
        sw_batch_size=1,
        overlap=0.5,
        mode="gaussian"
    )

    mae_metric = MAEMetric(reduction="mean")
    psnr_metric = PSNRMetric(max_val=1)
    ssim_metric = SSIMMetric(spatial_dims=3)
    mmsim_metric = MultiScaleSSIMMetric(spatial_dims=3)

    model, dataloader = accelerator.prepare(model, dataloader)
    direction = opt.training.direction


    with torch.no_grad():
        model.eval()
        for i, batch in tqdm(enumerate(dataloader), total=len(dataloader), disable=not accelerator.is_main_process):
            input_image,target_image,_,subject_information = set_input(batch,direction)

            if hasattr(model, 'netG'): # Common convention for generator in GANs
                output_image = sw_inferer(input_image, network=model.netG)
            else: # Fallback, assuming the model itself is the network
                output_image = sw_inferer(input_image, network=model)

            output_image = accelerator.gather(output_image)
            target_image = accelerator.gather(target_image)
            
            # Calculate metrics
            mae_metric(output_image, target_image)
            psnr_metric(output_image, target_image)
            ssim_metric(output_image, target_image)
            mmsim_metric(output_image, target_image) # Corrected metric usage

            # Aggregate and reset metrics per batch if you want per-batch metrics,
            # otherwise aggregate at the end of the loop.
            # Your original code aggregated and reset inside the loop, so I'll keep that behavior.
            # If you want overall average metrics, move these lines outside the loop.
            mae = mae_metric.aggregate().item()
            psnr = psnr_metric.aggregate().item()
            ssim = ssim_metric.aggregate().item()
            mmsim = mmsim_metric.aggregate().item()

            metrics = {
                "MAE": mae,
                "PSNR": psnr,
                "SSIM": ssim,
                "MM-SSIM": mmsim
            }

            subject_information = {"subject": batch['subject'][0], "dataset": batch['dataset'][0]}

            if accelerator.is_main_process:
                output_dir = os.path.join(opt.logging.save_dir,opt.training.direction,f"version_{str(opt.experiment_number)}","test_results")
                save_image(input_image[0], target_image[0], output_image[0], subject_information,output_dir,metrics)

            mae_metric.reset()
            psnr_metric.reset()
            ssim_metric.reset()
            mmsim_metric.reset()

        # if accelerator.is_main_process:
        #     final_mae = mae_metric.aggregate().item()
        #     final_ssim = ssim_metric.aggregate().item()
        #     final_msim = mmsim_metric.aggregate().item()
        #     final_psnr = psnr_metric.aggregate().item()
        #     print(f"Overall MAE: {final_mae}")
        #     print(f"Overall M-SSIM: {final_msim}")
        #     print(f"Overall SSIM: {final_ssim}")
        #     print(f"Overall PSNR: {final_psnr}")


def main():
    parser = argparse.ArgumentParser(description="Test Pix2Pix model with multi-GPU support using accelerate.")
    parser.add_argument("--experiment_path", type=str,
                        help="Path to the YAML file containing experiment options.")
    args = parser.parse_args()

    # Initialize accelerator
    accelerator = Accelerator()
    accelerator.print(f"Using device: {accelerator.device}")

    # Load options from YAML file
    experiment_path = args.experiment_path
    if not os.path.exists(experiment_path):
        accelerator.print(f"Errore: Il file delle opzioni '{experiment_path}' non trovato.")
        return

    with open(experiment_path, 'r') as file:
        options = yaml.safe_load(file)

    opt = util.DotDict(options)
    dataset_option = util.DotDict(options['data'])

    # Get test transforms
    test_transforms = get_transforms(dataset_option.validation)

    # Find test images
    test_image_paths = []
    if opt.training.direction == "CTtoMR":
        test_image_paths = CT2MRfindImages(dataset_option.basic_information, True)
    elif opt.training.direction == "MRtoCT":
        test_image_paths = MR2CTfindImages(dataset_option.basic_information, True)
    else:
        accelerator.print(f"Direzione di training non riconosciuta: {opt.training.direction}")
        return

    # Create dataset and dataloader
    test_dataset = CacheDataset(
        data=test_image_paths,
        transform=test_transforms,
        cache_rate=dataset_option.train.cache_rate,
        num_workers=dataset_option.train.cacheDataset_num_workers,
    )

    # DataLoader num_workers can be set higher as accelerate manages processes
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=dataset_option.validation.batch_size,
        shuffle=False, # Important for consistent testing
        num_workers=dataset_option.train.dataLoader_num_workers,
        pin_memory=False, # Accelerate handles pin_memory
    )

    # Load model checkpoint
    ckpt_dir = os.path.join(opt.logging.save_dir, opt.training.direction, f"version_{str(opt.experiment_number)}", "checkpoints")
    ckpt_paths = [os.path.join(ckpt_dir, x) for x in os.listdir(ckpt_dir) if "best" in x]

    if not ckpt_paths:
        accelerator.print(f"Errore: Nessun checkpoint 'best' trovato in {ckpt_dir}.")
        return
    
    ckpt_path = ckpt_paths[0] # Take the first 'best' checkpoint

    accelerator.print(f"Caricamento del modello dal checkpoint: {ckpt_path}")
    model = Pix2Pix.load_from_checkpoint(ckpt_path, opt=opt, strict=False)

    accelerator.print("Avvio del processo di testing...")
    test_model(model, test_dataloader, opt, accelerator)
    accelerator.print("Testing completato.")


if __name__ == "__main__":
    main()