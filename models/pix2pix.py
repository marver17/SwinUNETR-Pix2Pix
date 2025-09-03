import lightning as pl
import torch
import torch.nn as nn
import monai.losses
from models.discriminator import define_D
from models.generator import define_G
from models.visualization_utils import create_grid_image
from monai.metrics import FIDMetric, MAEMetric, PSNRMetric, MultiScaleSSIMMetric,SSIMMetric
from monai.inferers import SlidingWindowInferer # SlidingWindowInfererAdapt is not standard, assuming SlidingWindowInferer
from torch.amp import GradScaler # For mixed precision
from monai.transforms import SaveImage # Unused in this class
from models.utils import save_image
from models.utils import GAN_loss_definition, compute_gradient_penalty


class Pix2Pix(pl.LightningModule):
    """
    Implementazione in lightning di una pix2pix, con la possibilità di selezionare
    il modello di generatore e discriminatore
    """
    def __init__(
        self,
        opt,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters() # Saves opt and kwargs to self.hparams

        self.automatic_optimization = False   # Important: This property activates manual optimization.

        # network
        self.netG = define_G(opt.model.generator)
        self.netD = define_D(opt.model.discriminator)

        self.direction = opt.training.direction
        self.n_epochs = opt.training.epochs # Used for LR scheduler

        ### loss options definition

        ## gan loss
        self.gan_mode = opt.training.losses.gan_mode
        # Initialize GAN loss criterion using the utility function
        # This function should return a callable loss module (e.g., nn.BCELoss, or a custom WGAN loss)
        self.criterionGAN = GAN_loss_definition(self.gan_mode) # Ensure this is correctly defined and moved to device if needed

        self.lambda_gp = opt.training.lambda_gp if self.gan_mode == 'wgan-gp' else 0.0

        ### other loss
        self.lambda_L1 = opt.training.losses.lambda_L1
        self.lambda_perceptual = opt.training.losses.lambda_perceptual
        self.lambda_structural = opt.training.losses.lambda_structural

        #### loss instruction

        ### define structural similarity loss
        if self.lambda_structural > 0 :
            self.ssim_loss = monai.losses.ssim_loss.SSIMLoss( # MONAI's SSIMLoss expects inputs in range [0, 1] or [-1, 1]
                spatial_dims=3, # Assumes 3D images
                data_range=1.0  # Ensure this matches your image normalization (e.g., if images are [-1, 1], data_range should be 2.0)
            )
        else:
            self.ssim_loss = None # Explicitly set to None if not used

        self.perceptual_loss = None
        #### define reconstruction loss
        self.L1_loss = nn.L1Loss() if self.lambda_L1 > 0 else None

        if self.gan_mode == 'wgan-gp':
            # For WGAN-GP, we need to compute gradient penalty, so we define itambda_gp') else 10.0
            self.loss_D_fake = 0.0
            self.loss_D_real = 0.0
        

        ### optimizer parameters
        self.lr = opt.training.optimizer.learning_rate
        self.b1 = opt.training.optimizer.b1
        self.b2 = opt.training.optimizer.b2

        ### logging
        self.log_interval = opt.logging.log_interval

        infer_roi_size = opt.validation.roi_size if hasattr(opt.data.validation, 'infer_roi_size') else (128, 128, 128)
        ### validations
        self.sw_inferer = SlidingWindowInferer(
            roi_size= infer_roi_size, 
            sw_batch_size=1, 
            overlap=0.5,     
            mode="gaussian", 
            progress=False
        )


        ### metrics
        self.mae_metric = MAEMetric()
        self.psnr_metric = PSNRMetric(max_val=1.0) # Ensure max_val matches your image data range
        self.ssim_metric = MultiScaleSSIMMetric(spatial_dims=3, data_range=1.0) # Ensure data_range matches


        #### scaler for mixed precision training
        self.scaler_g = GradScaler()
        self.scaler_d = GradScaler()


        #### post_trans_saver
        self.output_dir =  opt.testing.saving_direction if opt.testing.saving_direction  is not None else "outputs" # Provide a default

    # The adversarial_loss function was commented out.
    # It's assumed that GAN_loss_definition in __init__ (self.criterionGAN) handles this.
    # If self.gan_mode == 'wgan-gp', self.criterionGAN should not expect target_is_real,
    # and the loss calculation in backward_D/G for WGAN-GP will be different.
    # The current structure of calling self.criterionGAN(output, True/False) is for vanilla/LSGAN.

    def setup(self, stage:str):
        if stage == "fit" or stage == "validate" or stage == "test":
            if self.lambda_perceptual > 0 and self.perceptual_loss is None:
                self.perceptual_loss = monai.losses.PerceptualLoss(
                    spatial_dims=3,
                    network_type="medicalnet_resnet50_23datasets",
                    is_fake_3d=False
                ).to(self.device) # Ensure it's on the correct device for this process




    def gradient_loss(self, y_hat, y, tissueMask):
        """
        Use this loss to emphasize the intensity variation between gray matter, white matter and csf
        in the image.
        This loss is used to force the generator to learn the intensity variation in the image.
        NOTE: This loss is defined but not currently used in the training loop.

        Args:
            y_hat (torch.Tensor): Generated image.
            y (torch.Tensor): Ground truth image.
            tissueMask (torch.Tensor): Mask to apply gradients.
        """
        from monai.transforms import SobelGradients # Consider initializing SobelGradients in __init__ if used frequently
        sobel = SobelGradients()
        y_hat_grad = sobel(y_hat) * tissueMask
        y_grad = sobel(y) * tissueMask
        return torch.nn.functional.l1_loss(y_hat_grad, y_grad)


    def set_input(self, input_batch): # Renamed for clarity
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input_batch (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        AtoB = self.direction == 'CTtoMR' # Simplified boolean assignment

        # Ensure inputs are moved to the correct device. self.device is managed by Lightning.
        self.real_A = input_batch['CT' if AtoB else 'MR'].to(self.device)
        self.real_B = input_batch['MR' if AtoB else 'CT'].to(self.device)
        self.mask = input_batch['MASK'].to(self.device) if 'MASK' in input_batch else None
        self.patient_information = {"subject": input_batch['subject'], "dataset": input_batch['dataset']}

    def infer(self, x, step):
        """
        Esegue l'inferenza con il generatore.
        - In training: inferenza diretta (forward classico)
        - In validazione/test: sliding window inferer su tutta l'immagine
        """
        if step == "training":
            return self.netG(x)
        elif step == "validation" or step == "test": # Added "test" step
            return self.sw_inferer(inputs=x, network=self.netG)
        else:
            raise ValueError(f"Unknown inference step: {step}")

    def backward_G(self): # optimizer_g removed as it's handled by manual_optimization_step
        """Calculate GAN and L1 loss for the generator"""
        # G(A) should fool the discriminator
        # self.fake_B is already computed in training_step and assigned
        fake_AB = torch.cat((self.real_A, self.fake_B), 1) # Create a combined image for the discriminator
        pred_fake = self.netD(fake_AB) # Discriminator's prediction on the fake image

        # Calculate GAN loss for generator
        # For non-WGAN modes (vanilla, lsgan), D aims to identify fakes as 0, G aims for D(fake) to be 1.
        # For WGAN, loss_G_GAN = -torch.mean(pred_fake).
        if self.gan_mode in ["wgan", 'wgan-gp']:
            self.loss_G_GAN = -torch.mean(pred_fake)
        elif self.gan_mode == "hinge":
            # Target is real (True) because generator wants discriminator to think fake_B is real
            self.loss_G_GAN = self.criterionGAN(pred_fake, True, is_discriminator=False) # Pass is_discriminator=False for generator if criterionGAN needs it
        elif self.gan_mode == "gan":
            # For vanilla GAN, we use BCE loss
            self.loss_G_GAN = self.criterionGAN(pred_fake, True)
            if self.loss_G_GAN.item() < 0:
                print(f"ATTENZIONE: g_loss_GAN negativa rilevata con gan_mode='{self.gan_mode}'!")
                print(f"  loss_G_GAN: {self.loss_G_GAN.item()}")
                print(f"  pred_fake (logits) stats: min={pred_fake.min().item()}, max={pred_fake.max().item()}, mean={pred_fake.mean().item()}")
                print(f"  pred_fake has NaN: {torch.isnan(pred_fake).any().item()}")
                print(f"  pred_fake has Inf: {torch.isinf(pred_fake).any().item()}")
                # Verifica i label usati internamente da criterionGAN
                # (potrebbe essere necessario accedere a _buffers o simili se non sono proprietà dirette)
                # print(f"  criterionGAN.real_label: {self.criterionGAN.real_label.item()}")
            
        # Reconstruction loss (L1)
        self.loss_G_L1 = 0.0
        if self.L1_loss is not None and self.lambda_L1 > 0: # Check if L1_loss module exists
            self.loss_G_L1 = self.L1_loss(self.fake_B, self.real_B) * self.lambda_L1

        # Perceptual loss
        self.loss_G_perceptual = 0.0
        if self.perceptual_loss is not None and self.lambda_perceptual > 0:
            # Ensure perceptual loss is on the same device
            # self._perceptual_loss.to(self.fake_B.device) # MONAI PerceptualLoss should handle device internally or be on device from init
            self.loss_G_perceptual = self.perceptual_loss(self.fake_B.float(), self.real_B.float()) * self.lambda_perceptual

        # Structural similarity loss (SSIM)
        self.loss_G_structural = 0.0
        if self.ssim_loss is not None and self.lambda_structural > 0:
            # SSIM loss is 1 - SSIM_index. To minimize loss, maximize SSIM_index.
            # MONAI's SSIMLoss already returns 1-SSIM, so direct use is fine.
            self.loss_G_structural = self.ssim_loss(self.fake_B, self.real_B) * self.lambda_structural # Ensure data range is correct

        # Total generator loss
        self.loss_G = self.loss_G_GAN + self.loss_G_L1 + self.loss_G_perceptual + self.loss_G_structural



    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        
        # Assicurati che fake_B sia completamente detached dai gradienti del generatore
        fake_B_detached = self.fake_B.detach()
        
        if self.gan_mode == "wgan-gp":
            # WGAN-GP: Loss specifico con gradient penalty
            fake_AB = torch.cat((self.real_A, fake_B_detached), 1)
            pred_fake = self.netD(fake_AB)
            
            real_AB = torch.cat((self.real_A, self.real_B), 1)
            pred_real = self.netD(real_AB)
            
            # WGAN loss: D vuole massimizzare D(real) - D(fake)
            # Per minimizzare, usiamo: -D(real) + D(fake)
            self.loss_D_fake = torch.mean(pred_fake)
            self.loss_D_real = -torch.mean(pred_real)
            
            # Gradient Penalty
            gp_term, _ = compute_gradient_penalty(
                critic=self.netD,
                real_B_samples=self.real_B,
                fake_B_samples=fake_B_detached,
                real_A_samples=self.real_A,
                device=self.real_B.device,
                lambda_gp=self.lambda_gp
            )
            
            self.loss_D = self.loss_D_fake + self.loss_D_real + gp_term
            
        elif self.gan_mode in ["gan", "lsgan"]:
            # Vanilla GAN e LSGAN: usa criterionGAN
            fake_AB = torch.cat((self.real_A, fake_B_detached), 1)
            pred_fake = self.netD(fake_AB)
            
            real_AB = torch.cat((self.real_A, self.real_B), 1)
            pred_real = self.netD(real_AB)
            
            # Discriminator loss: fake dovrebbe essere 0, real dovrebbe essere 1
            loss_D_fake = self.criterionGAN(pred_fake, False, is_discriminator=True)
            loss_D_real = self.criterionGAN(pred_real, True, is_discriminator=True)
            
            self.loss_D = (loss_D_fake + loss_D_real) * 0.5
            
            # Debug per valori anomali
            if self.loss_D.item() < 0:
                print(f"WARNING: Negative D loss in {self.gan_mode} mode!")
                print(f"  loss_D_fake: {loss_D_fake.item()}")
                print(f"  loss_D_real: {loss_D_real.item()}")
                print(f"  pred_fake stats: min={pred_fake.min()}, max={pred_fake.max()}")
                print(f"  pred_real stats: min={pred_real.min()}, max={pred_real.max()}")
            
        elif self.gan_mode == "hinge":
            # Hinge loss per discriminatore
            fake_AB = torch.cat((self.real_A, fake_B_detached), 1)
            pred_fake = self.netD(fake_AB)
            
            real_AB = torch.cat((self.real_A, self.real_B), 1)
            pred_real = self.netD(real_AB)
            
            # Hinge loss: max(0, 1 - D(real)) + max(0, 1 + D(fake))
            loss_D_real = torch.relu(1.0 - pred_real).mean()
            loss_D_fake = torch.relu(1.0 + pred_fake).mean()
            
            self.loss_D = loss_D_real + loss_D_fake
            
        else:
            raise NotImplementedError(f'Discriminator loss for {self.gan_mode} not implemented')

    def training_step(self, batch, batch_idx):
        """Training step con gestione migliorata"""
        self.set_input(batch)
        optimizer_g, optimizer_d = self.optimizers()

        # Generate fake image once
        self.fake_B = self.infer(self.real_A, "training")

        # ---------------------
        #  Train Discriminator
        # ---------------------
        self.toggle_optimizer(optimizer_d)
        optimizer_d.zero_grad()

        # Calculate D loss
        self.backward_D()
        
        # Controllo per loss anomale
        if torch.isnan(self.loss_D) or torch.isinf(self.loss_D):
            print(f"ERROR: Invalid discriminator loss at batch {batch_idx}")
            print(f"  loss_D: {self.loss_D}")
            # Skip questo batch o usa un loss di fallback
            self.loss_D = torch.tensor(1.0, device=self.device, requires_grad=True)

        # Backward pass per D
        self.manual_backward(self.loss_D)  
        optimizer_d.step()
        self.untoggle_optimizer(optimizer_d)

        # -----------------
        #  Train Generator
        # -----------------
        self.toggle_optimizer(optimizer_g)
        optimizer_g.zero_grad()

        # Calculate G loss
        self.backward_G()
        
        if torch.isnan(self.loss_G) or torch.isinf(self.loss_G):
            print(f"ERROR: Invalid generator loss at batch {batch_idx}, particoularly : {self.patient_information}")
            print(f"  loss_G: {self.loss_G}")

            self.loss_G = torch.tensor(1.0, device=self.device, requires_grad=True)

        self.manual_backward(self.loss_G)
        optimizer_g.step()
        self.untoggle_optimizer(optimizer_g)

        # Log losses
        self.log("training/g_loss", self.loss_G, prog_bar=True, on_step=True, on_epoch=False, logger=True)
        self.log("training/g_loss_GAN", self.loss_G_GAN, prog_bar=False, on_step=True, on_epoch=False, logger=True)
        self.log("training/g_loss_L1", self.loss_G_L1, prog_bar=False, on_step=True, on_epoch=False, logger=True)

        if self.lambda_perceptual > 0 and self.loss_G_perceptual is not None : # Check if it's computed
            self.log("training/g_loss_perceptual", self.loss_G_perceptual, prog_bar=False, on_step=True, on_epoch=False, logger=True)
        if self.lambda_structural > 0 and self.loss_G_structural is not None: # Check if it's computed
            self.log("training/g_loss_structural", self.loss_G_structural, prog_bar=False, on_step=True, on_epoch=False, logger=True)

        self.log("training/d_loss", self.loss_D, prog_bar=True, on_step=True, on_epoch=False, logger=True)
        if self.gan_mode == 'wgan-gp':
            self.log("training/d_loss_gp", self.loss_D - self.loss_D_fake - self.loss_D_real, prog_bar=False, on_step=True, on_epoch=False, logger=True) # Log GP component

    def validation_step(self, batch, batch_idx):
        torch.cuda.empty_cache() # Be cautious with empty_cache, it can slow things down.
        self.set_input(batch)
        # Use "validation" for inferer_step
        self.generated_imgs_val = self.infer(self.real_A, "validation") # Store with a different name to avoid conflict

        # Masking logic seems commented out, assuming full image comparison for now
        real_B_to_compare = self.real_B
        generated_imgs_to_compare = self.generated_imgs_val


        self.mae_metric(generated_imgs_to_compare, real_B_to_compare)
        self.psnr_metric(generated_imgs_to_compare, real_B_to_compare)
        self.ssim_metric(generated_imgs_to_compare, real_B_to_compare)


        # For visualization during validation (optional)
        if batch_idx == 0: # Log first batch
             grid = create_grid_image(self.real_A, self.real_B, self.generated_imgs_val.detach())
             self.logger.experiment.add_image("validation/images_comparison", grid, self.current_epoch)


    def on_validation_epoch_end(self):
        # Aggregate metrics
        mae = self.mae_metric.aggregate().item()
        psnr = self.psnr_metric.aggregate().item()
        ssim = self.ssim_metric.aggregate().item()

        self.log("validation/mae", mae, prog_bar=True, sync_dist=True)
        self.log("validation/psnr", psnr, prog_bar=True, sync_dist=True)
        self.log("validation/ssim", ssim, prog_bar=True, sync_dist=True)

        # Reset metrics for the next epoch
        self.mae_metric.reset()
        self.psnr_metric.reset()
        self.ssim_metric.reset()


    def configure_optimizers(self):
        opt_g = torch.optim.Adam(self.netG.parameters(),
                                lr=self.lr,
                                betas=(self.b1, self.b2))
        opt_d = torch.optim.Adam(self.netD.parameters(),
                                lr=self.lr,
                                betas=(self.b1, self.b2))

        def lr_lambda(current_epoch):
            if current_epoch < self.n_epochs * 0.5: # Constant LR for the first half
                return 1.0
            else: 
                return (self.n_epochs - current_epoch) / (self.n_epochs - self.n_epochs * 0.5)




        scheduler_g = {
            "scheduler": torch.optim.lr_scheduler.LambdaLR(opt_g, lr_lambda),
            "interval": "epoch", 
            "frequency": 1
        }

        scheduler_d = {
            "scheduler": torch.optim.lr_scheduler.LambdaLR(opt_d, lr_lambda),
            "interval": "epoch",
            "frequency": 1
        }

        return [opt_g, opt_d], [scheduler_g, scheduler_d]

    def test_step(self, batch, batch_idx):
        """Test step to generate images, calculate metrics and save results"""
        self.set_input(batch)
        self.generated_imgs_test = self.infer(self.real_A, "test")

        real_B_to_compare = self.real_B
        generated_imgs_to_compare = self.generated_imgs_test

        self.mae_metric(generated_imgs_to_compare, real_B_to_compare)
        self.psnr_metric(generated_imgs_to_compare, real_B_to_compare)
        self.ssim_metric(generated_imgs_to_compare, real_B_to_compare)

        mae_test = self.mae_metric.aggregate().item()
        psnr_test = self.psnr_metric.aggregate().item()
        ssim_test = self.ssim_metric.aggregate().item()
        
        self.log("test/mae", mae_val, prog_bar=True, on_step=False, on_epoch=True, logger=True)
        self.log("test/psnr", psnr_val, prog_bar=True, on_step=False, on_epoch=True, logger=True)
        self.log("test/ssim", ssim_val, prog_bar=True, on_step=False, on_epoch=True, logger=True)

        # Reset metrics after each test step if aggregating per step, or manage globally if aggregating over epoch
        self.mae_metric.reset()
        self.psnr_metric.reset()
        self.ssim_metric.reset()

        metrics_dict = {
            "mae": mae_test,
            "psnr": psnr_test,
            "ssim": ssim_test
            }

        save_image(
            self.real_A, # Original input
            self.real_B, # Ground truth
            self.generated_imgs_test, # Generated output
            self.patient_information, # Metadata
            self.output_dir,          # Output directory
            metrics=metrics_dict,     # Calculated metrics
            filename_prefix=f"test_batch{batch_idx}" # Example: make filenames unique
        )

    def forward(self, x): # Changed z to x for clarity, as it's typically the input image
        # This forward is used for inference when you call model(input)
        return self.netG(x)

