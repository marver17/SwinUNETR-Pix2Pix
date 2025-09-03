import torch
import torchvision
import torch.nn.functional as F


def resize_slice(slice_tensor, th, tw):
          return F.interpolate(slice_tensor, size=(th, tw), mode='bilinear', align_corners=False)


def create_grid_image(real_A, real_B, generated_imgs):
    slice_idx = real_A.shape[2] // 2  # Get middle slice along z-axis

    # Get central slices from both real and generated images
    input_slice = real_A[:, :, slice_idx, :, :]  # Shape: [B, C, H, W]
    generated_slice = generated_imgs[:, :, slice_idx, :, :]
    target_slice = real_B[:, :, slice_idx, :, :]

    # Concatenate along batch dimension for side-by-side visualization
    combined_slices = torch.cat([
        input_slice,
        generated_slice,
        target_slice
    ], dim=0)

    # Create grid for all three views (axial, coronal, sagittal)
    slice_idx_axial = real_A.shape[2] // 2  # Middle slice along z-axis (axial view)
    slice_idx_coronal = real_A.shape[3] // 2  # Middle slice along y-axis (coronal view)
    slice_idx_sagittal = real_A.shape[4] // 2  # Middle slice along x-axis (sagittal view)

    # Extract slices for each view
    axial_slices = torch.cat([
        real_A[:, :, slice_idx_axial, :, :],
        generated_imgs[:, :, slice_idx_axial, :, :],
        real_B[:, :, slice_idx_axial, :, :]
    ], dim=0)

    coronal_slices = torch.cat([
        real_A[:, :, :, slice_idx_coronal, :],
        generated_imgs[:, :, :, slice_idx_coronal, :],
        real_B[:, :, :, slice_idx_coronal, :]
    ], dim=0)

    sagittal_slices = torch.cat([
        real_A[:, :, :, :, slice_idx_sagittal],
        generated_imgs[:, :, :, :, slice_idx_sagittal],
        real_B[:, :, :, :, slice_idx_sagittal]
    ], dim=0)

    # Resize slices
    target_h, target_w = 193, 193

    axial_resized = resize_slice(axial_slices, target_h, target_w)
    coronal_resized = resize_slice(coronal_slices, target_h, target_w)
    sagittal_resized = resize_slice(sagittal_slices, target_h, target_w)

    # Combine resized views
    combined_views = torch.cat([axial_resized, coronal_resized, sagittal_resized], dim=3)

    # Create grid
    grid = torchvision.utils.make_grid(
        combined_views,
        nrow=real_A.shape[0],  # Number of images per row equals batch size
        normalize=True,
        pad_value=1,
        scale_each=True  # Scale each image independently for better visualization
    )

    return grid