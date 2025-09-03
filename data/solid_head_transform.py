import torch
import numpy as np


import cupy
from cupyx.scipy.ndimage import binary_closing as c_binary_closing
from scipy.ndimage import binary_closing

from monai.transforms import (
    MapTransform,
    KeepLargestConnectedComponent,
    FillHoles
)





class GenerateSolidHeadMaskd(MapTransform):
    """
    Genera una maschera 3D della testa combinando CT e MR.
    
    Rileva automaticamente il dispositivo (CPU/GPU) e usa CuPy per l'accelerazione
    se disponibile, altrimenti ripiega su SciPy.
    """
    def __init__(
        self,
        keys: tuple[str, str],
        new_key: str = "head_mask",
        ct_threshold: float = -200.0,
        mr_threshold: float = 50.0,
        closing_radius: int = 2,
        allow_missing_keys: bool = False
    ):
        """
        Args:
            keys: Tuple contenente le due chiavi per le immagini (es. ("CT", "MR")).
            new_key: Chiave per la nuova maschera creata.
            ct_threshold: Soglia in Unità Hounsfield per la CT.
            mr_threshold: Soglia di intensità per la MR.
            closing_radius: Raggio per l'operazione morfologica di chiusura.
        """
        super().__init__(keys, allow_missing_keys)
        self.ct_key, self.mr_key = keys
        self.new_key = new_key
        self.ct_threshold = ct_threshold
        self.mr_threshold = mr_threshold
        
        # L'elemento strutturante (kernel) è un array NumPy.
        # Verrà spostato su GPU solo al momento dell'uso, se necessario.
        kernel_size = closing_radius * 2 + 1
        self.structure = np.ones((kernel_size, kernel_size, kernel_size))

        # Inizializziamo le trasformazioni MONAI che useremo internamente
        self.keep_largest = KeepLargestConnectedComponent(applied_labels=[1])
        self.fill_holes = FillHoles(connectivity = 2)

    def _process_modality(self, img: torch.Tensor, threshold: float) -> torch.Tensor:
        """Esegue il processing su una singola modalità (CT o MR)."""
        # 1. Rileva il dispositivo direttamente dal tensore di input
        device = img.device
        # 2. Crea la maschera binaria iniziale
        mask_3d = (img > threshold).squeeze(0)
        N_ITERATIONS = 3

        if  device.type == 'cuda':
            mask_cupy = cupy.asarray(mask_3d)
            structure_cupy = cupy.asarray(self.structure)
            closed_mask_cupy = c_binary_closing(mask_cupy, structure=structure_cupy,iterations=N_ITERATIONS,brute_force = True)

            processed_mask_3d = torch.as_tensor(closed_mask_cupy, device=device)
        elif device.type == "cpu" : 
            mask_np = mask_3d.cpu().numpy()
            closed_mask_np = binary_closing(mask_np, structure=self.structure,iterations=N_ITERATIONS,brute_force = True)
            processed_mask_3d = torch.from_numpy(closed_mask_np).to(device)
            
        processed_mask = processed_mask_3d.unsqueeze(0).float()
        final_mask = self.keep_largest(processed_mask)
        
        return final_mask

    def __call__(self, data: dict) -> dict:
        d = dict(data)
        ct_img = d[self.ct_key]
        mr_img = d[self.mr_key]

        ct_mask = self._process_modality(ct_img, self.ct_threshold)
        mr_mask = self._process_modality(mr_img, self.mr_threshold)
        
        common_mask = torch.logical_and(ct_mask, mr_mask)

        final_mask = self.fill_holes(common_mask).float()

        d[self.new_key] = final_mask
        return d


class ApplyMaskAsBackgroundd(MapTransform):
    """
    Applica una maschera binaria alle immagini specificate, impostando i voxel
    esterni alla maschera a un valore di background.
    """
    def __init__(
        self,
        keys: list[str],
        mask_key: str,
        background_values: dict[str, int | float],
        allow_missing_keys: bool = False
    ):
        """
        Args:
            keys: Lista delle chiavi delle immagini a cui applicare la maschera.
            mask_key: Chiave della maschera da applicare.
            background_values: Dizionario che mappa ogni chiave immagine al suo 
                               valore di background (es. {"CT": -1024, "MR": 0}).
        """
        super().__init__(keys, allow_missing_keys)
        self.mask_key = mask_key
        self.background_values = background_values

    def __call__(self, data: dict) -> dict:
        d = dict(data)
        mask = d[self.mask_key]

        for key in self.keys:
            if key in d:
                img = d[key]
                bg_value = self.background_values.get(key, 0)
                
                # Applica la maschera. `torch.where` si adatta al dispositivo
                # dell'input. Creiamo il tensore per bg_value sullo stesso dispositivo
                # per evitare errori di mismatch.
                d[key] = torch.where(
                    mask > 0, 
                    img, 
                    torch.tensor(bg_value, dtype=img.dtype, device=img.device)
                )
        return d