import os
import SimpleITK as sitk
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Necessario per projection='3d'
from matplotlib.colors import Normalize  # <-- 1. Importa Normalize
from typing import Union, Tuple, Optional

from monai.transforms import (
    Compose,
    LoadImage,
    EnsureChannelFirst,
    Spacing,
    Orientation,
)

# Importa le funzioni di utility per il windowing dal modulo datautils
from data.datautils import window_image, get_window_viewing_value

def plot_slice_stack(volume_3d: np.ndarray, num_slices: int = 20, output_filename: str = "pila_di_slice.png", stride: int = 4, windowing: Optional[Union[str, Tuple[float, float]]] = None):
    """
    Crea una visualizzazione 3D di una pila di slice da un volume NumPy 3D.
    Questa funzione è CPU-bound a causa di Matplotlib.

    Args:
        volume_3d (np.ndarray): L'array NumPy 3D da visualizzare. Può anche essere un tensore PyTorch.
        num_slices (int): Numero di slice da visualizzare.
        output_filename (str): Nome del file per l'immagine di output.
        stride (int): Il passo per il campionamento della superficie (qualità vs velocità).
        windowing (Optional[Union[str, Tuple[float, float]]]): Applica il windowing.
            Può essere una stringa di preset (es. 'brain', 'bone') o una tupla (width, level).
            Se None, usa la normalizzazione con percentili.
    """
    # Se l'input è un tensore PyTorch, assicurati che sia un array NumPy sulla CPU
    if hasattr(volume_3d, 'cpu'): # Controlla se è un tensore PyTorch
        print("Rilevato tensore PyTorch, spostamento su CPU e conversione in NumPy...")
        volume_3d = volume_3d.cpu().numpy()
    
    # Rimuovi eventuali dimensioni di canale/batch per la visualizzazione
    if volume_3d.ndim > 3:
        volume_3d = volume_3d.squeeze()
        print(f"Array ridotto a 3 dimensioni, nuova shape: {volume_3d.shape}")

    # 3. Selezionare N slice a intervalli regolari FINO ALLA SLICE CENTRALE
    depth = volume_3d.shape[0]
    central_slice_index = depth // 2
    # Seleziona N slice spaziate dall'inizio fino alla slice centrale
    slice_indices = np.linspace(0, central_slice_index, num_slices, dtype=int)
    selected_slices = volume_3d[slice_indices]

    # 4. Impostare la figura e gli assi 3D per il plotting
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # --- WINDOWING E NORMALIZZAZIONE ---
    # Se è specificato un windowing, lo applichiamo. Altrimenti, usiamo la normalizzazione robusta.
    windowing_applied = False
    if windowing:
        print(f"Applicando windowing: {windowing}")
        if isinstance(windowing, str):
            try:
                width, level = get_window_viewing_value(windowing)
                windowing_applied = True
            except ValueError as e:
                print(f"Attenzione: {e}. Ritorno alla normalizzazione con percentili.")
        elif isinstance(windowing, (tuple, list)) and len(windowing) == 2:
            width, level = windowing
            windowing_applied = True
        else:
            print(f"Attenzione: formato 'windowing' non valido. Ritorno alla normalizzazione con percentili.")
        
        if windowing_applied:
             # Applica il windowing per tagliare i valori di intensità
             volume_3d = window_image(volume_3d, window_width=width, window_level=level)
             vmin, vmax = np.min(volume_3d), np.max(volume_3d)
    
    if not windowing_applied:
        print("Nessun windowing specificato, uso la normalizzazione con percentili.")
        vmin = np.percentile(volume_3d, 1)
        vmax = np.percentile(volume_3d, 99)

    if vmin == vmax: # Gestisce il caso di un'immagine con valori costanti
        vmin, vmax = vmin - 1, vmax + 1
    norm = Normalize(vmin=vmin, vmax=vmax)
    # -------------------------

    # 5. Itera e disegna ogni slice come una superficie nel plot 3D
    for i, slice_2d in enumerate(selected_slices):
        height, width = slice_2d.shape
        x, y = np.meshgrid(np.arange(width), np.arange(height))
        z_position = slice_indices[i]
        
        # 3. Applica la normalizzazione PRIMA di passare i dati alla colormap
        facecolors = plt.cm.gray(norm(slice_2d))
        
        # --- CONTROLLO QUALITÀ vs VELOCITÀ ---
        # I parametri 'rstride' e 'cstride' controllano la risoluzione della superficie.
        # - Valori alti (es. 10) sono VELOCI ma l'immagine risulta pixellata.
        # - Valori bassi (es. 1) producono un'immagine DETTAGLIATA ma sono MOLTO LENTI.
        # Un valore intermedio come 4 o 5 è spesso un buon compromesso.
        # Prova a modificare questo valore per trovare il giusto equilibrio.
        ax.plot_surface(x, y, np.full_like(x, z_position),
                        facecolors=facecolors, rstride=stride, cstride=stride, shade=False)

    # 6. Pulisci l'aspetto del grafico e salvalo
    ax.set_axis_off()
    # Imposta l'angolo di visuale per dare l'impressione che la pila "esca" dallo schermo.
    # 'elev=85' guarda la pila quasi dall'alto.
    # 'azim=-90' ruota la vista per allinearla con l'asse.
    ax.view_init(elev=85, azim=-90)
    ax.dist = 7
    plt.savefig(output_filename, dpi=150, transparent=True, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    print(f"✅ Immagine salvata come '{output_filename}'")

def plot_slice_stack_from_file_sitk(image_path: str, num_slices: int = 20, output_filename: str = "pila_di_slice_sitk.png", stride: int = 4, windowing: Optional[Union[str, Tuple[float, float]]] = None):
    """
    Funzione wrapper per caricare un'immagine da file con SimpleITK e visualizzarla.

    Args:
        image_path (str): Percorso del file dell'immagine (es. NIfTI, DICOM).
        num_slices (int): Numero di slice da visualizzare.
        output_filename (str): Nome del file per l'immagine di output.
        stride (int): Il passo per il campionamento della superficie.
        windowing (Optional[Union[str, Tuple[float, float]]]): Preset o valori (width, level) per il windowing.
    """
    try:
        image = sitk.ReadImage(image_path)
        print(f"✅ Immagine caricata con successo da: '{image_path}'")
        volume_3d = sitk.GetArrayFromImage(image)
        # Chiama la funzione di plotting principale
        plot_slice_stack(volume_3d, num_slices, output_filename, stride, windowing)
    except Exception as e:
        print(f"❌ Errore durante il caricamento o la visualizzazione dell'immagine: {e}")
        return


def plot_image_comparison(
    image_paths: list[str],
    image_names: list[str],
    output_path: str,
    views_to_show: list[str] = ["Assiale", "Coronale", "Sagittale"],
    title: str = 'Confronto Immagini Mediche'
):
    """
    Carica un numero arbitrario di immagini NIfTI, le preprocessa e genera
    un'immagine di confronto mostrando le viste anatomiche specificate.

    Args:
        image_paths (list[str]): Lista dei percorsi dei file NIfTI.
        image_names (list[str]): Lista dei nomi da visualizzare per ogni immagine.
        output_path (str): Percorso dove salvare l'immagine PNG generata.
        views_to_show (list[str]): Lista delle viste da mostrare (es. ["Assiale", "Coronale"]).
        title (str): Titolo principale del grafico.
    """
    if len(image_paths) != len(image_names):
        raise ValueError("La lunghezza di 'image_paths' e 'image_names' deve essere la stessa.")

    # 1. Pipeline di trasformazione robusta
    # Orientation assicura che tutte le immagini abbiano lo stesso orientamento (es. RAS)
    # Spacing riscala a uno spacing isotropico per un confronto corretto
    transforms = Compose([
        LoadImage(image_only=True, reader="ITKReader"),
        EnsureChannelFirst(),
        Orientation(axcodes="RAS"),
        Spacing(pixdim=(1.0, 1.0, 1.0), mode='bilinear'),
    ])

    loaded_images = []
    for path in image_paths:
        try:
            img = transforms(path)
            # Rimuove la dimensione del canale, la pipeline assicura che sia la prima
            if img.dim() == 4:
                img = img.squeeze(0)
            loaded_images.append(img)
            print(f"Caricata e processata: {os.path.basename(path)}, shape finale: {img.shape}")
        except Exception as e:
            print(f"Errore durante il caricamento di {path}: {e}")
            return

    # 2. Creare il plot con layout dinamico
    n_rows = len(loaded_images)
    n_cols = len(views_to_show)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows), facecolor='black')
    
    # Se abbiamo una sola riga o colonna, plt.subplots restituisce un array 1D
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    fig.suptitle(title, fontsize=20, color='white')

    # 3. Mappare i nomi delle viste alle dimensioni dell'array
    # Dopo Orientation="RAS": Asse 0 -> Sagittale (X), Asse 1 -> Coronale (Y), Asse 2 -> Assiale (Z)
    view_to_axis = {"Sagittale": 0, "Coronale": 1, "Assiale": 2}

    # 4. Popolare la griglia
    for i, (img_tensor, img_name) in enumerate(zip(loaded_images, image_names)):
        for j, view_name in enumerate(views_to_show):
            ax = axes[i, j]
            
            axis_idx = view_to_axis.get(view_name)
            if axis_idx is None:
                ax.text(0.5, 0.5, f"Vista '{view_name}'\nnon valida", color='red', ha='center', va='center')
                ax.axis('off')
                continue

            # Calcola la slice centrale per la vista corrente
            mid_slice_idx = img_tensor.shape[axis_idx] // 2
            
            # Estrai la slice usando torch.index_select per generalizzare l'indicizzazione
            slice_data = img_tensor.index_select(axis_idx, torch.tensor([mid_slice_idx])).squeeze(axis_idx)
            
            slice_np = slice_data.cpu().numpy()
            
            # Normalizzazione e rotazione per la visualizzazione
            slice_np = np.clip(slice_np, np.percentile(slice_np, 1), np.percentile(slice_np, 99))
            slice_rotated = np.rot90(slice_np, 1)

            ax.imshow(slice_rotated, cmap='gray', aspect='equal')
            ax.axis('off')

            # Aggiungi etichette per righe e colonne
            if j == 0: # Prima colonna
                ax.text(-0.1, 0.5, img_name, transform=ax.transAxes,
                        ha='right', va='center', fontsize=16, color='white', rotation=90)
            if i == 0: # Prima riga
                ax.set_title(view_name, fontsize=16, color='white')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # 5. Salvare l'immagine
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    try:
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1, facecolor='black', dpi=300)
        print(f"Immagine di confronto salvata in: {output_path}")
    except Exception as e:
        print(f"Errore durante il salvataggio dell'immagine: {e}")
    
    plt.close(fig)


def plot_ct_mr_comparison(ct_path: str, mr_path: str, output_path: str):
    """
    Wrapper per la funzione generalizzata per mantenere la retrocompatibilità.
    Carica una CT e una MR e genera un'immagine di confronto.

    Args:
        ct_path (str): Percorso del file NIfTI della CT.
        mr_path (str): Percorso del file NIfTI della MR.
        output_path (str): Percorso dove salvare l'immagine PNG generata.
    """
    plot_image_comparison(
        image_paths=[ct_path, mr_path],
        image_names=["CT", "MR"],
        output_path=output_path,
        title='Paired CT and MR sample'
    )
