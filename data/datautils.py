import numpy as np
from monai import transforms
import os

def get_window_viewing_value(view_window) : 
    """    
    Get CT window/level values for different viewing presets.
    This function returns the window width and window level (also known as window center) 
    values used for CT image windowing based on common radiological presets.
    Common presets include:
    - brain: Standard brain viewing (width=80, level=40)
    - stroke_1: Acute stroke protocol 1 (width=8, level=32) 
    - stroke_2: Acute stroke protocol 2 (width=40, level=40)
    - soft_tissue: Soft tissue viewing (width=350, level=40)
    - bone: Bone viewing (width=1800, level=400)
    Args:
        view_window (str): The name of the window preset to use. Must be one of:
                          'brain', 'stroke_1', 'stroke_2', 'soft_tissue', 'bone'
        tuple: A tuple containing (window_width, window_level) values for the specified preset
        ValueError: If the specified view_window is not a valid preset name
    References:
        - https://radiopaedia.org/articles/windowing-ct
        - https://www.radnote.it/tc-cranio/#Anatomia_TC_encefalo


    Args:
        view_window (_type_): _description_

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    window = {
        "brain" : (80, 40) , 
        "stroke_1" : (8, 32) ,
        "stroke_2" : (40, 40) ,
        "soft_tissue" : (350, 40) ,
        "bone" : (1800,400) }
    
    if view_window not in window:
        raise ValueError(f"Invalid window preset: {view_window}. Valid options are: {', '.join(window.keys())}")
        
    return window[view_window]

def crop_using_boundinx_box(image, x_min, x_max, y_min, y_max):
    """
    Ritaglia l'area definita dalla bounding box e applica zero padding se necessario.

    :param image: Array NumPy 2D originale
    :param x_min, x_max, y_min, y_max: Coordinate della bounding box
    :return: Immagine ritagliata con padding
    """

    cropped = image[y_min:y_max+1, x_min:x_max+1]
    target_height = y_max - y_min + 1
    target_width = x_max - x_min + 1
    padded_image = np.zeros((target_height, target_width), dtype=image.dtype)
    h, w = cropped.shape
    padded_image[:h, :w] = cropped

    return padded_image


def normalize_image(image_array, method='minmax', mask=None):
    """
    Normalize image values using different methods.
    
    Args:
        image_array: Input image array
        method: 'minmax' for [0,1] or 'symmetric' for [-1,1]
        mask: Optional binary mask to use for normalization
    
    Returns:
        Normalized image array
    """
    if mask is not None:
        # Use only masked values for normalization
        values = image_array[mask > 0]
        min_val = values.min()
        max_val = values.max()
    else:
        min_val = image_array.min()
        max_val = image_array.max()
    
    if method == 'minmax':
        normalized = (image_array - min_val) / (max_val - min_val)
    elif method == 'symmetric':
        normalized = 2 * ((image_array - min_val) / (max_val - min_val)) - 1
    else:
        raise ValueError("method must be 'minmax' or 'symmetric'")
    
    return normalized

def window_image(image_array, window_width, window_level, normalize=None, mask=None):
    """
    Apply intensity windowing to the image with optional normalization.
    
    Args:
        image_array: Input image array
        window_level: Center of the window (level)
        window_width: Width of the window
        normalize: None, 'minmax', or 'symmetric'
        mask: Optional binary mask for normalization
    
    Returns:
        Windowed and optionally normalized image array
    """
    window_min = window_level - window_width // 2
    window_max = window_level + window_width // 2
    
    windowed = np.clip(image_array, window_min, window_max)
    
    if normalize:
        windowed = normalize_image(windowed, method=normalize, mask=mask)
        
    return windowed


def SynthRAD2025findImages(opt): 
    """
    Find all images in the given directory and its subdirectories.

    Args:
        opt: Configuration object containing:
            data_root (str): Root directory path containing the images
            anatomical_district_task (str): Selected anatomical district
            conversion_task (str): Type of conversion task:
                - Task 1: MRI-to-sCT generation for MRI-only and MRI-based adaptive radiotherapy
                - Task 2: CBCT-to-sCT generation for CBCT-based adaptive radiotherapy

    Returns:
        dict: List of dictionary containing image file paths with:
            - "A": Path to input image (MRI or CBCT)
            - "B": Path to corresponding CT image
            - "MASK" : path of the imge mask
            - "subject": Subject ID
            
            
    """

    root_dir = os.path.join(opt.data_root,opt.conversion_task,opt.anatomical_district_task)
    image_paths = []
    for subject in os.listdir(root_dir):
        subject_dir = os.path.join(root_dir, subject)
        if os.path.isdir(subject_dir):
            for image_path in os.listdir(subject_dir):
                image_path = os.path.join(subject_dir, image_path)
                if "mr" in image_path:
                    A_path = image_path
                elif "cbct" in image_path:
                    A_path = image_path 
                elif "ct" in image_path:
                    B_path = image_path
                elif "mask" in image_path:
                    MASK_path = image_path
                
            image_paths.append({
                    "A" : A_path , 
                    "B" : B_path,
                    "MASK" : MASK_path,
                    "subject" : subject
                    })
            
    if opt.train_validation_split.enable:
        # Get split parameters
        ratio = opt.train_validation_split.ratio
        shuffle = opt.train_validation_split.shuffle
        seed = opt.train_validation_split.seed

        # Convert to numpy array for easier manipulation
        image_paths = np.array(image_paths)

        # Shuffle if requested
        if shuffle:
            if seed is not None:
                np.random.seed(seed)
            np.random.shuffle(image_paths)

        # Validate ratio
        if not 0 < ratio < 1:
            raise ValueError("Split ratio must be between 0 and 1")

        # Calculate split index and split data
        split_index = int(len(image_paths) * (1 - ratio))
        if split_index == 0 or split_index == len(image_paths):
            raise ValueError(f"Invalid split ratio {ratio} for dataset size {len(image_paths)}")

        val_image_paths = image_paths[:split_index]
        train_image_paths = image_paths[split_index:]

        # Convert back to list and return
        return train_image_paths.tolist(), val_image_paths.tolist()
    else:
        return image_paths

import os
import numpy as np # Assicurati di importare numpy

def CT2MRfindImages(opt, test_data=False):
    """
    Trova le immagini CT e MR per il problema CT2MR.
    Le immagini sono organizzate in una struttura specifica.

    Args:
        opt: Un oggetto con gli attributi `data_root`, `selected_dataset`,
             e potenzialmente `testing_selected_dataset`, `train_validation_split`.
        test_data (bool): Se True, seleziona i dataset specifici per il testing.

    Returns:
        dataset_path (list): Lista di dizionari organizzati come:
                             {"CT": ..., "MR": ..., "tissue_mask": ..., "subject": ..., "dataset": ...}
                             Se è abilitato lo split, ritorna (train_image_paths, val_image_paths)
    """

    main_path = opt.data_root
    selected_dataset = opt.selected_dataset if not test_data else opt.testing_selected_dataset
    print(f"Dataset selezionati: {selected_dataset}") # Migliorato il print per chiarezza

    data = [] # Questa lista raccoglierà tutti i dati di tutti i dataset

    for dataset in selected_dataset:
        print(f"Elaborando dataset: {dataset}") # Migliorato il print per chiarezza
        dataset_path = os.path.join(main_path, dataset)
        if not os.path.isdir(dataset_path):
            print(f"Warning: Il dataset {dataset} non esiste nel percorso {dataset_path}.")
            continue

        try:
            subjects = os.listdir(dataset_path)
        except Exception as e:
            print(f"Errore durante l'accesso ai soggetti nel dataset {dataset}: {e}")
            continue

        for subject in subjects:
            # Controllo aggiunto per escludere file non-directory (es. .DS_Store)
            if not os.path.isdir(os.path.join(dataset_path, subject)):
                continue

            subject_path = os.path.join(dataset_path, subject, "processed")
            mr_dir = os.path.join(subject_path, "MR", "MRonTemplate")
            ct_dir = os.path.join(subject_path, "CT", "CTonTemplate")

            # TODO da modificare: al momento non sono riuscito a copiare nel folder di processing le maschere di segmentazione dei tessuti.
            # Qui la logica per la maschera del tessuto deve essere rivista per puntare alla location corretta
            # Questo è un placeholder, il percorso della maschera deve essere gestito correttamente in base alla tua struttura
            mask_dir_base = "/home/jovyan/work/data/SegmentationDataset"
            # Assumendo che la maschera segua una struttura simile a subject_path
            # Es: /home/jovyan/work/data/SegmentationDataset/DATASET_NAME/SUBJECT_NAME/processed/MR/MRonTemplate/
            mask_dir_subject = os.path.join(mask_dir_base, dataset, subject, "processed", "MR", "MRonTemplate") # Aggiungi la struttura di subdirectory qui
            mask_dir = mask_dir_subject # Usa la directory completa per la maschera

            mr_image_path = None
            ct_image_path = None
            tissue_mask_path = None

            try:
                # Modificato il filtro per essere più robusto (es. se ci sono altri file)
                # Assicurati che "mrTemplateSpaceNormalized" sia sempre nel nome del file MR corretto
                mr_image_files = [f for f in os.listdir(mr_dir) if "mrTemplateSpaceNormalized" in f and f.endswith(('.nii', '.nii.gz'))]
                if mr_image_files:
                    mr_image_path = os.path.join(mr_dir, mr_image_files[0])
                else:
                    print(f"Warning: File MR mancante con 'mrTemplateSpaceNormalized' per il soggetto {subject} nel dataset {dataset}.")
            except (FileNotFoundError, StopIteration):
                print(f"Warning: Directory MR non trovata o vuota per il soggetto {subject} nel dataset {dataset} ({mr_dir}).")


            try:
                # Modificato il filtro per essere più robusto
                # Assicurati che "ctTemplatespace" sia sempre nel nome del file CT corretto
                ct_image_files = [f for f in os.listdir(ct_dir) if "ctTemplatespace" in f and f.endswith(('.nii', '.nii.gz'))]
                if ct_image_files:
                    ct_image_path = os.path.join(ct_dir, ct_image_files[0])
                else:
                    print(f"Warning: File CT mancante con 'ctTemplatespace' per il soggetto {subject} nel dataset {dataset}.")
            except (FileNotFoundError, StopIteration):
                print(f"Warning: Directory CT non trovata o vuota per il soggetto {subject} nel dataset {dataset} ({ct_dir}).")


            try:
                # Modificato il filtro per essere più robusto
                # Assicurati che "tissueSegmentationTemplatespace" sia sempre nel nome del file maschera corretto
                tissue_mask_files = [f for f in os.listdir(mask_dir) if "tissueSegmentationTemplatespace" in f and f.endswith(('.nii', '.nii.gz'))]
                if tissue_mask_files:
                    tissue_mask_path = os.path.join(mask_dir, tissue_mask_files[0])
                else:
                    print(f"Warning: Maschera dei tessuti mancante con 'tissueSegmentationTemplatespace' per il soggetto {subject} nel dataset {dataset} ({mask_dir}).")
            except (FileNotFoundError, StopIteration):
                print(f"Warning: Directory Maschera Tessuti non trovata o vuota per il soggetto {subject} nel dataset {dataset} ({mask_dir}).")


            if mr_image_path and ct_image_path:
                data.append({
                    "CT": ct_image_path,
                    "MR": mr_image_path,
                    "tissue_mask": tissue_mask_path, # Può essere None se non trovata
                    "subject": subject,
                    "dataset": dataset
                })
            else:
                missing_modalities = []
                if not mr_image_path:
                    missing_modalities.append("MR")
                if not ct_image_path:
                    missing_modalities.append("CT")
                print(f"Warning: Coppia CT/MR incompleta per il soggetto {subject} nel dataset {dataset}: {', '.join(missing_modalities)}. Ignorato.")

    # --- Logica di split o ritorno dei dati DA QUI IN POI ---
    # Questa parte deve essere eseguita DOPO che TUTTI i dataset sono stati elaborati.

    if opt.train_validation_split.enable and not test_data:
        # Get split parameters
        ratio = opt.train_validation_split.ratio
        shuffle = opt.train_validation_split.shuffle
        seed = opt.train_validation_split.seed

        data_array = np.array(data, dtype=object) # Converti a numpy array per lo shuffle

        # Shuffle if requested
        if shuffle:
            if seed is not None:
                np.random.seed(seed)
            np.random.shuffle(data_array)

        # Validate ratio
        if not 0 < ratio < 1:
            raise ValueError("Split ratio must be between 0 and 1")

        # Calculate split index and split data
        split_index = int(len(data_array) * (1 - ratio))
        if split_index == 0 or split_index == len(data_array):
            raise ValueError(f"Invalid split ratio {ratio} for dataset size {len(data_array)}")

        val_image_paths = data_array[:split_index].tolist() # Converti di nuovo a lista
        train_image_paths = data_array[split_index:].tolist() # Converti di nuovo a lista

        # Convert back to list and return
        return train_image_paths, val_image_paths
    elif test_data:
        # Ritorna tutti i dati raccolti per il testing
        return data
    else:
        # Nessuno split, ritorna tutti i dati
        return data



def MR2CTfindImages(opt, test_data=False):
    """
    Trova le immagini MR e CT per il problema MR2CT.
    Le immagini sono organizzate come segue:

        sub-xxx-ct.nii.
        sub-xxx-t1w.nii.gz
        processed
            MR
                MRonTemplate (non rilevante per MR2CT diretto, ma potrebbe contenere maschere)
            CT
                CTonMR
                CTonTemplate (non rilevante per MR2CT diretto)

    Args:
        opt: Un oggetto con gli attributi `data_root`, `selected_dataset`,
             e potenzialmente `testing_selected_dataset`, `train_validation_split`.
        test_data (bool): Se True, seleziona i dataset specifici per il testing.

    Returns:
        dataset_path (list): Lista di dizionari organizzati come:
                             {"CT": ..., "MR": ..., "subject": ..., "dataset": ...}
                             Se è abilitato lo split, ritorna (train_image_paths, val_image_paths)
    """

    main_path = opt.data_root
    selected_dataset = opt.selected_dataset if not test_data else opt.testing_selected_dataset
    print(f"Dataset selezionati: {selected_dataset}")
    #preprocessed_version = opt.MR2CTpreprocessed
    preprocessed_version = True
    data = [] 

    for dataset in selected_dataset:
        print(f"Elaborando dataset: {dataset}")
        dataset_path = os.path.join(main_path, dataset)
        if not os.path.isdir(dataset_path):
            print(f"Warning: Il dataset {dataset} non esiste nel percorso {dataset_path}.")
            continue

        try:
            subjects = os.listdir(dataset_path)
        except Exception as e:
            print(f"Errore durante l'accesso ai soggetti nel dataset {dataset}: {e}")
            continue

        for subject in subjects:
            # Controllo aggiunto per escludere file non-directory (es. .DS_Store)
            if not os.path.isdir(os.path.join(dataset_path, subject)):
                continue

            # Per MR2CT, l'MR originale è spesso nella directory principale del soggetto
            if preprocessed_version : 
                mr_dir = os.path.join(dataset_path, subject,"processed","MR")
            else: 
                mr_dir = os.path.join(dataset_path, subject)
            # La CT registrata al MR è nella sottocartella processed/CT/CTonMR
            ct_dir = os.path.join(dataset_path, subject, "processed", "CT", "CTonMR")

            mr_image_path = None
            ct_image_path = None

            try:
                # Cerca l'immagine T1w originale (sub-xxx-t1w.nii.gz)
                # Assicurati che "t1" sia univoco e presente nel nome del file MR desiderato
                if preprocessed_version :
                    mr_image_files = [f for f in os.listdir(mr_dir) if "MRCTmasked" in f and f.endswith(('.nii', '.nii.gz'))]
                else :
                    mr_image_files = [f for f in os.listdir(mr_dir) if "t1" in f and f.endswith(('.nii', '.nii.gz'))]
                if mr_image_files:
                    mr_image_path = os.path.join(mr_dir, mr_image_files[0])
                else:
                    print(f"Warning: File MR (t1w) mancante per il soggetto {subject} nel dataset {dataset}.")
            except FileNotFoundError:
                print(f"Warning: Directory MR non trovata o vuota per il soggetto {subject} nel dataset {dataset} ({mr_dir}).")
            except Exception as e:
                print(f"Errore durante la ricerca del file MR per {subject} nel dataset {dataset}: {e}")


            try:
                # Cerca l'immagine CT nello spazio MR (ctMRspace)
                # Assicurati che "ctMRspace" sia univoco e presente nel nome del file CT desiderato
                if preprocessed_version :
                    ct_image_files = [f for f in os.listdir(ct_dir) if "ctMRspaceMRCTmasked" in f and f.endswith(('.nii', '.nii.gz'))]
                else : 
                    ct_image_files = [f for f in os.listdir(ct_dir) if "ctMRspace" in f and "MRCT" not in f and f.endswith(('.nii', '.nii.gz'))]                    
                if ct_image_files:
                    ct_image_path = os.path.join(ct_dir, ct_image_files[0])
                else:
                    print(f"Warning: File CT (ctMRspace) mancante per il soggetto {subject} nel dataset {dataset}.")
            except FileNotFoundError:
                print(f"Warning: Directory CT non trovata o vuota per il soggetto {subject} nel dataset {dataset} ({ct_dir}).")
            except Exception as e:
                print(f"Errore durante la ricerca del file CT per {subject} nel dataset {dataset}: {e}")

            if mr_image_path and ct_image_path:
                data.append({
                    "CT": ct_image_path,
                    "MR": mr_image_path,
                    "subject": subject,
                    "dataset": dataset
                })
            else:
                missing_modalities = []
                if not mr_image_path:
                    missing_modalities.append("MR")
                if not ct_image_path:
                    missing_modalities.append("CT")
                print(f"Warning: Coppia MR/CT incompleta per il soggetto {subject} nel dataset {dataset}: {', '.join(missing_modalities)}. Ignorato.")

    # --- Logica di split o ritorno dei dati DA QUI IN POI ---
    # Questa parte deve essere eseguita DOPO che TUTTI i dataset sono stati elaborati.

    if opt.train_validation_split.enable and not test_data:
        # Get split parameters
        ratio = opt.train_validation_split.ratio
        shuffle = opt.train_validation_split.shuffle
        seed = opt.train_validation_split.seed

        data_array = np.array(data, dtype=object) # Converti a numpy array per lo shuffle

        # Shuffle if requested
        if shuffle:
            if seed is not None:
                np.random.seed(seed)
            np.random.shuffle(data_array)

        # Validate ratio
        if not 0 < ratio < 1:
            raise ValueError("Split ratio must be between 0 and 1")

        # Calculate split index and split data
        split_index = int(len(data_array) * (1 - ratio))
        if split_index == 0 or split_index == len(data_array):
            raise ValueError(f"Invalid split ratio {ratio} for dataset size {len(data_array)}. Adjust ratio or data size.")

        val_image_paths = data_array[:split_index].tolist() # Converti di nuovo a lista
        train_image_paths = data_array[split_index:].tolist() # Converti di nuovo a lista

        return train_image_paths, val_image_paths
    elif test_data:
        return data
    else:
        return data