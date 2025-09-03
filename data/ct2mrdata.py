import os 


def findImagePath(main_path,selected_dataset,test_data = False) : 
    """
        find ct2mr images. The images are organized as following :

            sub-xxx-ct.nii.
            sub-xxx-t1w.nii.gz
            processed 
                MR               
                    MRonTemplate
                CT
                    CTonMR
                    CTonTemplate

    Args:
        main_path (str): _description_
        selected_dataset(list): dataset to no considering 
    Returns:
        dataset_path(list) : list of dictionary organized as following  :
                             "ct" : 
                             "mr" : 
                             "subject":
                             "dataset"
     """
    
    data = []
    for dataset in selected_dataset:
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
            subject_path = os.path.join(dataset_path, subject, "processed")
            mr_dir = os.path.join(subject_path, "MR", "MRonTemplate")
            ct_dir = os.path.join(subject_path, "CT", "CTonTemplate")

            try:
                mr_image_path = next(
                    os.path.join(mr_dir, image)
                    for image in os.listdir(mr_dir)
                    if "mrTemplateSpaceNormalized" in image
                )
            except (StopIteration, FileNotFoundError):
                print(f"Warning: File MR mancante per il soggetto {subject} nel dataset {dataset}.")
                mr_image_path = None

            try:
                ct_image_path = next(
                    os.path.join(ct_dir, image)
                    for image in os.listdir(ct_dir)
                    if "ctTemplatespace" in image
                )
            except (StopIteration, FileNotFoundError):
                print(f"Warning: File CT mancante per il soggetto {subject} nel dataset {dataset}.")
                ct_image_path = None
            try:
                mask_path = mr_image_path.replace(main, "work/data/SegmentationDataset") ### non lo sono riuscito ad inserire nei file processati
                tissue_mask_path = next(
                    os.path.join(mask_path, image)
                    for image in os.listdir(ct_dir)
                    if "tissueSegmentationTemplatespace" in image
                )
            except (StopIteration, FileNotFoundError):
                print(f"Warning: Maschera dei tusseti mancante del {subject} nel dataset {dataset}.")
                tissue_mask_path = None


            if mr_image_path and ct_image_path and tissue_mask_path:
                data.append({
                    "ct": ct_image_path,
                    "mr": mr_image_path,
                    "tissue_mask": tissue_mask_path,
                    "subject": subject,
                    "dataset": dataset
                })
            else:
                missing_modalities = []
                if not mr_image_path:
                    missing_modalities.append("MR")
                if not ct_image_path:
                    missing_modalities.append("CT")
                print(f"Warning: Modalità mancanti per il soggetto {subject} nel dataset {dataset}: {', '.join(missing_modalities)}")




    return data