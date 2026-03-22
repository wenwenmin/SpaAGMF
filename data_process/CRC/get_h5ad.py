import json
from pathlib import Path
import numpy as np
import pandas as pd
import scipy
from PIL import Image
from anndata import AnnData
import scanpy as sc
from pretrained_model_uni.uni2 import UNI2Extractor

Image.MAX_IMAGE_PIXELS = None

def gene_selection(adata_list, save_dir, n_genes=3000):
    """
    Selection the gene in the all adata

    parameters:
        adata_list (list of AnnData)
    """

    # 1. Get the common gene
    adata_list[0].var_names_make_unique()
    common_genes = set(adata_list[0].var_names)
    for adata in adata_list[1:]:
        adata.var_names_make_unique()
        common_genes &= set(adata.var_names)
    common_genes = list(common_genes)

    # 2. Normalization
    adata_list = [adata[:, common_genes] for adata in adata_list]
    for adata in adata_list:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

    # 3. Get the HVG
    sc.pp.highly_variable_genes(adata_list[0], flavor="seurat_v3", n_top_genes=n_genes, subset=True)
    hvgs = adata_list[0].var_names
    adata_list = [adata[:, hvgs] for adata in adata_list]

    # 4. save
    for adata in adata_list:
        adata = adata.copy()
        if scipy.sparse.issparse(adata.X):
            adata.X = adata.X.toarray()
        name = adata.uns['name']
        save_path = save_dir / f"{name}.h5ad"
        adata.write_h5ad(save_path)
        print(f"Saved {name} shape: {adata.shape}, patch: {adata.obsm['patch'].shape}, cancer: {adata.obs['cancer'].mean() * 100:.1f}%")



def get_patch(image, adata, patch_size=224):
    """
    Extract patches from the image based on the spatial coordinates in adata.

    Parameters:
        image (np.array): High-resolution image (RGB format).
        adata (AnnData): Annotated data object containing spatial coordinates.
        patch_size (int): The size of the patch to extract (default: 224).
    """
    # 1. Get the spatial coordinates (pixel_x, pixel_y) from adata
    spatial_coords = adata.obsm['spatial']

    patches = []
    # 2. Process each patch in a loop
    for coord in spatial_coords:
        # 1. Extract the pixel coordinates (pixel_x, pixel_y)
        pixel_x, pixel_y = coord

        # 2. Calculate the coordinates of the patch
        half_patch = int((patch_size + 1) // 2)
        top_left_x = int(pixel_x) - half_patch
        top_left_y = int(pixel_y) - half_patch
        bottom_right_x = int(pixel_x) + half_patch
        bottom_right_y = int(pixel_y) + half_patch

        patch = image[top_left_y:bottom_right_y, top_left_x:bottom_right_x]

        # 3. Resize the patch to the desired patch size
        if top_left_x < 0 or top_left_y < 0 or bottom_right_x > image.shape[1] or bottom_right_y > image.shape[0]:
            patch_resized = Image.fromarray(patch).resize((patch_size, patch_size), Image.ANTIALIAS)
            patch = np.array(patch_resized)


        # 5. Store the patch
        patches.append(patch)

    # 3. Add the patches as a new observation in adata
    return patches


def process_and_save_to_h5ad(data_dir, label_dir, name, uni):
    # 1. Define the directory of gene express matrix data
    gene_path = data_dir / name / 'filtered_feature_bc_matrix/'
    spatial_path = data_dir / name / 'spatial/tissue_positions_list.csv'
    image_path = data_dir / name / 'spatial/tissue_hires_image.png'
    label_path = label_dir / f'{name}.csv'
    scale_path =  data_dir / name  / 'spatial/scalefactors_json.json'

    # 2. load the adata
    adata = sc.read_10x_mtx(
        gene_path,
        var_names='gene_symbols',
    )

    # 3. load the spatial
    spatial = pd.read_csv(spatial_path, header=None)
    spatial.columns = ["barcode", "in_tissue", "array_y", "array_x", "pxl_y", "pxl_x"]
    spatial_aligned = spatial[spatial['in_tissue'] == 1]   # select the valid pos
    spatial_aligned = spatial_aligned.set_index('barcode')
    spatial_aligned = spatial_aligned.loc[adata.obs_names]  # aligned with the adata

    # 4. load the scale
    with open(scale_path, 'r') as f:
        sf = json.load(f)
    spot_diameter = sf['spot_diameter_fullres']  # 17
    scale = sf['tissue_hires_scalef']  # 0.95

    # 5. load the cancer label
    label = pd.read_csv(label_path)
    label['label'] = label.iloc[:, 1].str.contains('tumor', case=False, na=False).astype(int)
    label.set_index('Barcode', inplace=True)
    # 6. store relevant data
    adata.uns["spot_diameter"] = spot_diameter * scale
    adata.obsm['spatial'] = (spatial_aligned[['pxl_x', 'pxl_y']] * scale).round().astype(int).values
    adata.obs['cancer'] = label['label']

    # 7. Lode the corresponding image
    image_hires = np.array(Image.open(image_path).convert("RGB"))
    patch = get_patch(image_hires, adata, adata.uns["spot_diameter"])

    # 8. Extract patch features by uni2
    patch_embedding = uni.extract(patch)
    adata.obsm["patch"] = patch_embedding

    # 9. named
    adata.uns['name'] = name

    return adata



if __name__ == '__main__':
    # 1. Define the data directory
    data_dir = Path('xxx') # Modify it to your CRC data directory
    label_dir = Path('xxx') # Modify it to your CRC_annotations directory (Pathology_SpotAnnotations)

    # 2. Define the save directory
    save_dir = Path(__file__).parents[2] / 'data/CRC/'
    save_dir.mkdir(parents=True, exist_ok=True)

    # 3. Define the slice names used in this dataset
    names = [p.name for p in data_dir.iterdir() if p.is_dir() and p.name.startswith("CRC")]
    exclude = {"CRC_F1", "CRC_F2"}
    names = [n for n in names if n not in exclude]

    names.sort()  # [CRC_A1, CRC_A2, CRC_B1...]

    uni = UNI2Extractor()
    # 4. Process each slice in a loop
    adata_list = []
    for name in names:
        adata = process_and_save_to_h5ad(data_dir, label_dir, name, uni)
        adata_list.append(adata)

    # gene select
    gene_selection(adata_list, save_dir, n_genes=3000)
