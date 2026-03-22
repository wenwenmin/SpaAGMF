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
            i = 1
            print(i)
            i = i + 1
            patch_resized = Image.fromarray(patch).resize(
                (patch_size, patch_size),
                resample = Image.Resampling.LANCZOS
            )
            patch = np.array(patch_resized)


        # 5. Store the patch
        patches.append(patch)

    # 3. Add the patches as a new observation in adata
    return patches


def process_ref(data_dir, name, uni):
    # 1. Define the directory of gene express matrix data
    gene_path = data_dir / f'ST-cnts/{name}.tsv.gz'
    spatial_path = data_dir / f'ST-spotfiles/{name}_selection.tsv'
    image_dir = data_dir / f'ST-imgs/{name[0]}/{name}/'
    image_path = list(image_dir.glob('*.jpg'))[0]
    label_path = data_dir / f'cancer/{name[0]}_label.txt'

    # 2. load the gene matrix
    gene = pd.read_csv(gene_path, sep='\t', header=0, index_col=0) # columns: gene_name  index: xXy

    # 3. load the spatial
    # columns: x y new_x new_y pixel_x pixel_y selected
    spatial = pd.read_csv(spatial_path, sep='\t', header=0, index_col=None)
    spatial_index = spatial['x'].astype(str) + 'x' + spatial['y'].astype(str)
    spatial.index = spatial_index

    # 4. load the table
    with open(label_path, 'r') as f:
        content = f.read().strip()
    label = list(map(int, content.split()))


    # 6. Create Precessed data
    adata = AnnData(gene)
    gene_index = gene.index
    adata.obsm["spatial"] = spatial.loc[gene_index][['pixel_x', 'pixel_y']].round().astype(int).values
    adata.obs['cancer'] = label
    adata.uns["spot_diameter"] = 224

    # 7. Lode the corresponding image
    image_hires = np.array(Image.open(image_path).convert("RGB"))
    patch = get_patch(image_hires, adata, adata.uns["spot_diameter"])

    # 8. Extract patch features by uni2
    patch_embedding = uni.extract(patch)
    adata.obsm["patch"] = patch_embedding

    # 9. named
    adata.uns['name'] = name[0]

    return adata

def process_tgt(data_dir, name, uni):
    # 1. Define the directory of gene express matrix data
    image_path = data_dir / 'image.jpg'
    label_path = data_dir / 'XeHBC_labels.txt'
    adata_path = data_dir / 'transfered.h5ad'

    adata = sc.read_h5ad(adata_path)
    label = pd.read_csv(label_path, header=None)

    label.index = label.index.astype(str)

    # 6. Create Precessed data
    adata.obs = adata.obs[[]]
    adata.obs['cancer'] = label
    adata.uns["spot_diameter"] = 110

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
    # 1. Define the data directoryref
    ref_dir = Path('xxx') # Modify it to your STHBC data directory (her2st-master/data)
    ref_names = ['A1', 'B1','C1','D1','E1','F1','G2','H1']

    tgt_dir = Path('xxx') # Modify it to your XeHBC data directory
    tgt_names = ['XeHBC']
    # 2. Define the save directory
    save_dir = Path(__file__).parents[2] / 'data/STHBC2XeHBC/'
    save_dir.mkdir(parents=True, exist_ok=True)
    # 3. Define the slice names used in this dataset
    uni = UNI2Extractor()
    # 4. Process each slice in a loop
    adata_list = []
    for name in ref_names:
        adata = process_ref(ref_dir, name, uni)
        adata_list.append(adata)

    for name in tgt_names:
        adata = process_tgt(tgt_dir, name, uni)
        adata_list.append(adata)

    # gene select
    gene_selection(adata_list, save_dir, n_genes=3000)
