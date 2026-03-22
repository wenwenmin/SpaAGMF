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
        half_patch = patch_size // 2
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


def get_patch_and_save(image, adata, patch_size=224, save_dir="./patches"):
    """
    Extract patches from the image based on spatial coordinates in adata,
    and save each patch as a PNG file.

    Parameters:
        image (np.array): High-resolution image (H x W x 3, RGB format)
        adata (AnnData): Annotated data object containing spatial coordinates
        patch_size (int): Size of the square patch
        save_dir (str or Path): Directory to save patch PNGs

    Returns:
        patches (list of np.array): List of extracted patches
    """
    spatial_coords = adata.obsm['spatial']
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)  # 创建目录

    patches = []
    for idx, coord in enumerate(spatial_coords):
        pixel_x, pixel_y = coord

        half_patch = int(patch_size // 2)
        top_left_x = int(pixel_x) - half_patch
        top_left_y = int(pixel_y) - half_patch
        bottom_right_x = int(pixel_x) + half_patch
        bottom_right_y = int(pixel_y) + half_patch

        patch = image[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
        # 如果裁剪后的 patch 大小不等于 patch_size，进行缩放
        if top_left_x < 0 or top_left_y < 0 or bottom_right_x > image.shape[1] or bottom_right_y > image.shape[0]:
            patch_resized = Image.fromarray(patch).resize((patch_size, patch_size), Image.ANTIALIAS)
            patch = np.array(patch_resized)

        patches.append(patch)

        # 保存为 PNG
        patch_image = Image.fromarray(patch)
        patch_path = save_dir / f"patch_{idx:04d}.png"
        patch_image.save(patch_path)

    return patches

def process_and_save_to_h5ad(data_dir, name, uni):
    # --------------------------------------------------
    # 1. Path definition
    # --------------------------------------------------
    gene_path = data_dir / f"ST-cnts/{name}.tsv.gz"
    spatial_path = data_dir / f"ST-spotfiles/{name}_selection.tsv"
    label_path = data_dir / f"ST-pat/lbl/{name}_labeled_coordinates.tsv"
    image_dir = data_dir / f"ST-imgs/{name[0]}/{name}/"
    image_path = list(image_dir.glob("*.jpg"))[0]

    # --------------------------------------------------
    # 2. Load gene expression (ORDER SOURCE)
    # --------------------------------------------------
    gene = pd.read_csv(
        gene_path,
        sep="\t",
        header=0,
        index_col=0,
    )  # index: "xXy"

    gene_index = gene.index.copy()  # ★ 主顺序，只认这个

    # --------------------------------------------------
    # 3. Load spatial & label tables
    # --------------------------------------------------
    spatial = pd.read_csv(spatial_path, sep="\t")
    label = pd.read_csv(label_path, sep="\t")

    # --------------------------------------------------
    # 4. Keep valid spatial spots
    # --------------------------------------------------
    spatial = spatial.loc[spatial["selected"] == 1].copy()

    spatial.index = (
        spatial["x"].astype(str) + "x" + spatial["y"].astype(str)
    )

    label = label.dropna(subset=["x", "y"]).copy()
    label.index = (
        label["x"].round(0).astype(int).astype(str)
        + "x"
        + label["y"].round(0).astype(int).astype(str)
    )

    # --------------------------------------------------
    # 5. ORDER-SAFE index filtering (NO intersection)
    # --------------------------------------------------
    valid_index = gene_index[
        gene_index.isin(spatial.index)
        & gene_index.isin(label.index)
    ]

    gene = gene.loc[valid_index]
    spatial = spatial.loc[valid_index]
    label = label.loc[valid_index]

    # --------------------------------------------------
    # 6. Cancer label (order already fixed)
    # --------------------------------------------------
    spatial["cancer"] = (
        label["label"]
        .str.contains("cancer", case=False, na=False)
        .astype(int)
    )

    # --------------------------------------------------
    # 7. Create AnnData
    # --------------------------------------------------
    adata = AnnData(X=gene.values)
    adata.obs_names = gene.index
    adata.var_names = gene.columns

    adata.obs["cancer"] = spatial["cancer"].values
    adata.obsm["spatial"] = spatial[["pixel_x", "pixel_y"]].values
    adata.uns["spot_diameter"] = 224
    adata.uns["name"] = name[0]

    # --------------------------------------------------
    # 8. Load image & extract patches
    # --------------------------------------------------
    image_hires = np.array(Image.open(image_path).convert("RGB"))
    patches = get_patch(
        image_hires,
        adata,
        patch_size=adata.uns["spot_diameter"],
    )

    patch_embedding = uni.extract(patches)
    adata.obsm["patch"] = patch_embedding.astype(np.float16)

    # --------------------------------------------------
    # 9. Hard safety check (debug only)
    # --------------------------------------------------
    assert (adata.obs_names == valid_index).all()

    return adata


if __name__ == '__main__':
    # 1. Define the data directory
    data_dir = Path('xxx')  # Modify it to your STHBC data directory (her2st-master/data)

    # 2. Define the save directory
    save_dir = Path(__file__).parents[2] / 'data/STHBC/'
    save_dir.mkdir(parents=True, exist_ok=True)

    # 3. Define the slice names used in this dataset
    names = ['A1', 'B1','C1','D1','E1','F1','G2','H1']

    uni = UNI2Extractor()
    # 4. Process each slice in a loop
    adata_list = []
    for name in names:
        adata = process_and_save_to_h5ad(data_dir, name, uni)
        adata_list.append(adata)

    gene_selection(adata_list, save_dir, n_genes=3000)
