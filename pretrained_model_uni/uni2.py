import os
import timm
import torch
from PIL import Image
from huggingface_hub import login
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class UNIDataset(Dataset):
    def __init__(self, patch_list, transform):
        """
        Dataset class for handling patches of images for feature extraction.

        Parameters:
            patch_list: List of image patches (numpy arrays)
            transform: Transformation to apply to each image patch
        """
        super().__init__()
        self.patch_list = patch_list
        self.transform = transform

    def __len__(self):
        return len(self.patch_list)

    def __getitem__(self, idx):
        """
        Get a single patch and apply the transformation.

        Parameters:
            idx: Index of the patch in the list
        """
        pil_image = Image.fromarray(self.patch_list[idx].astype('uint8'))
        image = self.transform(pil_image)
        return image


class UNI2Extractor:
    def __init__(self, batch_size=256, device=None):
        """
        Initialize the UNIExtractor class with necessary configurations.

        Parameters:
            batch_size: Number of patches to process per batch
            device: Device to run the models on (CPU or CUDA)
        """
        self.batch_size = batch_size
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Login to the Hugging Face Hub (necessary to access pre-trained models)
        login('Your key')

        local_dir = '../../pretrained_model_uni/uni2/'
        os.makedirs(local_dir, exist_ok=True)  # create directory if it does not exist
        timm_kwargs = {
            'model_name': 'vit_giant_patch14_224',
            'img_size': 224,
            'patch_size': 14,
            'depth': 24,
            'num_heads': 24,
            'init_values': 1e-5,
            'embed_dim': 1536,
            'mlp_ratio': 2.66667 * 2,
            'num_classes': 0,
            'no_embed_class': True,
            'mlp_layer': timm.layers.SwiGLUPacked,
            'act_layer': torch.nn.SiLU,
            'reg_tokens': 8,
            'dynamic_img_size': True
        }
        self.model = timm.create_model(
            pretrained=False, **timm_kwargs
        )
        self.model.load_state_dict(torch.load(os.path.join(local_dir, "pytorch_model.bin"), map_location="cpu"), strict=True)
        self.transform = transforms.Compose(
            [
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
        self.model.eval()

    def extract(self, patch):
        """
        Use the pre-trained models UNI2 to extract features from a batch of image patches.

        Parameters:
            patch: List of image patches (each patch is a 3D numpy array)
        """
        self.model.eval()  # Set the models to evaluation mode
        self.model.to(self.device)

        # Create a dataset and dataloader for the image patches
        dataset = UNIDataset(patch, self.transform)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)

        features = []
        with torch.inference_mode():  # Disable gradient tracking during inference
            for images in dataloader:
                images = images.to(self.device)
                output = self.model.forward_features(images)[:, :9, :]
                features.append(output.cpu())

        # Concatenate all extracted features into a single tensor
        features = torch.cat(features, dim=0)
        features = features.to(torch.float16).cpu().numpy()
        return features


