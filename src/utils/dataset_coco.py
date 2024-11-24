import os
import json
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from pycocotools import mask as coco_mask

class HeartSegmentationDataset(torch.utils.data.Dataset):
    def __init__(self, json_path, images_dir, transform=None):
        self.images_dir = images_dir
        self.transform = transform
        with open(json_path, 'r') as f:
            self.coco_data = json.load(f)
        self.image_info = {img['id']: img for img in self.coco_data['images']}
        self.annotations = self._organize_annotations(self.coco_data['annotations'])

    def _organize_annotations(self, annotations):
        organized_annotations = {}
        for ann in tqdm(annotations, desc="Organize annotations"):
            image_id = ann['image_id']
            if image_id not in organized_annotations:
                organized_annotations[image_id] = []
            organized_annotations[image_id].append(ann)
        return organized_annotations

    def _generate_mask(self, annotations, image_size):
        mask = np.zeros(image_size, dtype=np.uint8)
        for ann in annotations:
            rle = coco_mask.frPyObjects(ann['segmentation'], image_size[0], image_size[1])
            rle = coco_mask.merge(rle)
            mask |= coco_mask.decode(rle).astype(np.uint8)
        return mask

    def __len__(self):
        return len(self.image_info)

    def __getitem__(self, idx):
        # Get image info
        image_id = list(self.image_info.keys())[idx]
        image_info = self.image_info[image_id]
        
        # Load image
        image_path = os.path.join(self.images_dir, image_info['file_name'])
        image = np.array(Image.open(image_path).convert("RGB"))
        
        # Generate mask
        image_size = (image_info['height'], image_info['width'])
        annotations = self.annotations.get(image_id, [])
        mask = self._generate_mask(annotations, image_size)

        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']  # This should now be a tensor thanks to ToTensorV2
            mask = transformed['mask']    # This should now be a tensor thanks to ToTensorV2
            return image, mask.float()  # Ensure mask is float type
        
        # If no transforms, convert to tensor manually (though this shouldn't happen with your setup)
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        mask = torch.from_numpy(mask).float()
        return image, mask