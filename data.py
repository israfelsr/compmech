import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms  
from PIL import Image, ImageDraw, ImageFont



class ConceptAttributesDataset(Dataset):
    """
    PyTorch Dataset for concepts, their attributes, and images.

    For each concept, it returns a transformed image tensor and a 
    multi-hot encoded vector of its attributes.
    """
    def __init__(self, concept_file, attribute_file, image_dir, transform=None):
        """
        Args:
            concept_file (string): Path to the json file with concepts and their attributes.
            attribute_file (string): Path to the json file with attribute taxonomy.
            image_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.image_dir = image_dir
        
        # Load data from files
        with open(concept_file, 'r') as f:
            concept_data = json.load(f)
        with open(attribute_file, 'r') as f:
            attribute_data = json.load(f)

        # Create vocabulary and mappings for attributes
        self.all_attributes = sorted(list(attribute_data.keys()))
        self.attribute_to_idx = {attr: i for i, attr in enumerate(self.all_attributes)}
        self.idx_to_attribute = {i: attr for i, attr in enumerate(self.all_attributes)}
        self.num_attributes = len(self.all_attributes)

        # Process concepts and group attributes by concept
        self.concepts = []
        self.concept_to_attributes = {}
        for concept, attribute in concept_data:
            if concept not in self.concept_to_attributes:
                self.concepts.append(concept)
                self.concept_to_attributes[concept] = []
            self.concept_to_attributes[concept].append(attribute)
        
        self.concepts = sorted(self.concepts)
        self.concept_to_idx = {concept: i for i, concept in enumerate(self.concepts)}
        self.idx_to_concept = {i: concept for i, concept in enumerate(self.concepts)}

        # Define default transformations if none are provided
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform

    def __len__(self):
        """Returns the total number of concepts."""
        return len(self.concepts)

    def __getitem__(self, idx):
        """
        Fetches a sample from the dataset.

        Args:
            idx (int): Index of the sample.

        Returns:
            tuple: (image_tensor, attribute_vector) where attribute_vector is a
                   multi-hot encoded tensor.
        """
        # Get concept name from index
        concept_name = self.idx_to_concept[idx]
        
        # Load and transform the image
        # Assumes image files are named like '{concept_name}.png'
        img_path = os.path.join(self.image_dir, f"{concept_name}.png")
        try:
            image = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            # Handle missing images gracefully by creating a placeholder
            print(f"Warning: Image for '{concept_name}' not found. Using a placeholder.")
            image = self._create_placeholder_image(concept_name)

        if self.transform:
            image_tensor = self.transform(image)

        # Get the list of attributes for the concept
        attributes = self.concept_to_attributes[concept_name]
        
        # Create the multi-hot encoded vector for attributes
        attribute_vector = torch.zeros(self.num_attributes, dtype=torch.float32)
        for attr in attributes:
            if attr in self.attribute_to_idx:
                attribute_vector[self.attribute_to_idx[attr]] = 1

        return image_tensor, attribute_vector

    def _create_placeholder_image(self, text):
        """Generates a placeholder image with text."""
        img = Image.new('RGB', (128, 128), color = (200, 200, 200))
        d = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("arial.ttf", 15)
        except IOError:
            font = ImageFont.load_default()
        d.text((10, 60), text, fill=(0, 0, 0), font=font)
        return img


class AttributeTaxonomyDataset(Dataset):
    """
    PyTorch Dataset for attributes and their taxonomy categories.

    For each attribute, it returns a one-hot encoded vector of its
    taxonomic category.
    """
    def __init__(self, attribute_file):
        """
        Args:
            attribute_file (string): Path to the json file with attribute taxonomy.
        """
        with open(attribute_file, 'r') as f:
            self.attribute_data = json.load(f)

        # Create vocabulary and mappings for attributes
        self.all_attributes = sorted(list(self.attribute_data.keys()))
        self.attribute_to_idx = {attr: i for i, attr in enumerate(self.all_attributes)}
        self.idx_to_attribute = {i: attr for i, attr in enumerate(self.all_attributes)}
        
        # Create vocabulary and mappings for taxonomy categories
        self.all_taxonomies = sorted(list(set(self.attribute_data.values())))
        self.taxonomy_to_idx = {tax: i for i, tax in enumerate(self.all_taxonomies)}
        self.idx_to_taxonomy = {i: tax for i, tax in enumerate(self.all_taxonomies)}
        self.num_taxonomies = len(self.all_taxonomies)

    def __len__(self):
        """Returns the total number of attributes."""
        return len(self.all_attributes)

    def __getitem__(self, idx):
        """
        Fetches a sample from the dataset.

        Args:
            idx (int): Index of the sample.

        Returns:
            tuple: (attribute_idx, taxonomy_vector) where taxonomy_vector is a
                   one-hot encoded tensor.
        """
        # Get attribute name and its taxonomy from index
        attribute_name = self.idx_to_attribute[idx]
        taxonomy_name = self.attribute_data[attribute_name]

        # Get the index for the taxonomy category
        taxonomy_idx = self.taxonomy_to_idx[taxonomy_name]
        
        # Create the one-hot encoded vector
        taxonomy_vector = torch.zeros(self.num_taxonomies, dtype=torch.float32)
        taxonomy_vector[taxonomy_idx] = 1

        # The input is the index of the attribute itself
        attribute_idx = torch.tensor(idx, dtype=torch.long)

        return attribute_idx, taxonomy_vector