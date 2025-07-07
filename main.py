import os
import json
import argparse
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torchvision.models as models
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

# Import the dataset classes from your file
from data import ConceptAttributesDataset, AttributeTaxonomyDataset

# --- Setup logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# --- Set random seed for reproducibility ---
torch.manual_seed(42)

# --- Probe for Task 1: Attribute Prediction from Images ---

class AttributeProbe(nn.Module):
    """
    A multi-label probe to predict attributes from image embeddings.
    """
    def __init__(self, input_dim: int, num_attributes: int, hidden_dim: int = 512) -> None:
        """
        Initializes the probe.
        :param input_dim: The dimensionality of the input image embeddings.
        :param num_attributes: The number of possible attributes (output size).
        :param hidden_dim: The dimensionality of the hidden layer.
        """
        super(AttributeProbe, self).__init__()
        self.probe = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_attributes)
        )

    def forward(self, x):
        return self.probe(x)

    def train_probe(self, train_embeddings: torch.Tensor, train_labels: torch.Tensor,
                    dev_embeddings: torch.Tensor, dev_labels: torch.Tensor,
                    num_epochs: int = 20, learning_rate: float = 0.001,
                    batch_size: int = 32, device: str = 'cpu'):
        """
        Trains the probe on the embeddings.
        """
        self.to(device)
        train_embeddings, train_labels = train_embeddings.to(device), train_labels.to(device)
        dev_embeddings, dev_labels = dev_embeddings.to(device), dev_labels.to(device)

        # BCEWithLogitsLoss is suitable for multi-label classification
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        logging.info('Training the Attribute Probe...')
        for epoch in range(num_epochs):
            self.train()
            for i in range(0, len(train_embeddings), batch_size):
                batch_embeddings = train_embeddings[i:i+batch_size]
                batch_labels = train_labels[i:i+batch_size]

                outputs = self(batch_embeddings)
                loss = criterion(outputs, batch_labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Evaluate on dev set after each epoch
            dev_accuracy = self.evaluate(dev_embeddings, dev_labels, batch_size, device)
            logging.info(f'Epoch [{epoch+1}/{num_epochs}], Dev Accuracy: {dev_accuracy:.4f}')
        logging.info('Training finished.')

    def evaluate(self, data_embeddings: torch.Tensor, labels: torch.Tensor, batch_size: int = 32, device: str = 'cpu') -> float:
        """
        Evaluates the probe's performance on unseen data.
        Returns the accuracy (fraction of correctly predicted labels).
        """
        self.eval()
        self.to(device)
        data_embeddings, labels = data_embeddings.to(device), labels.to(device)
        
        all_preds = []
        with torch.no_grad():
            for i in range(0, len(data_embeddings), batch_size):
                batch_embeddings = data_embeddings[i:i+batch_size]
                outputs = self(batch_embeddings)
                # Use sigmoid and a 0.5 threshold for multi-label prediction
                preds = (torch.sigmoid(outputs) > 0.5).float()
                all_preds.append(preds)
        
        all_preds = torch.cat(all_preds, dim=0)
        
        correct = (all_preds == labels).sum().item()
        accuracy = correct / labels.numel() # numel() gives total number of elements
        return accuracy

# --- Probe for Task 2: Taxonomy Prediction from Attributes ---

class TaxonomyProbe(nn.Module):
    """
    A probe to predict taxonomy category from an attribute.
    It learns its own embeddings for the attributes.
    """
    def __init__(self, num_attributes: int, num_taxonomies: int, embedding_dim: int = 128, hidden_dim: int = 256) -> None:
        """
        Initializes the probe.
        :param num_attributes: The number of attributes (input vocab size).
        :param num_taxonomies: The number of taxonomy categories (output size).
        :param embedding_dim: The dimensionality of the learned attribute embeddings.
        :param hidden_dim: The dimensionality of the hidden layer.
        """
        super(TaxonomyProbe, self).__init__()
        self.embedding = nn.Embedding(num_attributes, embedding_dim)
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_taxonomies)
        )

    def forward(self, x):
        embedded = self.embedding(x)
        return self.classifier(embedded)
    
    def train_probe(self, train_loader: DataLoader, dev_loader: DataLoader,
                    num_epochs: int = 10, learning_rate: float = 0.001, device: str = 'cpu'):
        """Trains the probe."""
        self.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        logging.info('Training the Taxonomy Probe...')
        for epoch in range(num_epochs):
            self.train()
            for attr_indices, taxonomy_vectors in train_loader:
                attr_indices = attr_indices.to(device)
                # CrossEntropyLoss expects class indices, not one-hot vectors
                taxonomy_labels = torch.argmax(taxonomy_vectors, dim=1).to(device)

                outputs = self(attr_indices)
                loss = criterion(outputs, taxonomy_labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            dev_accuracy = self.evaluate(dev_loader, device)
            logging.info(f'Epoch [{epoch+1}/{num_epochs}], Dev Accuracy: {dev_accuracy:.4f}')
        logging.info('Training finished.')

    def evaluate(self, data_loader: DataLoader, device: str = 'cpu') -> float:
        """Evaluates the probe's performance."""
        self.eval()
        self.to(device)
        correct = 0
        total = 0
        with torch.no_grad():
            for attr_indices, taxonomy_vectors in data_loader:
                attr_indices = attr_indices.to(device)
                taxonomy_labels = torch.argmax(taxonomy_vectors, dim=1).to(device)

                outputs = self(attr_indices)
                _, predicted = torch.max(outputs.data, 1)
                
                total += taxonomy_labels.size(0)
                correct += (predicted == taxonomy_labels).sum().item()
        
        accuracy = correct / total
        return accuracy

# --- Helper function for image embedding extraction ---

def get_image_embeddings(feature_extractor, dataloader, device='cpu'):
    """
    Extracts embeddings for all images in a dataloader.
    """
    all_embeddings = []
    all_labels = []
    feature_extractor.eval()
    feature_extractor.to(device)
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Extracting image embeddings"):
            images = images.to(device)
            embeddings = feature_extractor(images)
            all_embeddings.append(embeddings.cpu())
            all_labels.append(labels.cpu())
    
    return torch.cat(all_embeddings, dim=0), torch.cat(all_labels, dim=0)

def get_args_parser():
    parser = argparse.ArgumentParser("Linear Probing")
    parser.add_argument(
        "--model",
        default="facebook/dinov2-base",
        type=str,
        help="Name of model to train",
    )
    return parser

# --- Main execution logic ---

def main(args):
    """Main function to run the probing experiment."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f"Using device: {device}")

    # --- Attribute Probe Workflow ---
    if args.probe_type == 'attribute':
        logging.info("--- Starting Attribute Probe Workflow ---")
        
        # 1. Load data
        dataset = ConceptAttributesDataset(
            concept_file='concept_attributes.json',
            attribute_file='attributes_taxonomy.json',
            image_dir='images'
        )
        # Create train/dev splits (e.g., 80/20)
        dev_size = int(len(dataset) * 0.2)
        train_size = len(dataset) - dev_size
        train_dataset, dev_dataset = torch.utils.data.random_split(dataset, [train_size, dev_size])

        # Use batch_size=len(dataset) to get all embeddings at once
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
        dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False)
        
        # 2. Load pre-trained model for feature extraction
        logging.info("Loading pre-trained ResNet-18 model...")
        feature_extractor = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # Remove the final classification layer to get embeddings
        feature_extractor.fc = nn.Identity()

        # 3. Extract embeddings
        train_embeddings, train_labels = get_image_embeddings(feature_extractor, train_loader, device)
        dev_embeddings, dev_labels = get_image_embeddings(feature_extractor, dev_loader, device)
        
        # 4. Initialize and train the probe
        input_dim = train_embeddings.shape[1]
        num_attributes = train_labels.shape[1]
        probe = AttributeProbe(input_dim=input_dim, num_attributes=num_attributes)
        probe.train_probe(train_embeddings, train_labels, dev_embeddings, dev_labels,
                          num_epochs=args.epochs, learning_rate=args.lr, batch_size=args.batch_size, device=device)

        # 5. Final evaluation
        final_accuracy = probe.evaluate(dev_embeddings, dev_labels, args.batch_size, device)
        logging.info(f"Final Attribute Probe Accuracy on Dev Set: {final_accuracy*100:.2f}%")

    # --- Taxonomy Probe Workflow ---
    elif args.probe_type == 'taxonomy':
        logging.info("--- Starting Taxonomy Probe Workflow ---")
        
        # 1. Load data
        dataset = AttributeTaxonomyDataset(attribute_file='attributes_taxonomy.json')
        dev_size = int(len(dataset) * 0.2)
        train_size = len(dataset) - dev_size
        train_dataset, dev_dataset = torch.utils.data.random_split(dataset, [train_size, dev_size])

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False)
        
        # 2. Initialize and train the probe
        probe = TaxonomyProbe(
            num_attributes=len(dataset.all_attributes),
            num_taxonomies=dataset.num_taxonomies
        )
        probe.train_probe(train_loader, dev_loader, num_epochs=args.epochs, learning_rate=args.lr, device=device)
        
        # 3. Final evaluation
        final_accuracy = probe.evaluate(dev_loader, device)
        logging.info(f"Final Taxonomy Probe Accuracy on Dev Set: {final_accuracy*100:.2f}%")


def setup_dummy_data():
    """Creates dummy data files and images required for the datasets."""
    if os.path.exists('attributes_taxonomy.json'):
        logging.info("Dummy data files already exist. Skipping creation.")
        return

    logging.info("--- Setting up dummy data for demonstration ---")
    attribute_taxonomy_data = {
        "a_bird": "taxonomic", "a_building": "taxonomic", "a_carnivore": "taxonomic",
        "a_cat": "taxonomic", "a_container": "taxonomic", "a_fish": "taxonomic",
        "a_food": "visual", "a_fruit": "visual", "a_gun": "functional",
        "a_herbivore": "taxonomic", "a_house": "taxonomic", "is_red": "visual",
        "can_fly": "functional"
    }
    with open('attributes_taxonomy.json', 'w') as f:
        json.dump(attribute_taxonomy_data, f, indent=4)

    concept_attributes_data = [
        ["bird", "a_bird"], ["bird", "can_fly"], ["cardinal", "a_bird"],
        ["cardinal", "is_red"], ["cardinal", "can_fly"], ["house", "a_building"],
        ["house", "a_house"], ["apple", "a_fruit"], ["apple", "a_food"],
        ["apple", "is_red"], ["lion", "a_cat"], ["lion", "a_carnivore"],
        ["salmon", "a_fish"], ["salmon", "a_food"]
    ]
    with open('concept_attributes.json', 'w') as f:
        json.dump(concept_attributes_data, f, indent=4)

    image_dir = 'images'
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    
    concepts_for_images = set(item[0] for item in concept_attributes_data)
    for concept in concepts_for_images:
        img_path = os.path.join(image_dir, f"{concept}.png")
        if not os.path.exists(img_path):
            img = Image.new('RGB', (128, 128), color = (73, 109, 137))
            d = ImageDraw.Draw(img)
            d.text((10, 60), f"Image of\n{concept}", fill=(255, 255, 0))
            img.save(img_path)
    logging.info("Dummy data setup complete.")


if __name__ == '__main__':
    # Create dummy data if it doesn't exist
    setup_dummy_data()

    parser = argparse.ArgumentParser(description="Train probes to evaluate representations.")
    parser.add_argument('--probe_type', type=str, required=True, choices=['attribute', 'taxonomy'],
                        help="The type of probe to train and evaluate.")
    parser.add_argument('--epochs', type=int, default=20, help="Number of training epochs.")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate.")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for training.")
    
    args = parser.parse_args()
    main(args)