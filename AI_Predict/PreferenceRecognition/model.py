import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, Subset, SubsetRandomSampler
import torchvision
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
import torchvision.transforms.functional as TF
from PIL import Image
from pathlib import Path
import shutil
from sklearn.model_selection import StratifiedKFold
import copy

# Fix ImageDataGenerator import for PyTorch compatibility
try:
    # Create a compatibility wrapper for Keras's ImageDataGenerator using PyTorch
    class ImageDataGenerator:
        def __init__(self, rescale=None, rotation_range=0, width_shift_range=0,
                    height_shift_range=0, shear_range=0, zoom_range=0,
                    horizontal_flip=False, vertical_flip=False, fill_mode='nearest',
                    brightness_range=None, validation_split=0):
            self.rescale = rescale
            self.rotation_range = rotation_range
            self.width_shift_range = width_shift_range
            self.height_shift_range = height_shift_range
            self.shear_range = shear_range
            self.zoom_range = zoom_range
            self.horizontal_flip = horizontal_flip
            self.vertical_flip = vertical_flip
            self.fill_mode = fill_mode
            self.brightness_range = brightness_range
            self.validation_split = validation_split
        
        def flow_from_directory(self, directory, target_size=(224, 224), batch_size=32, 
                                class_mode='categorical', shuffle=True, subset=None):
            # Define PyTorch transformations that match the Keras augmentations
            transform_list = []
            
            # Augmentation transformations
            if self.rotation_range > 0:
                transform_list.append(transforms.RandomRotation(self.rotation_range))
                
            if self.width_shift_range > 0 or self.height_shift_range > 0:
                max_shift = max(self.width_shift_range, self.height_shift_range)
                transform_list.append(transforms.RandomAffine(
                    degrees=0, translate=(max_shift, max_shift)
                ))
                
            if self.shear_range > 0:
                transform_list.append(transforms.RandomAffine(
                    degrees=0, shear=self.shear_range * 180
                ))
                
            if self.zoom_range > 0:
                if isinstance(self.zoom_range, (list, tuple)):
                    zoom_low, zoom_high = self.zoom_range
                else:
                    zoom_low, zoom_high = 1-self.zoom_range, 1+self.zoom_range
                transform_list.append(transforms.RandomResizedCrop(
                    target_size, scale=(1/zoom_high, 1/zoom_low)
                ))
            else:
                transform_list.append(transforms.Resize(target_size))
                
            if self.horizontal_flip:
                transform_list.append(transforms.RandomHorizontalFlip())
                
            if self.vertical_flip:
                transform_list.append(transforms.RandomVerticalFlip())
                
            if self.brightness_range:
                brightness_factor = max(0, 1-self.brightness_range[0]), 1+self.brightness_range[1]
                transform_list.append(transforms.ColorJitter(brightness=brightness_factor))
            
            # Always add ToTensor and normalization
            transform_list.append(transforms.ToTensor())
            
            # Normalization based on rescale parameter
            if self.rescale == 1./255:
                # No additional normalization needed as ToTensor already scales to [0,1]
                pass 
            else:
                # Use ImageNet normalization as a good default
                transform_list.append(transforms.Normalize([0.485, 0.456, 0.406], 
                                                          [0.229, 0.224, 0.225]))
            
            data_transform = transforms.Compose(transform_list)
            
            # Create dataset directory - handling validation_split
            if subset == 'training' and self.validation_split > 0:
                # Custom PyTorch dataset with validation split
                full_dataset = ImageFolder(directory, transform=data_transform)
                
                dataset_size = len(full_dataset)
                split = int(np.floor(self.validation_split * dataset_size))
                indices = list(range(dataset_size))
                
                if shuffle:
                    np.random.shuffle(indices)
                
                train_indices = indices[split:]
                
                if shuffle:
                    train_sampler = SubsetRandomSampler(train_indices)
                    dataset = Subset(full_dataset, train_indices)
                    dataloader = DataLoader(
                        dataset, batch_size=batch_size, sampler=train_sampler
                    )
                else:
                    dataset = Subset(full_dataset, train_indices)
                    dataloader = DataLoader(
                        dataset, batch_size=batch_size
                    )
                    
            elif subset == 'validation' and self.validation_split > 0:
                full_dataset = ImageFolder(directory, transform=data_transform)
                
                dataset_size = len(full_dataset)
                split = int(np.floor(self.validation_split * dataset_size))
                indices = list(range(dataset_size))
                
                if shuffle:
                    np.random.shuffle(indices)
                
                val_indices = indices[:split]
                
                if shuffle:
                    val_sampler = SubsetRandomSampler(val_indices)
                    dataset = Subset(full_dataset, val_indices)
                    dataloader = DataLoader(
                        dataset, batch_size=batch_size, sampler=val_sampler
                    )
                else:
                    dataset = Subset(full_dataset, val_indices)
                    dataloader = DataLoader(
                        dataset, batch_size=batch_size
                    )
            else:
                # Regular dataset without validation split
                dataset = ImageFolder(directory, transform=data_transform)
                dataloader = DataLoader(
                    dataset, batch_size=batch_size, shuffle=shuffle
                )
            
            # Print dataset creation info
            print(f"Loading {'training' if subset == 'training' else 'validation' if subset == 'validation' else 'test'} dataset from {directory}")
            
            # Create a wrapper object with same attributes as Keras ImageDataGenerator
            class DatasetWrapper:
                def __init__(self, dataloader, dataset):
                    self.dataloader = dataloader
                    self.dataset = dataset
                    self.n = len(dataset)
                    self.samples = self.n
                    
                    # Handle class indices based on dataset type
                    if hasattr(dataset, 'dataset') and hasattr(dataset.dataset, 'class_to_idx'):
                        self.class_indices = dataset.dataset.class_to_idx
                    elif hasattr(dataset, 'class_to_idx'):
                        self.class_indices = dataset.class_to_idx
                    else:
                        self.class_indices = {i: i for i in range(2)}  # Default binary
                        
                    self.classes = list(self.class_indices.keys())
                
                def __iter__(self):
                    for inputs, targets in self.dataloader:
                        if class_mode == 'binary':
                            targets = targets.float()
                        yield inputs.numpy(), targets.numpy()
                        
                def reset(self):
                    pass  # PyTorch DataLoader handles this internally
                    
            wrapper = DatasetWrapper(dataloader, dataset)
            return wrapper
except ImportError:
    raise ImportError("Could not import required PyTorch libraries")

class PreferenceModel:
    def __init__(self, model_path=None):
        self.model = None
        self.model_path = model_path or os.path.join(os.path.dirname(__file__), 'saved_models','ensemble' , 'preference_model.pt')
        self.img_height = 224
        self.img_width = 224
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def build_model(self, transfer_learning=True, base_network='MobileNetV2'):
        """
        Xây dựng mô hình neural network để nhận diện sở thích
        
        Args:
            transfer_learning: Sử dụng transfer learning với mô hình đã train sẵn
            base_network: Loại mô hình cơ sở ('MobileNetV2', 'VGG16', 'ResNet50')
        """
        if transfer_learning:
            # Lựa chọn mô hình cơ sở dựa trên tham số
            if base_network == 'VGG16':
                base_model = models.vgg16(weights='IMAGENET1K_V1')
                n_features = base_model.classifier[6].in_features
                base_model.classifier[6] = nn.Identity()  # Remove the last layer
            
            elif base_network == 'ResNet50':
                base_model = models.resnet50(weights='IMAGENET1K_V1')
                n_features = base_model.fc.in_features
                base_model.fc = nn.Identity()  # Remove the last layer
            
            elif base_network == 'EfficientNet':
                base_model = models.efficientnet_b0(weights='IMAGENET1K_V1')
                n_features = base_model.classifier[1].in_features
                base_model.classifier = nn.Identity()  # Remove the classifier
            
            else:  # Default: MobileNetV2
                base_model = models.mobilenet_v2(weights='IMAGENET1K_V1')
                n_features = base_model.classifier[1].in_features
                base_model.classifier = nn.Identity()  # Remove the classifier
            
            # Freeze early layers
            for param in list(base_model.parameters())[:-20]:
                param.requires_grad = False
            
            # Create the complete model
            self.model = nn.Sequential(
                base_model,
                nn.Dropout(0.5),
                nn.Linear(n_features, 512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(128, 1),
                nn.Sigmoid()
            )
        else:
            # Custom CNN model
            self.model = nn.Sequential(
                # First conv block
                nn.Conv2d(3, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                
                # Second conv block
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                
                # Third conv block
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                
                # Fourth conv block
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                
                # Flatten and FC layers
                nn.Flatten(),
                nn.Linear(256 * 14 * 14, 512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, 1),
                nn.Sigmoid()
            )
        
        # Move model to the appropriate device
        self.model = self.model.to(self.device)
        return self.model
    
    def train(self, training_dir, validation_split=0.2, epochs=30, batch_size=16):
        """
        Train mô hình với dữ liệu từ thư mục
        
        Args:
            training_dir: Thư mục chứa dữ liệu training, cần có 2 thư mục con 'like' và 'dislike'
            validation_split: Tỉ lệ dữ liệu dùng cho validation
            epochs: Số epochs để train
            batch_size: Kích thước batch
        """
        if not self.model:
            self.build_model(transfer_learning=True)
            
        # Data augmentation cho PyTorch
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=40,
            width_shift_range=0.3,
            height_shift_range=0.3,
            shear_range=0.3,
            zoom_range=0.3,
            horizontal_flip=True,
            vertical_flip=False,
            fill_mode='reflect',
            brightness_range=[0.7, 1.3],
            validation_split=validation_split
        )
        
        # Lấy dữ liệu từ thư mục
        train_generator = train_datagen.flow_from_directory(
            training_dir,
            target_size=(self.img_height, self.img_width),
            batch_size=batch_size,
            class_mode='binary',
            subset='training',
            shuffle=True
        )
        
        validation_generator = train_datagen.flow_from_directory(
            training_dir,
            target_size=(self.img_height, self.img_width),
            batch_size=batch_size,
            class_mode='binary',
            subset='validation'
        )
        
        # Tạo thư mục để lưu model
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        
        # Loss function and optimizer
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.0001)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3, verbose=True)
        
        # Tính class weights để cân bằng dữ liệu
        class_weights = None
        try:
            class_indices = train_generator.class_indices
            classes = list(class_indices.keys())
            
            # Đếm số lượng mẫu từ thư mục
            class_counts = {}
            for class_name in classes:
                class_dir = os.path.join(training_dir, class_name)
                class_counts[class_name] = len(list(Path(class_dir).glob('*.jpg')))
            
            # Tạo class weights nếu có đủ dữ liệu
            if sum(class_counts.values()) > 0:
                max_count = max(class_counts.values())
                weight_dict = {
                    class_indices[class_name]: max_count / count if count > 0 else 1.0
                    for class_name, count in class_counts.items()
                }
                
                # Convert to tensor weights
                if len(weight_dict) == 2:  # Binary classification
                    pos_weight = torch.tensor([weight_dict.get(1, 1.0) / weight_dict.get(0, 1.0)])
                    print(f"Using class weights with pos_weight: {pos_weight.item()}")
                    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(self.device))
        except Exception as e:
            print(f"Could not compute class weights: {e}")
        
        # Training history
        history = {
            'accuracy': [],
            'loss': [],
            'val_accuracy': [],
            'val_loss': []
        }
        
        best_val_loss = float('inf')
        best_model_state = None
        
        # Training loop
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            for inputs, targets in train_generator:
                inputs = torch.from_numpy(inputs).to(self.device)
                targets = torch.from_numpy(targets).float().to(self.device).view(-1, 1)
                
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                # Statistics
                running_loss += loss.item() * inputs.size(0)
                predicted = (outputs >= 0.5).float()
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
                
                # Break after one epoch
                if total >= train_generator.n:
                    break
            
            train_loss = running_loss / train_generator.n
            train_acc = correct / total
            
            # Validation phase
            self.model.eval()
            running_loss = 0.0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for inputs, targets in validation_generator:
                    inputs = torch.from_numpy(inputs).to(self.device)
                    targets = torch.from_numpy(targets).float().to(self.device).view(-1, 1)
                    
                    # Forward pass
                    outputs = self.model(inputs)
                    loss = criterion(outputs, targets)
                    
                    # Statistics
                    running_loss += loss.item() * inputs.size(0)
                    predicted = (outputs >= 0.5).float()
                    total += targets.size(0)
                    correct += (predicted == targets).sum().item()
                    
                    # Break after one epoch
                    if total >= validation_generator.n:
                        break
            
            val_loss = running_loss / validation_generator.n
            val_acc = correct / total
            
            # Learning rate scheduler step
            scheduler.step(val_loss)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = copy.deepcopy(self.model.state_dict())
                torch.save(self.model.state_dict(), self.model_path)
                print(f"Model saved at epoch {epoch+1} with val_loss: {val_loss:.4f}")
            
            # Update history
            history['accuracy'].append(train_acc)
            history['loss'].append(train_loss)
            history['val_accuracy'].append(val_acc)
            history['val_loss'].append(val_loss)
            
            print(f'Epoch {epoch+1}/{epochs}: loss: {train_loss:.4f} - accuracy: {train_acc:.4f} - val_loss: {val_loss:.4f} - val_accuracy: {val_acc:.4f}')
        
        # Load best model
        if best_model_state:
            self.model.load_state_dict(best_model_state)
        
        return history
    
    def load_trained_model(self):
        """Tải mô hình đã train từ file"""
        if os.path.exists(self.model_path):
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            self.model.eval()
            return True
        else:
            print(f"Không tìm thấy model tại {self.model_path}")
            return False
            
    def predict(self, image):
        """
        Dự đoán sở thích dựa trên ảnh
        
        Args:
            image: Ảnh đầu vào (numpy array) có shape (img_height, img_width, 3)
            
        Returns:
            probability: Xác suất thích (0-1)
        """
        if not self.model:
            loaded = self.load_trained_model()
            if not loaded:
                raise Exception("Cần train hoặc load model trước khi dự đoán")
                
        # Chuẩn bị ảnh
        if len(image.shape) == 3:  # Đảm bảo có batch dimension
            img = np.expand_dims(image, axis=0)
        else:
            img = image
            
        # Normalize ảnh nếu chưa normalize
        if img.max() > 1.0:
            img = img / 255.0
            
        # Kiểm tra shape của ảnh
        if img.shape[1:3] != (self.img_height, self.img_width):
            # Resize ảnh nếu cần thiết
            from PIL import Image
            resized_img = np.zeros((img.shape[0], self.img_height, self.img_width, 3))
            for i in range(img.shape[0]):
                pil_img = Image.fromarray((img[i] * 255).astype(np.uint8))
                resized_pil = pil_img.resize((self.img_width, self.img_height))
                resized_img[i] = np.array(resized_pil) / 255.0
            img = resized_img
            
        # Convert numpy to tensor
        img = torch.from_numpy(img.transpose(0, 3, 1, 2)).float().to(self.device)
        
        # Set to evaluation mode
        self.model.eval()
        
        # Get prediction
        with torch.no_grad():
            prediction = self.model(img)
            
        # Debug: in ra shape và giá trị của prediction
        print(f"Prediction shape: {prediction.shape}, values: {prediction}")
        
        # Convert to python float for compatibility
        probability = float(prediction[0][0])
            
        return probability
        
    def evaluate(self, test_data_dir):
        """
        Đánh giá mô hình trên tập dữ liệu test
        
        Args:
            test_data_dir: Thư mục chứa dữ liệu test
            
        Returns:
            results: Dictionary chứa kết quả đánh giá
        """
        if not self.model:
            loaded = self.load_trained_model()
            if not loaded:
                raise Exception("Cần train hoặc load model trước khi đánh giá")
        
        test_datagen = ImageDataGenerator(rescale=1./255)
        test_generator = test_datagen.flow_from_directory(
            test_data_dir,
            target_size=(self.img_height, self.img_width),
            batch_size=32,
            class_mode='binary',
            shuffle=False
        )
        
        # Evaluate model
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        criterion = nn.BCELoss()
        
        with torch.no_grad():
            for inputs, targets in test_generator:
                inputs = torch.from_numpy(inputs).to(self.device)
                targets = torch.from_numpy(targets).float().to(self.device).view(-1, 1)
                
                # Forward pass
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                
                # Statistics
                running_loss += loss.item() * inputs.size(0)
                predicted = (outputs >= 0.5).float()
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
                
                # Break after one epoch
                if total >= test_generator.n:
                    break
        
        test_loss = running_loss / test_generator.n
        test_acc = correct / total
        
        return {
            'loss': test_loss,
            'accuracy': test_acc
        }

class EnsembleModel:
    """
    Implement Ensemble technique with 5-Fold Cross-Validation for the preference model.
    This model trains 5 different models on different subsets of the data and combines
    their predictions to get a more robust result.
    """
    
    def __init__(self, n_folds=5, model_dir=None):
        """
        Initialize the ensemble model
        
        Args:
            n_folds: Number of folds for cross-validation
            model_dir: Directory to save the models
        """
        self.n_folds = n_folds
        self.models = []
        self.img_height = 224
        self.img_width = 224
        self.model_dir = model_dir or os.path.join(os.path.dirname(__file__), 'saved_models', 'ensemble')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        os.makedirs(self.model_dir, exist_ok=True)
        
    def _prepare_fold_data(self, data_dir, fold_idx, train_indices, val_indices):
        """
        Prepare data for a specific fold by creating symlinks to the original data
        
        Args:
            data_dir: Original data directory
            fold_idx: Current fold index
            train_indices: Indices for training images
            val_indices: Indices for validation images
            
        Returns:
            fold_dir: Directory containing the fold's data
        """
        # Create fold directory
        fold_dir = os.path.join(self.model_dir, f"fold_{fold_idx}")
        train_dir = os.path.join(fold_dir, "train")
        val_dir = os.path.join(fold_dir, "val")
        
        # Create directories for classes
        for split_dir in [train_dir, val_dir]:
            os.makedirs(os.path.join(split_dir, "like"), exist_ok=True)
            os.makedirs(os.path.join(split_dir, "dislike"), exist_ok=True)
        
        # Get all files from original directory
        like_files = list(Path(os.path.join(data_dir, "like")).glob("*.jpg"))
        dislike_files = list(Path(os.path.join(data_dir, "dislike")).glob("*.jpg"))
        all_files = like_files + dislike_files
        
        # Create labels (1 for like, 0 for dislike)
        labels = [1] * len(like_files) + [0] * len(dislike_files)
        
        # Split into training and validation
        train_files = [all_files[i] for i in train_indices]
        train_labels = [labels[i] for i in train_indices]
        val_files = [all_files[i] for i in val_indices]
        val_labels = [labels[i] for i in val_indices]
        
        # Create symbolic links for training files
        for file, label in zip(train_files, train_labels):
            class_name = "like" if label == 1 else "dislike"
            dest_path = os.path.join(train_dir, class_name, file.name)
            shutil.copy2(file, dest_path)
        
        # Create symbolic links for validation files
        for file, label in zip(val_files, val_labels):
            class_name = "like" if label == 1 else "dislike"
            dest_path = os.path.join(val_dir, class_name, file.name)
            shutil.copy2(file, dest_path)
        
        return fold_dir
        
    def train(self, data_dir, epochs=30, batch_size=16, base_network='ResNet50', 
              class_weights=None, transfer_learning=True):
        """
        Train the ensemble model using 5-fold cross-validation
        
        Args:
            data_dir: Directory containing the training data (with like/dislike subdirs)
            epochs: Number of training epochs
            batch_size: Batch size for training
            base_network: Base network for transfer learning
            class_weights: Optional class weights for imbalanced data
            transfer_learning: Whether to use transfer learning
            
        Returns:
            histories: List of training histories for each fold
        """
        # Get all files from directory
        like_files = list(Path(os.path.join(data_dir, "like")).glob("*.jpg"))
        dislike_files = list(Path(os.path.join(data_dir, "dislike")).glob("*.jpg"))
        
        all_files = like_files + dislike_files
        
        # Create labels (1 for like, 0 for dislike)
        labels = [1] * len(like_files) + [0] * len(dislike_files)
        
        # Create stratified k-fold
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        
        # Train a model for each fold
        histories = []
        
        for fold_idx, (train_indices, val_indices) in enumerate(skf.split(all_files, labels)):
            print(f"\n===== Training Fold {fold_idx + 1}/{self.n_folds} =====")
            
            # Prepare data for this fold
            fold_dir = self._prepare_fold_data(data_dir, fold_idx, train_indices, val_indices)
            train_dir = os.path.join(fold_dir, "train")
            val_dir = os.path.join(fold_dir, "val")
            
            # Create a new model for this fold
            model_path = os.path.join(self.model_dir, f"preference_model_fold{fold_idx}.pt")
            model = PreferenceModel(model_path=model_path)
            model.build_model(transfer_learning=transfer_learning, base_network=base_network)
            
            # Train the model
            history = model.train(
                training_dir=train_dir,
                validation_split=0.0,  # No additional validation split
                epochs=epochs,
                batch_size=batch_size
            )
            
            # Save the model in the ensemble
            self.models.append(model)
            histories.append(history)
            
            print(f"Fold {fold_idx + 1} training complete. Model saved to {model_path}")
        
        return histories
        
    def predict(self, image):
        """
        Predict using the ensemble of models
        
        Args:
            image: Image to predict (numpy array)
            
        Returns:
            probability: Average prediction probability
        """
        if not self.models:
            # Try to load pre-trained models
            self._load_models()
            
            if not self.models:
                raise Exception("No models available. Please train the ensemble first.")
        
        # Get predictions from all models
        predictions = []
        for model in self.models:
            pred = model.predict(image)
            predictions.append(pred)
        
        # Calculate average prediction
        avg_pred = sum(predictions) / len(predictions)
        
        return avg_pred
        
    def _load_models(self):
        """Load pre-trained models from disk"""
        self.models = []
        
        for i in range(self.n_folds):
            model_path = os.path.join(self.model_dir, f"preference_model_fold{i}.pt")
            
            if os.path.exists(model_path):
                model = PreferenceModel(model_path=model_path)
                model.build_model()  # Create the model architecture
                if model.load_trained_model():
                    self.models.append(model)
        
        return len(self.models) > 0
        
    def evaluate(self, test_dir):
        """
        Evaluate the ensemble model on test data
        
        Args:
            test_dir: Directory containing test data
            
        Returns:
            results: Dictionary containing evaluation results
        """
        if not self.models:
            # Try to load pre-trained models
            if not self._load_models():
                raise Exception("No models available. Please train the ensemble first.")
        
        # Create test data generator
        test_datagen = ImageDataGenerator(rescale=1./255)
        test_generator = test_datagen.flow_from_directory(
            test_dir,
            target_size=(self.img_height, self.img_width),
            batch_size=32,
            class_mode='binary',
            shuffle=False
        )
        
        # Get ground truth labels
        y_true = test_generator.classes
        
        # Reset generator
        test_generator.reset()
        
        # Get predictions from each model
        all_preds = []
        
        with torch.no_grad():
            for model in self.models:
                model_preds = []
                test_generator.reset()
                
                for inputs, _ in test_generator:
                    inputs = torch.from_numpy(inputs.transpose(0, 3, 1, 2)).float().to(self.device)
                    outputs = model.model(inputs)
                    model_preds.extend(outputs.cpu().numpy().flatten())
                    
                    # Break if we've processed all samples
                    if len(model_preds) >= len(y_true):
                        model_preds = model_preds[:len(y_true)]
                        break
                
                all_preds.append(model_preds)
        
        # Average predictions from all models
        y_pred_proba = np.mean(all_preds, axis=0)
        y_pred = (y_pred_proba >= 0.5).astype(int)
        
        # Calculate accuracy
        accuracy = np.mean(y_pred == y_true)
        
        # Calculate confusion matrix
        from sklearn.metrics import confusion_matrix, classification_report
        cm = confusion_matrix(y_true, y_pred)
        
        # Generate report
        report = classification_report(y_true, y_pred, target_names=['Dislike', 'Like'], output_dict=True)
        
        return {
            'accuracy': float(accuracy),
            'confusion_matrix': cm.tolist(),
            'classification_report': report,
            'y_true': y_true.tolist(),
            'y_pred': y_pred.tolist(),
            'y_pred_proba': y_pred_proba.tolist()
        }