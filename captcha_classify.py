<<<<<<< HEAD
import logging
import os
import re
from pathlib import Path
from typing import List, Union
import csv

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

logger = logging.getLogger(__name__)

class CaptchaLengthNet(nn.Module):
    def __init__(self):
        super(CaptchaLengthNet, self).__init__()
        
        # Define a CNN model for length classification
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 12 * 6, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 5)  # Output classes from 2 to 6 (index 0 = length 2, 1 = length 3, etc.)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.fc(x)
        return x

class CaptchaModel(nn.Module):
    def __init__(self, num_chars: int, num_classes: int, backbone='resnet18', pretrained=True):
        super().__init__()
        
        backbone_configs = {
            'resnet18': (models.resnet18, 512),
            'resnet34': (models.resnet34, 512),
            'resnet50': (models.resnet50, 2048)
        }
        
        if backbone not in backbone_configs:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        backbone_fn, feature_size = backbone_configs[backbone]
        self.backbone = backbone_fn(pretrained=pretrained)
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        
        self.dropout = nn.Dropout(0.5)
        self.char_outputs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_size, 256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, num_classes)
            )
            for _ in range(num_chars)
        ])
        
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.backbone(x)
        features = features.view(features.size(0), -1)
        features = self.dropout(features)
        return [char_out(features) for char_out in self.char_outputs]

class CaptchaClassifier:
    def __init__(self, length_model_path: Union[str, Path], symbols_file: Union[str, Path],
                 model_dir: Union[str, Path], width=198, height=96, backbone='resnet18', device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.width = width
        self.height = height
        
        # Load symbols
        with open(symbols_file, 'r') as f:
            self.symbols = f.readline().strip()
        self.num_classes = len(self.symbols)
        
        # Load length classifier model
        self.length_model = CaptchaLengthNet().to(self.device)
        self.length_model.load_state_dict(torch.load(length_model_path, map_location=self.device)['model_state_dict'])
        self.length_model.eval()
        
        # Load solver models for each length
        self.models = {}
        for length in range(2, 7):
            model_path = os.path.join(model_dir, f'captcha_model{length}_best.pth')
            captcha_model = CaptchaModel(length, self.num_classes, backbone=backbone, pretrained=False)
            captcha_model.load_state_dict(torch.load(model_path, map_location=self.device)['model_state_dict'])
            captcha_model = captcha_model.to(self.device)
            captcha_model.eval()
            self.models[length] = captcha_model

        # Setup transform
        self.transform = transforms.Compose([
            transforms.Resize((height, width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        logger.info(f"Models loaded successfully on {self.device}")

    def preprocess_image(self, image_input: Union[str, Path, Image.Image]) -> torch.Tensor:
        if isinstance(image_input, (str, Path)):
            image = Image.open(image_input).convert('RGB')
        elif isinstance(image_input, Image.Image):
            image = image_input
        else:
            raise ValueError("Input must be either a path or PIL Image")
            
        return self.transform(image).unsqueeze(0).to(self.device)

    def classify_length(self, image: torch.Tensor) -> int:
        with torch.no_grad():
            output = self.length_model(image)
            _, predicted_length_index = torch.max(output, 1)
            return predicted_length_index.item() + 2  # Map 0 -> 2, 1 -> 3, ..., 4 -> 6

    def decode_prediction(self, outputs):
        result = []
        for output in outputs:
            _, predicted = torch.max(output, 1)
            result.append(self.symbols[predicted.item()])
        return ''.join(result)

    def classify(self, image_input: Union[str, Path, Image.Image]) -> str:
        image = self.preprocess_image(image_input)
        
        # Step 1: Predict CAPTCHA length
        captcha_length = self.classify_length(image)
        logger.info(f"Predicted CAPTCHA length: {captcha_length}")
        
        # Step 2: Select corresponding model
        model = self.models[captcha_length]
        
        # Step 3: Use the selected model to decode CAPTCHA
        with torch.no_grad():
            outputs = model(image)
            return self.decode_prediction(outputs)

    def classify_batch(self, image_paths: List[Union[str, Path]], batch_size: int = 32) -> List[str]:
        predictions = []
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            batch_images = torch.cat([self.preprocess_image(path) for path in batch_paths])
            
            for j in range(batch_images.size(0)):
                image = batch_images[j].unsqueeze(0)
                
                # Step 1: Predict CAPTCHA length for each image
                captcha_length = self.classify_length(image)
                
                # Step 2: Use corresponding model for CAPTCHA decoding
                model = self.models[captcha_length]
                
                with torch.no_grad():
                    outputs = model(image)
                    predictions.append(self.decode_prediction(outputs))
                    
            logger.debug(f"Processed batch {i // batch_size + 1}/{(len(image_paths) - 1) // batch_size + 1}")
        
        return predictions

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='CAPTCHA Classification Script')
    parser.add_argument('--length_model', type=str, required=True, help='Path to length classifier model')
    parser.add_argument('--symbols', type=str, required=True, help='Path to symbols file')
    parser.add_argument('--model_dir', type=str, required=True, help='Directory containing CAPTCHA solver models')
    parser.add_argument('--image', type=str, help='Path to single image')
    parser.add_argument('--dir', type=str, help='Path to images directory')
    parser.add_argument('--output', type=str, help='Path for results')
    parser.add_argument('--width', type=int, default=198, help='Image width')
    parser.add_argument('--height', type=int, default=96, help='Image height')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--log-level', type=str, default='INFO',
                      choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s [%(levelname)s] %(message)s'
    )

    try:
        classifier = CaptchaClassifier(
            length_model_path=args.length_model, 
            symbols_file=args.symbols,
            model_dir=args.model_dir,
            width=args.width, height=args.height
        )

        results = []
        if args.image:
            prediction = classifier.classify(args.image)
            results.append((os.path.basename(args.image), prediction))
            logger.info(f"Image: {args.image}, Prediction: {prediction}")

        elif args.dir:
            image_files = [os.path.join(args.dir, f) for f in os.listdir(args.dir)
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            if not image_files:
                logger.error(f"No valid images found in {args.dir}")
                return

            predictions = classifier.classify_batch(image_files, args.batch_size)
            results.extend(zip(map(os.path.basename, image_files), predictions))
            
            for img_file, pred in results:
                logger.info(f"Image: {img_file}, Prediction: {pred}")

        if args.output and results:
            with open(args.output, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["samdanit"])  
                writer.writerows(results)
            logger.info(f"Results saved to {args.output}")

    except Exception as e:
        logger.error(f"Script execution failed: {str(e)}")
        raise

if __name__ == '__main__':
    main()
=======
import logging
import os
import re
from pathlib import Path
from typing import List, Union
import csv

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

logger = logging.getLogger(__name__)

class CaptchaLengthNet(nn.Module):
    def __init__(self):
        super(CaptchaLengthNet, self).__init__()
        
        # Define a CNN model for length classification
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 12 * 6, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 5)  # Output classes from 2 to 6 (index 0 = length 2, 1 = length 3, etc.)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.fc(x)
        return x

class CaptchaModel(nn.Module):
    def __init__(self, num_chars: int, num_classes: int, backbone='resnet18', pretrained=True):
        super().__init__()
        
        backbone_configs = {
            'resnet18': (models.resnet18, 512),
            'resnet34': (models.resnet34, 512),
            'resnet50': (models.resnet50, 2048)
        }
        
        if backbone not in backbone_configs:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        backbone_fn, feature_size = backbone_configs[backbone]
        self.backbone = backbone_fn(pretrained=pretrained)
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        
        self.dropout = nn.Dropout(0.5)
        self.char_outputs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_size, 256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, num_classes)
            )
            for _ in range(num_chars)
        ])
        
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.backbone(x)
        features = features.view(features.size(0), -1)
        features = self.dropout(features)
        return [char_out(features) for char_out in self.char_outputs]

class CaptchaClassifier:
    def __init__(self, length_model_path: Union[str, Path], symbols_file: Union[str, Path],
                 model_dir: Union[str, Path], width=198, height=96, backbone='resnet18', device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.width = width
        self.height = height
        
        # Load symbols
        with open(symbols_file, 'r') as f:
            self.symbols = f.readline().strip()
        self.num_classes = len(self.symbols)
        
        # Load length classifier model
        self.length_model = CaptchaLengthNet().to(self.device)
        self.length_model.load_state_dict(torch.load(length_model_path, map_location=self.device)['model_state_dict'])
        self.length_model.eval()
        
        # Load solver models for each length
        self.models = {}
        for length in range(2, 7):
            model_path = os.path.join(model_dir, f'captcha_model{length}_best.pth')
            captcha_model = CaptchaModel(length, self.num_classes, backbone=backbone, pretrained=False)
            captcha_model.load_state_dict(torch.load(model_path, map_location=self.device)['model_state_dict'])
            captcha_model = captcha_model.to(self.device)
            captcha_model.eval()
            self.models[length] = captcha_model

        # Setup transform
        self.transform = transforms.Compose([
            transforms.Resize((height, width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        logger.info(f"Models loaded successfully on {self.device}")

    def preprocess_image(self, image_input: Union[str, Path, Image.Image]) -> torch.Tensor:
        if isinstance(image_input, (str, Path)):
            image = Image.open(image_input).convert('RGB')
        elif isinstance(image_input, Image.Image):
            image = image_input
        else:
            raise ValueError("Input must be either a path or PIL Image")
            
        return self.transform(image).unsqueeze(0).to(self.device)

    def classify_length(self, image: torch.Tensor) -> int:
        with torch.no_grad():
            output = self.length_model(image)
            _, predicted_length_index = torch.max(output, 1)
            return predicted_length_index.item() + 2  # Map 0 -> 2, 1 -> 3, ..., 4 -> 6

    def decode_prediction(self, outputs):
        result = []
        for output in outputs:
            _, predicted = torch.max(output, 1)
            result.append(self.symbols[predicted.item()])
        return ''.join(result)

    def classify(self, image_input: Union[str, Path, Image.Image]) -> str:
        image = self.preprocess_image(image_input)
        
        # Step 1: Predict CAPTCHA length
        captcha_length = self.classify_length(image)
        logger.info(f"Predicted CAPTCHA length: {captcha_length}")
        
        # Step 2: Select corresponding model
        model = self.models[captcha_length]
        
        # Step 3: Use the selected model to decode CAPTCHA
        with torch.no_grad():
            outputs = model(image)
            return self.decode_prediction(outputs)

    def classify_batch(self, image_paths: List[Union[str, Path]], batch_size: int = 32) -> List[str]:
        predictions = []
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            batch_images = torch.cat([self.preprocess_image(path) for path in batch_paths])
            
            for j in range(batch_images.size(0)):
                image = batch_images[j].unsqueeze(0)
                
                # Step 1: Predict CAPTCHA length for each image
                captcha_length = self.classify_length(image)
                
                # Step 2: Use corresponding model for CAPTCHA decoding
                model = self.models[captcha_length]
                
                with torch.no_grad():
                    outputs = model(image)
                    predictions.append(self.decode_prediction(outputs))
                    
            logger.debug(f"Processed batch {i // batch_size + 1}/{(len(image_paths) - 1) // batch_size + 1}")
        
        return predictions

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='CAPTCHA Classification Script')
    parser.add_argument('--length_model', type=str, required=True, help='Path to length classifier model')
    parser.add_argument('--symbols', type=str, required=True, help='Path to symbols file')
    parser.add_argument('--model_dir', type=str, required=True, help='Directory containing CAPTCHA solver models')
    parser.add_argument('--image', type=str, help='Path to single image')
    parser.add_argument('--dir', type=str, help='Path to images directory')
    parser.add_argument('--output', type=str, help='Path for results')
    parser.add_argument('--width', type=int, default=198, help='Image width')
    parser.add_argument('--height', type=int, default=96, help='Image height')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--log-level', type=str, default='INFO',
                      choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s [%(levelname)s] %(message)s'
    )

    try:
        classifier = CaptchaClassifier(
            length_model_path=args.length_model, 
            symbols_file=args.symbols,
            model_dir=args.model_dir,
            width=args.width, height=args.height
        )

        results = []
        if args.image:
            prediction = classifier.classify(args.image)
            results.append((os.path.basename(args.image), prediction))
            logger.info(f"Image: {args.image}, Prediction: {prediction}")

        elif args.dir:
            image_files = [os.path.join(args.dir, f) for f in os.listdir(args.dir)
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            if not image_files:
                logger.error(f"No valid images found in {args.dir}")
                return

            predictions = classifier.classify_batch(image_files, args.batch_size)
            results.extend(zip(map(os.path.basename, image_files), predictions))
            
            for img_file, pred in results:
                logger.info(f"Image: {img_file}, Prediction: {pred}")

        if args.output and results:
            with open(args.output, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["samdanit"])  
                writer.writerows(results)
            logger.info(f"Results saved to {args.output}")

    except Exception as e:
        logger.error(f"Script execution failed: {str(e)}")
        raise

if __name__ == '__main__':
    main()
>>>>>>> 85039266770e8c8422c5828ab1d132e406db23c8
