import logging
import os
import re
from pathlib import Path
from typing import List, Optional, Union
import csv

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

logger = logging.getLogger(__name__)

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
    def __init__(self, model_path: Union[str, Path], symbols_file: Union[str, Path],
                 width=198, height=96, length=None, backbone='resnet18', device=None):
        try:
            self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.width = width
            self.height = height
            
            # Load symbols
            with open(symbols_file, 'r') as f:
                self.symbols = f.readline().strip()
            self.num_classes = len(self.symbols)
            
            # Detect CAPTCHA length if not provided
            if length is None:
                length = self._detect_captcha_length(model_path)
                logger.info(f"Detected CAPTCHA length: {length}")
            
            # Create and load model
            self.model = CaptchaModel(length, self.num_classes, backbone=backbone, pretrained=False)
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model = self.model.to(self.device)
            self.model.eval()
            
            # Setup transform
            self.transform = transforms.Compose([
                transforms.Resize((height, width)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            logger.info(f"Model loaded successfully on {self.device}")
            if 'best_accuracy' in checkpoint:
                logger.info(f"Model accuracy: {checkpoint['best_accuracy']}%")
                
        except Exception as e:
            logger.error(f"Failed to initialize classifier: {str(e)}")
            raise

    @staticmethod
    def _detect_captcha_length(checkpoint_path: Union[str, Path]) -> int:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict = checkpoint['model_state_dict']
        
        char_indices = {int(match.group(1)) for key in state_dict.keys() 
                       if (match := re.search(r'char_outputs\.(\d+)\.', key))}
        
        if not char_indices:
            raise ValueError("Could not detect CAPTCHA length from checkpoint")
        return max(char_indices) + 1

    def preprocess_image(self, image_input: Union[str, Path, Image.Image]) -> torch.Tensor:
        try:
            if isinstance(image_input, (str, Path)):
                image = Image.open(image_input).convert('RGB')
            elif isinstance(image_input, Image.Image):
                image = image_input
            else:
                raise ValueError("Input must be either a path or PIL Image")
                
            return self.transform(image).unsqueeze(0).to(self.device)
        except Exception as e:
            logger.error(f"Image processing failed: {str(e)}")
            raise

    def decode_prediction(self, outputs):
        result = []
        for output in outputs:
            if output.dim() == 2:
                _, predicted = torch.max(output, 1)
            else:
                _, predicted = torch.max(output.unsqueeze(0), 1)
            result.append(self.symbols[predicted.item()])
        return ''.join(result)

    def classify(self, image_input: Union[str, Path, Image.Image]) -> str:
        try:
            image = self.preprocess_image(image_input)
            with torch.no_grad():
                outputs = self.model(image)
                return self.decode_prediction(outputs)
        except Exception as e:
            logger.error(f"Classification failed: {str(e)}")
            raise

    def classify_batch(self, image_paths: List[Union[str, Path]], batch_size: int = 32) -> List[str]:
        try:
            predictions = []
            for i in range(0, len(image_paths), batch_size):
                batch_paths = image_paths[i:i + batch_size]
                batch_images = torch.cat([self.preprocess_image(path) for path in batch_paths])
                
                with torch.no_grad():
                    batch_outputs = self.model(batch_images)
                    for j in range(batch_images.size(0)):
                        image_outputs = [output[j] for output in batch_outputs]
                        predictions.append(self.decode_prediction(image_outputs))
                
                logger.debug(f"Processed batch {i//batch_size + 1}/{(len(image_paths)-1)//batch_size + 1}")
            
            return predictions
        except Exception as e:
            logger.error(f"Batch processing failed: {str(e)}")
            raise

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='CAPTCHA Classification Script')
    parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--symbols', type=str, required=True, help='Path to symbols file')
    parser.add_argument('--image', type=str, help='Path to single image')
    parser.add_argument('--dir', type=str, help='Path to images directory')
    parser.add_argument('--output', type=str, help='Path for results')
    parser.add_argument('--width', type=int, default=198, help='Image width')
    parser.add_argument('--height', type=int, default=96, help='Image height')
    parser.add_argument('--length', type=int, help='CAPTCHA length')
    parser.add_argument('--backbone', type=str, default='resnet18',
                      choices=['resnet18', 'resnet34', 'resnet50'])
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
            args.model, args.symbols,
            width=args.width, height=args.height,
            length=args.length, backbone=args.backbone
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

        # if args.output and results:
        #     with open(args.output, 'w') as f:
        #         for img_file, pred in results:
        #             f.write(f"{img_file}\t{pred}\n")
        #     logger.info(f"Results saved to {args.output}")

        if args.output and results:
            with open(args.output, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Image File", "Prediction"])  # Optional: Write header
                writer.writerows(results)
            logger.info(f"Results saved to {args.output}")

    except Exception as e:
        logger.error(f"Script execution failed: {str(e)}")
        raise

if __name__ == '__main__':
    main()