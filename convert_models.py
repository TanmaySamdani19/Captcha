import tensorflow as tf
import numpy as np
import os

def create_length_classifier_model(input_shape=(96, 198, 3)):
    """Create length classifier model"""
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=input_shape),
        
        # Conv1
        tf.keras.layers.Conv2D(32, kernel_size=3, padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.MaxPooling2D(2, 2),
        
        # Conv2
        tf.keras.layers.Conv2D(64, kernel_size=3, padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.MaxPooling2D(2, 2),
        
        # Conv3
        tf.keras.layers.Conv2D(128, kernel_size=3, padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.MaxPooling2D(2, 2),
        
        # Conv4
        tf.keras.layers.Conv2D(256, kernel_size=3, padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.MaxPooling2D(2, 2),
        
        # Flatten and FC layers
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(256),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(5, activation='softmax')
    ])
    
    return model

def create_captcha_model(num_chars, num_classes, input_shape=(96, 198, 3)):
    """Create captcha recognition model"""
    inputs = tf.keras.Input(shape=input_shape)
    
    # CNN backbone
    x = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')(inputs)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Flatten()(x)
    
    # Character outputs
    outputs = []
    for _ in range(num_chars):
        char_output = tf.keras.layers.Dense(256, activation='relu')(x)
        char_output = tf.keras.layers.Dropout(0.5)(char_output)
        char_output = tf.keras.layers.Dense(num_classes, activation='softmax')(char_output)
        outputs.append(char_output)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

def convert_to_tflite(model, output_path):
    """Convert Keras model to TFLite"""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    
    with open(output_path, 'wb') as f:
        f.write(tflite_model)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, required=True,
                      help='Output directory for TFLite models')
    parser.add_argument('--num_classes', type=int, required=True,
                      help='Number of character classes')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create and convert length classifier
    length_model = create_length_classifier_model()
    length_model.compile(optimizer='adam', loss='categorical_crossentropy')
    convert_to_tflite(length_model, 
                     os.path.join(args.output_dir, 'length_classifier.tflite'))
    
    # Create and convert captcha models for each length
    for length in range(2, 7):
        model = create_captcha_model(length, args.num_classes)
        model.compile(optimizer='adam', loss='categorical_crossentropy')
        convert_to_tflite(model,
                         os.path.join(args.output_dir, f'captcha_model{length}_best.tflite'))
    
    print(f"Models created successfully in {args.output_dir}")

if __name__ == '__main__':
    main()