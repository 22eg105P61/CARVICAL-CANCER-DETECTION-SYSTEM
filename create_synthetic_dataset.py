"""
Creates synthetic microscopy-like images for testing
This is ONLY for demonstration and testing your application
NOT for actual medical use
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import os
import random

def create_cell_image(is_abnormal=False, img_size=224):
    """
    Creates a synthetic microscopy-like image
    Simulates normal vs abnormal cells with different characteristics
    """
    
    # Create base image with tissue-like background
    img = Image.new('RGB', (img_size, img_size), color=(240, 220, 220))
    draw = ImageDraw.Draw(img)
    
    # Add background texture
    for _ in range(500):
        x = random.randint(0, img_size)
        y = random.randint(0, img_size)
        size = random.randint(1, 3)
        color = (
            random.randint(220, 250),
            random.randint(200, 230),
            random.randint(200, 230)
        )
        draw.ellipse([x, y, x+size, y+size], fill=color)
    
    # Number of cells in image
    num_cells = random.randint(3, 8)
    
    for _ in range(num_cells):
        # Random position
        x = random.randint(30, img_size-30)
        y = random.randint(30, img_size-30)
        
        if is_abnormal:
            # Abnormal cells: irregular, darker, varied sizes
            size = random.randint(15, 35)
            irregularity = random.uniform(0.6, 1.0)
            
            # Nucleus (darker, irregular)
            nucleus_color = (
                random.randint(80, 120),
                random.randint(50, 90),
                random.randint(100, 140)
            )
            
            # Draw irregular nucleus
            for angle in range(0, 360, 30):
                offset = random.randint(-5, 5)
                rad = size * irregularity + offset
                x_offset = int(rad * np.cos(np.radians(angle)))
                y_offset = int(rad * np.sin(np.radians(angle)))
                draw.ellipse(
                    [x-rad+x_offset, y-rad+y_offset, 
                     x+rad+x_offset, y+rad+y_offset],
                    fill=nucleus_color
                )
            
            # Irregular cytoplasm
            cyto_size = size * random.uniform(1.3, 1.8)
            cyto_color = (
                random.randint(180, 200),
                random.randint(150, 180),
                random.randint(180, 210)
            )
            draw.ellipse(
                [x-cyto_size, y-cyto_size, x+cyto_size, y+cyto_size],
                outline=cyto_color, width=2
            )
            
        else:
            # Normal cells: regular, lighter, uniform
            size = random.randint(18, 25)
            
            # Nucleus (light, regular)
            nucleus_color = (
                random.randint(140, 180),
                random.randint(120, 160),
                random.randint(160, 200)
            )
            draw.ellipse(
                [x-size, y-size, x+size, y+size],
                fill=nucleus_color
            )
            
            # Regular cytoplasm
            cyto_size = size * 1.4
            cyto_color = (
                random.randint(200, 220),
                random.randint(180, 210),
                random.randint(200, 230)
            )
            draw.ellipse(
                [x-cyto_size, y-cyto_size, x+cyto_size, y+cyto_size],
                outline=cyto_color, width=2
            )
    
    # Apply slight blur to make it look more realistic
    img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
    
    # Add slight noise
    img_array = np.array(img)
    noise = np.random.randint(-10, 10, img_array.shape, dtype=np.int16)
    img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(img_array)
    
    return img

def create_dataset(train_count=100, val_count=30):
    """
    Creates a synthetic dataset with train and validation splits
    """
    
    # Create directory structure
    dirs = [
        'dataset/train/normal',
        'dataset/train/abnormal',
        'dataset/validation/normal',
        'dataset/validation/abnormal'
    ]
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
    
    print("Creating synthetic dataset...")
    print("="*60)
    
    # Generate training images
    print(f"\nGenerating {train_count} training images per class...")
    for i in range(train_count):
        # Normal cells
        img_normal = create_cell_image(is_abnormal=False)
        img_normal.save(f'dataset/train/normal/normal_{i:04d}.jpg')
        
        # Abnormal cells
        img_abnormal = create_cell_image(is_abnormal=True)
        img_abnormal.save(f'dataset/train/abnormal/abnormal_{i:04d}.jpg')
        
        if (i+1) % 20 == 0:
            print(f"  Progress: {i+1}/{train_count} images generated")
    
    # Generate validation images
    print(f"\nGenerating {val_count} validation images per class...")
    for i in range(val_count):
        # Normal cells
        img_normal = create_cell_image(is_abnormal=False)
        img_normal.save(f'dataset/validation/normal/normal_val_{i:04d}.jpg')
        
        # Abnormal cells
        img_abnormal = create_cell_image(is_abnormal=True)
        img_abnormal.save(f'dataset/validation/abnormal/abnormal_val_{i:04d}.jpg')
    
    print(f"\n✅ Dataset created successfully!")
    print("="*60)
    print(f"\nDataset Summary:")
    print(f"  Training: {train_count*2} images ({train_count} normal + {train_count} abnormal)")
    print(f"  Validation: {val_count*2} images ({val_count} normal + {val_count} abnormal)")
    print(f"\nYou can now run: python train_image_model.py")
    print("="*60)
    
    # Create sample preview
    print("\nCreating preview image...")
    preview = Image.new('RGB', (224*4, 224*2), color='white')
    
    for i in range(4):
        normal = create_cell_image(is_abnormal=False)
        abnormal = create_cell_image(is_abnormal=True)
        preview.paste(normal, (i*224, 0))
        preview.paste(abnormal, (i*224, 224))
    
    preview.save('dataset_preview.jpg')
    print("✅ Preview saved as 'dataset_preview.jpg'")
    print("   (Top row: Normal cells, Bottom row: Abnormal cells)")

if __name__ == "__main__":
    # Create dataset with default sizes
    # Adjust counts as needed: create_dataset(train_count=200, val_count=50)
    create_dataset(train_count=100, val_count=30)