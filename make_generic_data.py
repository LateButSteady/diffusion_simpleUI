#! usr/bin/env python

##### Script for creating generic data for train Diffusion
import cv2
import numpy as np
import os
import random
import matplotlib.pyplot as plt
from tqdm import tqdm

os.chdir(os.path.dirname(__file__))
dir_root = os.path.join(os.path.dirname(__file__), 'generic_data')

############# config ##############
sz_resize = 1024
sz_object = int(sz_resize / 2)
sz_crop = 128
num_img = 10000
weight_object = 0.8

def main():
    ############# Load images #############
    img1 = cv2.imread(os.path.join(dir_root, 'vine_1920.png'))
    img2 = cv2.imread(os.path.join(dir_root, 'background.png'))

    sz_img1 = img1.shape
    img1 = img1[:int(sz_img1[0]//2), :int(sz_img1[1]//2)]
    
    ############# Preprocess loaded images #############
    # Resize to 1024x1024
    img1_resized = cv2.resize(img1, (sz_resize, sz_resize), interpolation=cv2.INTER_LINEAR)
    img2_resized = cv2.resize(img2, (sz_object, sz_object), interpolation=cv2.INTER_LINEAR)

    img1_resized = img1_resized[:sz_object, :sz_object]


    ######## img, caption 데이터 생성
    # img1과 img2_half를 합성 (단, 합산된 값이 255를 넘지 않도록 클리핑)
    img1_resized_scaled = (img1_resized * weight_object).astype(np.uint8)
    img2_resized_scaled = (img2_resized * (1 - weight_object)).astype(np.uint8)

    img = cv2.add(img1_resized_scaled, img2_resized_scaled)

    # Create directory
    dir_img_save = os.path.join(dir_root, 'img')
    dir_caption_save = os.path.join(dir_root, 'caption')
    os.makedirs(dir_img_save, exist_ok=True)
    os.makedirs(dir_caption_save, exist_ok=True)

    list_coords_x = []
    list_coords_y = []

    # subsampling 랜덤 위치
    for i in tqdm(range(1, num_img + 1)):
        
        # Pick random coordinate
        x = random.randint(0, sz_object - sz_crop)
        y = random.randint(0, sz_object - sz_crop)
        list_coords_x.append(x)
        list_coords_y.append(y)

        # Crop to 128x128
        sub_img = img[y: y+sz_crop, x: x+sz_crop].copy()
        
        # Add a pattern on each image (Assuming defect)
        image_defect, defect = generate_random_shapes_image(sub_img)
        caption_filename = os.path.join(dir_caption_save, f'{i}.txt')
        
        # Write caption
        with open(caption_filename, 'w') as f:
            f.write(f'({x}, {y}), {defect}')
        
        # Write Image
        img_filename = os.path.join(dir_img_save, f'{i}.bmp')
        cv2.imwrite(img_filename, image_defect)

    # Check scatter plot of generated coordinates
    plt.plot(list_coords_x, list_coords_y, '.')
    plt.show()
########


def generate_random_shapes_image(image:np.ndarray, shape_count=1):
    # Create a blank white image
    size = image.shape[0]

    # Define possible shapes
    shapes = ["triangle", "star", "circle", "square"]
    
    # Loop through the shape_count and randomly place shapes
    for _ in range(shape_count):
        shape = random.choice(shapes)
        
        # Random size and rotation
        scale = random.uniform(0.1, 0.2) * size  # size scale
        angle = random.uniform(0, 360)  # rotation angle
        
        # Random position near the center
        center_x = size//2
        center_y = size//2
        
        # Draw the shape
        if shape == "triangle":
            # Calculate vertices for triangle
            pts = np.array([
                [center_x, int(center_y - scale)],
                [int(center_x - scale * np.sin(np.radians(60))), int(center_y + scale * np.cos(np.radians(60)))],
                [int(center_x + scale * np.sin(np.radians(60))), int(center_y + scale * np.cos(np.radians(60)))]
            ], np.int32)
            M = cv2.getRotationMatrix2D((center_x, center_y), angle, 1.0)
            pts = cv2.transform(np.array([pts]), M)[0]
            pts = pts.astype(np.int32)
            cv2.fillPoly(image, [pts], (0, 0, 0))
        
        elif shape == "star":
            # Calculate vertices for a star shape
            pts = []
            for i in range(5):
                angle_deg = i * 144 + angle
                x = int(center_x + scale * np.cos(np.radians(angle_deg)))
                y = int(center_y + scale * np.sin(np.radians(angle_deg)))
                pts.append([x, y])
            pts = np.array(pts, np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(image, [pts], (0, 0, 0))
        
        elif shape == "circle":
            # Draw a filled circle
            cv2.circle(image, (center_x, center_y), int(scale / 2), (0, 0, 0), -1)
        
        elif shape == "square":
            # Calculate vertices for a rotated square
            pts = np.array([
                [int(center_x - scale/2), int(center_y - scale/2)],
                [int(center_x + scale/2), int(center_y - scale/2)],
                [int(center_x + scale/2), int(center_y + scale/2)],
                [int(center_x - scale/2), int(center_y + scale/2)]
            ])

            # Rotation matrix
            M = cv2.getRotationMatrix2D((center_x, center_y), angle, 1.0)
            pts = cv2.transform(np.array([pts]), M)[0]
            pts = pts.astype(np.int32)
            cv2.fillPoly(image, [pts], (0, 0, 0))

    # # Show the image
    # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # plt.axis('off')
    # plt.show()

    return image, shape


if __name__ == "__main__":
    main()
