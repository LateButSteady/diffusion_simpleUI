#! usr/bin/env python

##### Diffusion 이미지 데이터를 Generic하게 만들기

import cv2
import numpy as np
import os
import random
import matplotlib.pyplot as plt
from tqdm import tqdm

dir_root = os.path.join(os.path.dirname(__file__), 'generic_data')

# 이미지 로드
img1 = cv2.imread(os.path.join(dir_root, 'pattern.png'))
img2 = cv2.imread(os.path.join(dir_root, 'background.png'))

# img1을 1024x1024로 리사이즈 (interpolation은 INTER_LINEAR 사용)
img1_resized = cv2.resize(img1, (1024, 1024), interpolation=cv2.INTER_LINEAR)
img2_resized = cv2.resize(img2, (1024, 1024), interpolation=cv2.INTER_LINEAR)

######## img, caption 데이터 생성
if False:
    # img2를 중앙에서 1024x1024로 크롭
    # height2, width2 = img2.shape[:2]
    # start_x = (width2 - 1024) // 2
    # start_y = (height2 - 1024) // 2
    # img2_cropped = img2[start_y:start_y + 1024, start_x:start_x + 1024]

    # img2의 값을 1/2로 줄임 (투명도 적용을 위한 효과)
    # img2_half = img2_cropped

    # img1과 img2_half를 합성 (단, 합산된 값이 255를 넘지 않도록 클리핑)
    img1_resized_scaled = (img1_resized * 0.7).astype(np.uint8)
    img2_resized_scaled = (img2_resized * 0.3).astype(np.uint8)


    img = cv2.add(img1_resized_scaled, img2_resized_scaled)
    # plt.imshow(img)
    # plt.show()

    # 저장 경로 설정
    img_save_path = os.path.join(dir_root, 'img')
    caption_save_path = os.path.join(dir_root, 'caption')
    os.makedirs(img_save_path, exist_ok=True)
    os.makedirs(caption_save_path, exist_ok=True)

    # 5000개의 subsampling (128x128 사이즈, 랜덤 위치)
    for i in tqdm(range(1, 5001)):
        # 랜덤 좌표 생성
        x = random.randint(0, 1024 - 128)
        y = random.randint(0, 1024 - 128)
        
        # 128x128 크기로 이미지 잘라내기
        sub_img = img[y:y+128, x:x+128]
        
        # 이미지 저장
        img_filename = os.path.join(img_save_path, f'{i}.bmp')
        cv2.imwrite(img_filename, sub_img)
        
        # 사분면 정보 계산
        if x < 512 and y < 512:
            quadrant = 'first1st'
        elif x >= 512 and y < 512:
            quadrant = 'second2nd'
        elif x < 512 and y >= 512:
            quadrant = 'third3rd'
        else:
            quadrant = 'fourth4th'
        
        # 텍스트 파일에 좌표와 사분면 정보 저장
        caption_filename = os.path.join(caption_save_path, f'{i}.txt')
        with open(caption_filename, 'w') as f:
            f.write(f'({x}, {y}), {quadrant}')

########

pick_ind = [3831, 653, 3463, 1697, 1047, 284, 3639, 2527, 4389, 901]


##### mask 만들기
mask = np.any(img1_resized <= 200, axis=2).astype(np.uint8)


# plt.imshow(img1_resized)
# plt.show()

# plt.imshow(mask)
# plt.show()

