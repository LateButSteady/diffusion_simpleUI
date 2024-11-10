#! usr/bin/env python

##### Diffusion 이미지 데이터를 Generic하게 만들기

import cv2
import numpy as np
import os
import random
import matplotlib.pyplot as plt
from tqdm import tqdm

dir_root = os.path.join(os.path.dirname(__file__), 'generic_data')

############# config ##############
sz_resize = 2048
sz_object = int(sz_resize / 2)
sz_quadrant = int(sz_object / 2)
sz_crop = 128
num_img = 10000
add_defect = False
weight_object = 0.8
gen_data = True
############# 이미지 로드 #############
#img1 = cv2.imread(os.path.join(dir_root, 'pattern.png'))
#sz_resize = 1024
img1 = cv2.imread(os.path.join(dir_root, 'flower_1920.jpg'))
img2 = cv2.imread(os.path.join(dir_root, 'background.png'))


############# 전처리 #############
# img1을 1024x1024로 리사이즈 (interpolation은 INTER_LINEAR 사용)
img1_resized = cv2.resize(img1, (sz_resize, sz_resize), interpolation=cv2.INTER_LINEAR)
img2_resized = cv2.resize(img2, (sz_object, sz_object), interpolation=cv2.INTER_LINEAR)

img1_resized = img1_resized[:sz_object, :sz_object]


######## img, caption 데이터 생성
if gen_data:
    # img2를 중앙에서 1024x1024로 크롭
    # height2, width2 = img2.shape[:2]
    # start_x = (width2 - 1024) // 2
    # start_y = (height2 - 1024) // 2
    # img2_cropped = img2[start_y:start_y + 1024, start_x:start_x + 1024]

    # img2의 값을 1/2로 줄임 (투명도 적용을 위한 효과)
    # img2_half = img2_cropped

    # img1과 img2_half를 합성 (단, 합산된 값이 255를 넘지 않도록 클리핑)
    img1_resized_scaled = (img1_resized * weight_object).astype(np.uint8)
    img2_resized_scaled = (img2_resized * (1 - weight_object)).astype(np.uint8)


    img = cv2.add(img1_resized_scaled, img2_resized_scaled)
    # plt.imshow(img)
    # plt.show()

    # 저장 경로 설정
    img_save_path = os.path.join(dir_root, 'img')
    caption_save_path = os.path.join(dir_root, 'caption')
    os.makedirs(img_save_path, exist_ok=True)
    os.makedirs(caption_save_path, exist_ok=True)

    list_coords_x = []
    list_coords_y = []

    # subsampling 랜덤 위치
    for i in tqdm(range(1, num_img + 1)):
        # 랜덤 좌표 생성
        x = random.randint(0, sz_object - sz_crop)
        y = random.randint(0, sz_object - sz_crop)
        
        list_coords_x.append(x)
        list_coords_y.append(y)

        # 128x128 크기로 이미지 잘라내기
        sub_img = img[y: y+sz_crop, x: x+sz_crop]
        
        # 이미지 저장
        img_filename = os.path.join(img_save_path, f'{i}.bmp')
        cv2.imwrite(img_filename, sub_img)
        
        if add_defect:
            # 사분면 정보 계산
            if x < sz_quadrant and y < sz_quadrant:
                quadrant = 'first1st'
            elif x >= sz_quadrant and y < sz_quadrant:
                quadrant = 'second2nd'
            elif x < sz_quadrant and y >= sz_quadrant:
                quadrant = 'third3rd'
            else:
                quadrant = 'fourth4th'
            
            # 텍스트 파일에 좌표와 사분면 정보 저장
            caption_filename = os.path.join(caption_save_path, f'{i}.txt')
            with open(caption_filename, 'w') as f:
                f.write(f'({x}, {y}), {quadrant}')
        else:
            # 텍스트 파일에 좌표와 사분면 정보 저장
            caption_filename = os.path.join(caption_save_path, f'{i}.txt')
            with open(caption_filename, 'w') as f:
                f.write(f'({x}, {y}), abcd')
    

    # 좌표 확인
    plt.plot(list_coords_x, list_coords_y, '.')
    plt.show()
########


