#! usr/bin/env python

import os
import random

# 파일들이 있는 경로
directory = r'G:\project\genAI\stable_diffusion_from_scratch\StableDiffusion-PyTorch\data\test\caption'

# 불량명 리스트
defects = ['defect_a', 'defect_b', 'defect_c']

# 경로 내의 모든 .txt 파일에 대해 작업 수행
for filename in os.listdir(directory):
    if filename.endswith('.txt'):
        file_path = os.path.join(directory, filename)

        # 숫자1과 숫자2를 0~128 사이에서 랜덤으로 생성
        number1 = random.randint(0, 128)
        number2 = random.randint(0, 128)

        # 불량명을 랜덤으로 선택
        defect_name = random.choice(defects)

        # 새 내용 작성
        new_content = f"{number1}, {number2}, {defect_name}"

        # 파일을 열고 새로운 내용을 덮어쓰기
        with open(file_path, 'w') as file:
            file.write(new_content)

        print(f"Updated file: {filename} with content: {new_content}")

print("All files updated successfully.")
