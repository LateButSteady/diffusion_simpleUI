
## 1. 최소 사양
   - RAM: 16GB
   - GPU: RTX 3060 이상
   - 구동 환경: Windows 10 이상

## 2. python 설치 (3.9 버전 이상)
   - 링크에서 python 설치 파일 다운로드  https://www.python.org/downloads/release/python-3920/
   - python 설치시 경로는 알기 쉬운 경로로 설치 권장 (예: D:\python\python39)

## 3. 가상환경 설치
   - python 3.9 버전 설치
   - virtualenv 설치 후 가상환경 생성
     <br>
     ```
     pip install virtualenv
     virtualenv venv --python=[python 설치 경로]\Scripts\python.exe
     ```
     <br>
   - 현재 폴더에 venv라는 폴더가 만들어짐
   - run_venv.bat 파일 실행시 (venv)로 시작하는 명령프롬프트가 실행되면 성공

## 4. 가상환경 셋업
   - **패키지 설치** (torch, torchvision 제외)
     <br>
     ```
     pip install -r requirements.txt
     ```
     <br>
   - **torch**
     - url 링크 접속: https://download.pytorch.org/whl/torch/
     - torch-1.11.0%2Bcu115-cp39-cp39-win_amd64.whl 파일 다운로드 (기본 다운로드 폴더 추천)
   - **torchvision**
     - url 링크 접속: https://download.pytorch.org/whl/torchvision/
     - torchvision-0.12.0%2Bcu115-cp39-cp39-win_amd64.whl 파일 다운로드 (기본 다운로드 폴더 추천)
   - 다운로드 완료 후 
     <br>
     ```
     pip install torch-1.11.0+cu115-cp39-cp39-win_amd64.whl
     pip install torchvision-0.12.0+cu115-cp39-cp39-win_amd64.whl
     ```
     <br>
   - **tokenizer**
     - url 링크 접속: https://huggingface.co/openai/clip-vit-base-patch16/tree/main
     - Model card 옆에 Files 탭 클릭
     - 모든 파일을 openai/clip-vit-base-patch16 폴더에 다운로드

## 5. Diffusion UI 실행
   - git clone 받은 폴더로 이동
   - sd_ui.py 스크립트 실행
     <br>
     ```
     python -m UI.sd_ui
     ```
     <br>
