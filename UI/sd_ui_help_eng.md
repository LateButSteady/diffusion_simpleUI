## 1. Minimum Requirements
   - RAM: 16GB
   - GPU: RTX 3060 or higher
   - Operating Environment: Windows 10 or higher

## 2. Install Python (version 3.10 or higher)
   - https://www.python.org/downloads/release/python-3100/
   - It is recommended to install Python in an easily accessible path (e.g., D:\python\python310)
   - Remember the [Python installation path]

## 3. Extract DiffuGen zip file and navigate to the folder

## 4. Create a Virtual Environment
   - Install Python 3.10 version
   - Install virtualenv and create a virtual environment
     <br>
     ```
     pip install virtualenv
     virtualenv venv --python=[Python installation path]\Scripts\python.exe
     ```
     <br>
   - A folder named venv will be created in the current directory
   - Run the run_venv.bat file; if a command prompt starting with (venv) appears, the setup is successful

## 5. Virtual Environment Setup
   - **torch**
     - Access URL: https://download.pytorch.org/whl/torch/
     - Download torch-1.11.0+cu115-cp310-cp310-win_amd64.whl file (recommended: default download folder)
   - **torchvision**
     - Access URL: https://download.pytorch.org/whl/torchvision/
     - Download torchvision-0.12.0+cu115-cp310-cp310-win_amd64.whl file (recommended: default download folder)
   - Once the torch and torchvision files are downloaded, run install_packages.bat
     <br>
     ```
     install_packages.bat
     ```
     <br>
   - **tokenizer**
     - Access URL: https://huggingface.co/openai/clip-vit-base-patch16/tree/main
     - Click the Files tab next to Model card
     - Download all files to the openai/clip-vit-base-patch16 folder

## 6. Run DiffuGen
   - Navigate to the DiffuGen folder
   - Run DiffuGen
     <br>
     ```
     python -m UI.sd_ui.pyc
     ```
     <br>
