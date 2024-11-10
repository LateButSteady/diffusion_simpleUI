#! usr/bin/env python
#-*- encoding: utf-8 -*-

import os, sys
import markdown2
import yaml
import threading    # 학습을 백그라운드에서 실행하기 위함
from concurrent.futures import ThreadPoolExecutor

from PyQt5 import uic
from PyQt5.QtWidgets import QApplication, QWidget, QDialog, QMainWindow, QVBoxLayout, QTextBrowser, QMessageBox, QTextEdit
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt, QMetaObject, Q_ARG, pyqtSlot

##### 모듈을 가져오기 위한 세팅 #####
# PyInstaller로 패키징된 경우 임시 폴더 경로를 추가
if hasattr(sys, '_MEIPASS'):
    try:
        sys.path.append(os.path.join(sys._MEIPASS))
        sys.path.append(os.path.join(sys._MEIPASS, 'tools'))
        print("sys._MEIPASS (sd_ui): ", sys._MEIPASS)
    except Exception as e:
        print(f"[WARN] sd_ui.py - An error occurred while setting the working directory: {e}")
        os.chdir(os.getcwd())

    # 실제 파일 시스템 경로에서 폴더 생성
    dir_root = os.getcwd()

    # 패키징된 실행 파일의 경로로 작업 디렉토리 설정
    exec_dir = os.path.dirname(sys.executable)
    os.chdir(exec_dir)
    print(f"[INFO] Current directory: {exec_dir}")

# 개발 환경에서의 일반적인 경로 설정
else:
    dir_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(os.path.join(dir_root))
#############################################
from tools.train_vqvae_UI import train as run_train_vqvae
from tools.train_ddpm_cond_UI import train as run_train_ddpm
from tools.sample_ddpm_text_cond_UI import infer

# 폴더 준비
config_dir = os.path.join(dir_root, "config")
data_dir = os.path.join(dir_root, "data")

# if not os.path.exists(config_dir):
#     print("Creating config folder")
#     os.mkdir(config_dir)
if not os.path.exists(data_dir):
    print("Creating data folder")
    os.mkdir(data_dir)


def resource_path(relative_path):
    """ PyInstaller가 패키징할 때 리소스 파일의 경로를 찾도록 도와줌 """
    try:
        # PyInstaller로 패키징된 환경에서 실행되는 경우, sys._MEIPASS 경로 사용
        base_path = sys._MEIPASS
    except Exception as e:
        # 개발 환경에서는 현재 파일의 절대 경로를 기준으로 리소스 경로 설정
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)


# 경로 설정
path_icon       = resource_path(os.path.join("UI", "icon.png"))
path_ui         = resource_path(os.path.join("UI", "sd_ui.ui"))
path_help_md    = resource_path(os.path.join("UI", "sd_ui_help.md"))
path_config     = resource_path(os.path.join("config", "config.yaml"))


# UI load
try:
    form_class = uic.loadUiType(path_ui)[0]
except Exception as e:
    print(f"Failed to load UI: {e}")


# 메인 기능 class
class WindowClass(QMainWindow, form_class):

    def __init__(self):
        super().__init__()
        self.setupUi(self)  # designer에서 초기화 시킴
        self.initUI()
        self.loadConfig()

        ##### 오브젝트 초기화 #####
        # 스레드와 중단 플래그
        self.thread_train = None
        self.thread_check_dataset = None
        self.thread_gen = None
        self.stop_training_flag = False
        self.stop_genImg_flag = False

        # 오브젝트 초기화
        self.progressBar_dataset.setValue(0)
        self.progressBar_vae.setValue(0)
        self.progressBar_ddpm.setValue(0)
        self.progressBar_gen.setValue(0)
        self.edit_console.setReadOnly(True)

        # 오브젝트 동작 연결
        self.btn_help.clicked.connect(self.click_help)
        self.btn_checkImgPath.clicked.connect(self.click_checkImgPath)
        self.btn_trainVae.clicked.connect(self.train_VAE_thread)
        self.btn_trainDdpm.clicked.connect(self.train_DDPM_thread)
        self.btn_stopTrainDdpm.clicked.connect(self.stop_train)
        self.btn_stopTrainVae.clicked.connect(self.stop_train)
        self.btn_genImg.clicked.connect(self.gen_img_thread)
        self.btn_stopGenImg.clicked.connect(self.stop_genImg)
        self.checkBox_randomCoord.stateChanged.connect(self.toggle_gen_coord)

        # 언어 전환 버튼 기능 연결
        self.btn_toggleLanguage.clicked.connect(self.toggleLanguage)

        # data 관련
        self.defects = []
        self.coords = []


    # UI custom 설정
    def initUI(self):
        self.setWindowTitle('DiffuGen')
        self.setWindowIcon(QIcon(path_icon))

        # 최상단 표시
        self.raise_()
        self.activateWindow()

        # 임시 초기값 설정
        self.edit_pathImg.setText(r"G:\project\genAI\stable_diffusion_from_scratch\StableDiffusion-PyTorch\generic_data")
        self.edit_taskName.setText("task1_flower")
        self.edit_epochVae.setText("50")
        self.edit_epochDdpm.setText("1")
        self.edit_timeStep.setText("1000")
        self.edit_numGenImg.setText("5")
        self.edit_coordJitterStd.setText("0")
        self.btn_help.setEnabled(True)

    # UI 업데이트가 메인 스레드에서 안전하게 실행되도록 함
    # thread에서 UI 업데이트 발생하는건 좋지 않음
    # chatGPT: 메소드 시그니처 불일치: 
    #   - QMetaObject::invokeMethod는 대상 메소드의 정확한 시그니처(매개변수 포함)를 요구합니다. 
    #   - PyQt에서는 기본적으로 append_to_console과 같은 메소드를 자동으로 슬롯으로 인식하지 않습니다.
    #   => pyqtSlot 데코레이터를 사용하여 append_to_console 메소드를 슬롯으로 등록하면,
    #      메소드가 C++의 시그널-슬롯 시스템과 연결되어 해당 메소드를 정확히 호출할 수 있음
    @pyqtSlot(str)
    def append_to_console(self, message):
        """
        메인 스레드에서 콘솔에 메시지 추가
        """
        QMetaObject.invokeMethod(self.edit_console, "appendPlainText",
                                 Q_ARG(str, message))


    ########### config 파일 로딩 ############
    def loadConfig(self):
        try:
            with open(path_config, 'r') as file:
                self.config = yaml.safe_load(file)
                QMetaObject.invokeMethod(self, "append_to_console", Qt.QueuedConnection, Q_ARG(str, "config 파일 로딩 - OK"))
        except yaml.YAMLError as exc:
            # self.append_to_console("[에러] config 파일 로딩 - Fail")
            QMetaObject.invokeMethod(self, "append_to_console", Qt.QueuedConnection, Q_ARG(str, "[에러] config 파일 로딩 - Fail"))
            QMetaObject.invokeMethod(self, "append_to_console", Qt.QueuedConnection, Q_ARG(str, exc))
            # self.append_to_console(exc)


    ########### 이미지 경로 체크 ############
    # error msg를 console에 출력하기 위해 msg도 return 고려
    def checkImgPath(self):
        """
        dataset 유효성 검사
        """
        QMetaObject.invokeMethod(self, "append_to_console", Qt.QueuedConnection, Q_ARG(str, f"데이터셋 검사 시작..."))

        self.pathImg = self.edit_pathImg.toPlainText()
        self.config["dataset_params"]["im_path"] = self.pathImg
        dir_caption = os.path.join(self.pathImg, "caption")
        dir_img = os.path.join(self.pathImg, "img")

        # caption, img 폴더 유무
        if not os.path.exists(dir_caption) or not os.path.exists(dir_img):
            msg = "[에러] 데이터셋 검사 - Fail: 폴더/파일을 찾을 수 없음"
            QMetaObject.invokeMethod(self, "append_to_console", Qt.QueuedConnection, Q_ARG(str, msg))
            return False#, msg

        # caption 파일들 보면서 불량명 취합 --> dropdown 업데이트
        # 불량명 combobox 업데이트
        QMetaObject.invokeMethod(self, "append_to_console", Qt.QueuedConnection, Q_ARG(str, f"불량명 취합 시작..."))
        self.get_caption_info()
        if not self.defects:
            msg = "[에러] 불량명 취합 - Fail: 불량명을 찾을 수 없음"
            QMetaObject.invokeMethod(self, "append_to_console", Qt.QueuedConnection, Q_ARG(str, msg))
            return False#, msg
        
        # 메인 스레드에서 콤보박스 업데이트 실행
        QMetaObject.invokeMethod(self, "update_defect_combobox", Qt.QueuedConnection, Q_ARG(list, self.defects))
        QMetaObject.invokeMethod(self, "append_to_console", Qt.QueuedConnection, Q_ARG(str, "불량명 취합 - OK"))
                
        # TODO caption에는 있는데, img에는 없는 (vice versa) 불일치 파일 체크

        QMetaObject.invokeMethod(self, "append_to_console", Qt.QueuedConnection, Q_ARG(str, f"데이터셋 검사 - OK"))

        return True # error msg를 console에 출력하기 위해 msg도 return 고려


    ########### VAE 학습 실행 (thread 함수) ############
    def train_VAE(self):        

        ##### 2. VAE 학습 영역 검사 #####
        # VAE epoch
        if not self.check_emptyEditText(self.edit_epochVae.toPlainText(), 
                                        msg_dialog="2. VAE Epoch 값을 입력해주세요",
                                        msg_console="[에러] VAE Epoch 값 입력되지 않음"):
            return
        # TODO numeric인지 검사 필요
        self.config["train_params"]["autoencoder_epochs"] = int(self.edit_epochVae.toPlainText())
        

        # VAE progressbar 활성화
        QMetaObject.invokeMethod(self.progressBar_vae, "setEnabled", Q_ARG(bool, True))
        if self.config['train_params']['continue_training_vae']:
            # -1을 해야 progress bar 셈이 맞음
            QMetaObject.invokeMethod(self.progressBar_vae, "setValue", Q_ARG(int, self.config['train_params']['continue_epoch_vae']-1))
        else:
            QMetaObject.invokeMethod(self.progressBar_vae, "setValue", Q_ARG(int, 0))
        QMetaObject.invokeMethod(self.progressBar_vae, "setMaximum", Q_ARG(int, self.config['train_params']['autoencoder_epochs']))

        # VAE 외 영역 비활성화
        QMetaObject.invokeMethod(self.progressBar_ddpm, "setEnabled", Q_ARG(bool, False))
        QMetaObject.invokeMethod(self.progressBar_gen, "setEnabled", Q_ARG(bool, False))
        QMetaObject.invokeMethod(self.btn_checkImgPath, "setEnabled", Q_ARG(bool, False))
        QMetaObject.invokeMethod(self.btn_trainDdpm, "setEnabled", Q_ARG(bool, False))
        QMetaObject.invokeMethod(self.btn_stopTrainDdpm, "setEnabled", Q_ARG(bool, False))
        QMetaObject.invokeMethod(self.btn_genImg, "setEnabled", Q_ARG(bool, False))
        QMetaObject.invokeMethod(self.btn_stopGenImg, "setEnabled", Q_ARG(bool, False))
        # 콤보 활성화
        QMetaObject.invokeMethod(self.combo_defects, "setEnabled", Q_ARG(bool, True))


        ##### VAE 학습 실행 #####
        try:
            ##### 1. 데이터 설정 영역 검사 #####
            # 실패하면 대기 상태로 return
            if not self.prep_dataset_config():
                
                return


            # 메인 스레드에서 콘솔에 메시지를 출력
            # 다른 스레드에서 콘솔 변경을 시도하는건 맞지 않음
            QMetaObject.invokeMethod(self, "append_to_console", Qt.QueuedConnection, Q_ARG(str, "VAE 학습 시작..."))

            def update_progress(epoch):
                """
                학습 진행도 ProgressBar 업데이트
                """
                QMetaObject.invokeMethod(self.progressBar_vae, "setValue", Q_ARG(int, epoch))

            # 학습 함수 호출 시 언제든지 중지할수 있도록 stop_flag 함수 전달
            # progress bar를 사용할 수 있도록 callback 전달
            run_train_vqvae(self.config, self.is_training_stopped, progress_callback=update_progress)

            # ****** 학습 중지 플래그 확인 ******
            if self.stop_training_flag:
                QMetaObject.invokeMethod(self, "append_to_console", Qt.QueuedConnection, Q_ARG(str, "VAE 학습 중지됨."))
            else:
                QMetaObject.invokeMethod(self, "append_to_console", Qt.QueuedConnection, Q_ARG(str, "VAE 학습 완료"))

        except Exception as e:
            self.show_error_dialog_thread("VAE 학습 실패")
            QMetaObject.invokeMethod(self, "append_to_console", Qt.QueuedConnection, Q_ARG(str, f"[에러] VAE 학습 실패: {e}"))
        finally:
            self.thread_train = None
            # 다른 영역 재활성화
            QMetaObject.invokeMethod(self.progressBar_gen, "setEnabled", Q_ARG(bool, False))
            QMetaObject.invokeMethod(self.progressBar_ddpm, "setEnabled", Q_ARG(bool, False))
            QMetaObject.invokeMethod(self.progressBar_vae, "setEnabled", Q_ARG(bool, False))
            QMetaObject.invokeMethod(self.btn_checkImgPath, "setEnabled", Q_ARG(bool, True))
            QMetaObject.invokeMethod(self.btn_trainDdpm, "setEnabled", Q_ARG(bool, True))
            QMetaObject.invokeMethod(self.btn_stopTrainDdpm, "setEnabled", Q_ARG(bool, True))
            QMetaObject.invokeMethod(self.btn_genImg, "setEnabled", Q_ARG(bool, True))
            QMetaObject.invokeMethod(self.btn_stopGenImg, "setEnabled", Q_ARG(bool, True))
            # 콤보 재활성화
            QMetaObject.invokeMethod(self.combo_defects, "setEnabled", Q_ARG(bool, True))


    ############ DDPM 학습 실행 ############
    def train_DDPM(self):

        ##### 3. Diffusion 학습 영역 검사 #####
        # DDPM epoch
        # TODO numeric인지 검사
        if not self.check_emptyEditText(
            self.edit_epochDdpm.toPlainText(), 
            msg_dialog="3. Diffusion Epoch 값을 입력해주세요",
            msg_console="[에러] Diffusion Epoch 값 입력되지 않음"):
            return
        self.config["train_params"]["ldm_epochs"] = int(self.edit_epochDdpm.toPlainText())

        # Time Step
        # TODO numeric인지 검사
        if not self.check_emptyEditText(
            self.edit_timeStep.toPlainText(), 
            msg_dialog="Time Step 값을 입력하세요",
            msg_console="[에러] Time Step 값 입력되지 않음"):
            return
        self.config["diffusion_params"]["num_timesteps"] = int(self.edit_timeStep.toPlainText())

        # Task name
        # TODO numeric인지 검사
        if not self.check_emptyEditText(
            self.edit_taskName.toPlainText(), 
            msg_dialog="Task명을 입력하세요",
            msg_console="[에러] Task명 입력되지 않음"):
            return
        self.config["train_params"]["task_name"] = self.edit_taskName.toPlainText()


        # DDPM progressbar 활성화
        QMetaObject.invokeMethod(self.progressBar_ddpm, "setEnabled", Q_ARG(bool, True))
        QMetaObject.invokeMethod(self.progressBar_ddpm, "setValue", Q_ARG(int, 0))
        QMetaObject.invokeMethod(self.progressBar_ddpm, "setMaximum", Q_ARG(int, self.config['train_params']['ldm_epochs']))

        # DDPM 외 영역 비활성화
        QMetaObject.invokeMethod(self.progressBar_gen, "setEnabled", Q_ARG(bool, False))
        QMetaObject.invokeMethod(self.progressBar_vae, "setEnabled", Q_ARG(bool, False))
        QMetaObject.invokeMethod(self.btn_checkImgPath, "setEnabled", Q_ARG(bool, False))
        QMetaObject.invokeMethod(self.btn_trainVae, "setEnabled", Q_ARG(bool, False))
        QMetaObject.invokeMethod(self.btn_stopTrainVae, "setEnabled", Q_ARG(bool, False))
        QMetaObject.invokeMethod(self.btn_genImg, "setEnabled", Q_ARG(bool, False))
        QMetaObject.invokeMethod(self.btn_stopGenImg, "setEnabled", Q_ARG(bool, False))
        # combo 재활성화
        QMetaObject.invokeMethod(self.progressBar_vae, "setEnabled", Q_ARG(bool, True))


        ##### DDPM 학습 실행 #####
        try:
            ##### 입력 파일 검사 #####
            # VAE ckpt 유무
            if not (os.path.exists(os.path.join(dir_root, self.config["train_params"]["task_name"], "vqvae_autoencoder_ckpt.pth")) and os.path.exists(os.path.join(dir_root, self.config["train_params"]["task_name"], "vqvae_discriminator_ckpt.pth"))) :
                self.show_error_dialog_thread(f"""VAE 모델 파일 일부를 찾을 수 없습니다.
 - {self.config["train_params"]["task_name"]}/vqvae_autoencoder_ckpt.pth
 - {self.config["train_params"]["task_name"]}/vqvae_discriminator_ckpt.pth""")
                QMetaObject.invokeMethod(self, "append_to_console", Qt.QueuedConnection, Q_ARG(str, f"[에러] VAE 모델 파일 없음"))
                return
            
            ##### 1. 데이터 설정 영역 검사 #####
            # 실패하면 대기 상태로 return
            if not self.prep_dataset_config():
                return

            QMetaObject.invokeMethod(self, "append_to_console", Qt.QueuedConnection, Q_ARG(str, f"Diffusion 학습 시작..."))
            
            def update_progress(epoch):
                """
                학습 진행도 ProgressBar 업데이트
                """
                QMetaObject.invokeMethod(self.progressBar_ddpm, "setValue", Q_ARG(int, epoch))

            # 학습 함수 호출 시 언제든지 중지할수 있도록 stop_flag 함수 전달
            run_train_ddpm(self.config, self.is_training_stopped, progress_callback=update_progress)
            
            # ****** 학습 중지 플래그 확인 ******
            if self.stop_training_flag:
                QMetaObject.invokeMethod(self, "append_to_console", Qt.QueuedConnection, Q_ARG(str, f"Diffusion 학습 중지됨."))
            else:
                QMetaObject.invokeMethod(self, "append_to_console", Qt.QueuedConnection, Q_ARG(str, f"Diffusion 학습 완료"))    

        except Exception as e:
            self.show_error_dialog_thread("Diffusion 학습 실패")
            QMetaObject.invokeMethod(self, "append_to_console", Qt.QueuedConnection, Q_ARG(str, f"[에러] 학습 실패: {e}"))                
        finally:
            self.thread_train = None
            # 다른 영역 재활성화
            QMetaObject.invokeMethod(self.progressBar_gen, "setEnabled", Q_ARG(bool, False))
            QMetaObject.invokeMethod(self.progressBar_ddpm, "setEnabled", Q_ARG(bool, False))
            QMetaObject.invokeMethod(self.progressBar_vae, "setEnabled", Q_ARG(bool, False))
            QMetaObject.invokeMethod(self.btn_checkImgPath, "setEnabled", Q_ARG(bool, True))
            QMetaObject.invokeMethod(self.btn_trainVae, "setEnabled", Q_ARG(bool, True))
            QMetaObject.invokeMethod(self.btn_stopTrainVae, "setEnabled", Q_ARG(bool, True))
            QMetaObject.invokeMethod(self.btn_genImg, "setEnabled", Q_ARG(bool, True))
            QMetaObject.invokeMethod(self.btn_stopGenImg, "setEnabled", Q_ARG(bool, True))
            # 콤보 재활성화
            QMetaObject.invokeMethod(self.combo_defects, "setEnabled", Q_ARG(bool, True))


    # 학습 중지 여부를 확인하기 위해 train으로 넘어갈 함수
    def is_training_stopped(self):
        return self.stop_training_flag
    

    def train_DDPM_thread(self):
        # 학습중인데 또 학습 버튼 누르면
        if self.thread_train is not None and self.thread_train.is_alive():
            self.append_to_console("이미 Diffusion 학습이 진행 중입니다.")
            return

        self.stop_training_flag = False
        self.thread_train = threading.Thread(target=self.train_DDPM)
        self.thread_train.start()


    def train_VAE_thread(self):
        # 학습중인데 또 학습 버튼 누르면
        if self.thread_train is not None and self.thread_train.is_alive():
            self.append_to_console("이미 VAE 학습이 진행 중입니다.")
            return

        self.stop_training_flag = False
        self.thread_train = threading.Thread(target=self.train_VAE)
        self.thread_train.start()


    def stop_train(self):
        if self.thread_train is not None and self.thread_train.is_alive():
            self.stop_training_flag = True
            self.append_to_console("학습 중지 요청됨...")
        else:
            self.append_to_console("[에러] 중지할 학습이 없습니다.")


    def gen_img_thread(self):
        # 학습중인데 또 학습 버튼 누르면 안내
        if self.thread_gen is not None and self.thread_gen.is_alive():
            self.append_to_console("이미 이미지 생성이 진행 중입니다.")
            return

        self.stop_genImg_flag = False
        self.thread_gen = threading.Thread(target=self.gen_img)
        self.thread_gen.start()


    def gen_img(self):
        ##### 4. 이미지 생성 영역 검사 #####
        # 생성 이미지 개수
        # TODO numeric인지 검사
        if not self.check_emptyEditText(
            self.edit_numGenImg.toPlainText(), 
            msg_dialog="생성 이미지 개수를 입력해주세요.",
            msg_console="[에러] 생성 이미지 개수 입력되지 않음"):
            return
        self.config["sample_params"]["num_gen_img"] = int(self.edit_numGenImg.toPlainText())

        # 좌표
        # TODO numeric인지 검사
        if not self.check_emptyEditText(
            self.edit_genImgCoordX.toPlainText() and self.edit_genImgCoordY.toPlainText(), # 둘 중 하나라도 빈 str이면 함수 안에서 False
            msg_dialog="생성 좌표 값을 입력하세요",
            msg_console="[에러] 생성 좌표 값 입력되지 않음"):
            return
        self.config["sample_params"]["gen_coord"] = (int(self.edit_genImgCoordX.toPlainText()), int(self.edit_genImgCoordY.toPlainText()))

        # Time Step
        # TODO numeric인지 검사
        if not self.check_emptyEditText(
            self.edit_timeStep.toPlainText(), 
            msg_dialog="Time Step 값을 입력하세요",
            msg_console="[에러] Time Step 값 입력되지 않음"):
            return
        self.config["diffusion_params"]["num_timesteps"] = int(self.edit_timeStep.toPlainText())


        # jittering
        # TODO numeric인지 검사
        if not self.edit_coordJitterStd.toPlainText():
            self.config["sample_params"]["jitter_std"] = 0
        else:
            self.config["sample_params"]["jitter_std"] = int(self.edit_coordJitterStd.toPlainText())

        # defect dropbox
        if not self.check_emptyEditText(
            self.combo_defects.currentText(),
            msg_dialog="생성하려는 불량을 선택하세요",
            msg_console="[에러] 생성 불량 선택되지 않음"):
            return
        self.config["sample_params"]["defect_gen"] = self.combo_defects.currentText()


        # 랜덤 좌표
        if self.checkBox_randomCoord.isChecked():
            self.config["sample_params"]["random_coord"] = True
        else:
            self.config["sample_params"]["random_coord"] = False


        # gen progressbar 활성화
        QMetaObject.invokeMethod(self.progressBar_gen, "setEnabled", Q_ARG(bool, True))
        QMetaObject.invokeMethod(self.progressBar_gen, "setValue", Q_ARG(int, 0))
        QMetaObject.invokeMethod(self.progressBar_gen, "setMaximum", Q_ARG(int, self.config['sample_params']['num_gen_img']))
        # gen 외 영역 비활성화
        QMetaObject.invokeMethod(self.progressBar_ddpm, "setEnabled", Q_ARG(bool, False))
        QMetaObject.invokeMethod(self.progressBar_vae, "setEnabled", Q_ARG(bool, False))
        QMetaObject.invokeMethod(self.btn_checkImgPath, "setEnabled", Q_ARG(bool, False))
        QMetaObject.invokeMethod(self.btn_trainVae, "setEnabled", Q_ARG(bool, False))
        QMetaObject.invokeMethod(self.btn_trainDdpm, "setEnabled", Q_ARG(bool, False))
        QMetaObject.invokeMethod(self.btn_stopTrainVae, "setEnabled", Q_ARG(bool, False))
        QMetaObject.invokeMethod(self.btn_stopTrainDdpm, "setEnabled", Q_ARG(bool, False))
        QMetaObject.invokeMethod(self.btn_genImg, "setEnabled", Q_ARG(bool, False))

        try:
            QMetaObject.invokeMethod(self, "append_to_console", Qt.QueuedConnection, Q_ARG(str, f"이미지 생성 시작..."))

            def update_progress(j):
                """
                학습 진행도 ProgressBar 업데이트
                """
                QMetaObject.invokeMethod(self.progressBar_gen, "setValue", Q_ARG(int, j))

            # 학습 함수 호출 시 언제든지 중지할수 있도록 stop_flag 함수 전달
            # progress bar를 사용할 수 있도록 callback 전달
            infer(self.config, self.is_genImg_stopped, progress_callback=update_progress)
            QMetaObject.invokeMethod(self, "append_to_console", Qt.QueuedConnection, Q_ARG(str, f"이미지 생성 완료"))
        except Exception as e:
            QMetaObject.invokeMethod(self, "append_to_console", Qt.QueuedConnection, Q_ARG(str, f"이미지 생성 - Fail"))
            self.show_error_dialog_thread(f"이미지 생성 - Fail: {e}")
        finally:
            # 다른 영역 재활성화
            QMetaObject.invokeMethod(self.progressBar_gen, "setEnabled", Q_ARG(bool, False))
            QMetaObject.invokeMethod(self.progressBar_ddpm, "setEnabled", Q_ARG(bool, False))
            QMetaObject.invokeMethod(self.progressBar_vae, "setEnabled", Q_ARG(bool, False))
            QMetaObject.invokeMethod(self.btn_checkImgPath, "setEnabled", Q_ARG(bool, True))
            QMetaObject.invokeMethod(self.btn_trainVae, "setEnabled", Q_ARG(bool, True))
            QMetaObject.invokeMethod(self.btn_trainDdpm, "setEnabled", Q_ARG(bool, True))
            QMetaObject.invokeMethod(self.btn_stopTrainVae, "setEnabled", Q_ARG(bool, True))
            QMetaObject.invokeMethod(self.btn_stopTrainDdpm, "setEnabled", Q_ARG(bool, True))
            QMetaObject.invokeMethod(self.btn_genImg, "setEnabled", Q_ARG(bool, True))
            QMetaObject.invokeMethod(self.btn_stopGenImg, "setEnabled", Q_ARG(bool, True))



    # 생성 중지 여부를 확인하기 위해 infer로 넘어갈 함수
    def is_genImg_stopped(self):
        return self.stop_genImg_flag


    def stop_genImg(self):
        if self.thread_gen is not None and self.thread_gen.is_alive():
            self.stop_genImg_flag = True
            self.append_to_console("이미지 생성 중지 요청됨...")
        else:
            self.append_to_console("[에러] 중지할 생성 작업이 없습니다.")


    def check_emptyEditText(self, text: str, msg_dialog: str, msg_console: str) -> bool:
        """
        빈칸 체크
        input
          - text: edit에 입력된 text
          - msg: 빈칸일 경우 다이얼로그 메시지
        output
          - True or False
        """
        if not text:
            # 다이얼로그 팝업
            self.show_error_dialog_thread(msg_dialog)
            # 콘솔 출력
            self.append_to_console(msg_console)
            return False
        else:
            return True



    ########### btn ############
    ########### 입력 이미지 체크 버튼 ############
    def click_checkImgPath(self):
        # 스레드가 이미 실행 중인 경우 방지
        if self.thread_check_dataset is not None and self.thread_check_dataset.is_alive():
            self.append_to_console("[에러] 데이터 체크가 이미 진행 중입니다.")
            return
        
        # 새 스레드로 작업 실행
        self.thread_check_dataset = threading.Thread(target=self.check_img_path_thread)
        self.thread_check_dataset.start()


    def check_img_path_thread(self):
        """
        이미지 thread로 체크
        """
        try:
            
            QMetaObject.invokeMethod(self.progressBar_dataset, "setEnabled", Q_ARG(bool, True))
            # 동작하는동안 버튼 비활성화
            QMetaObject.invokeMethod(self.btn_trainVae, "setEnabled", Q_ARG(bool, False))
            QMetaObject.invokeMethod(self.btn_trainDdpm, "setEnabled", Q_ARG(bool, False))
            QMetaObject.invokeMethod(self.btn_stopTrainVae, "setEnabled", Q_ARG(bool, False))
            QMetaObject.invokeMethod(self.btn_stopTrainDdpm, "setEnabled", Q_ARG(bool, False))
            QMetaObject.invokeMethod(self.btn_genImg, "setEnabled", Q_ARG(bool, False))
            QMetaObject.invokeMethod(self.btn_stopGenImg, "setEnabled", Q_ARG(bool, False))

            # input 값 가져오기
            self.pathImg = self.edit_pathImg.toPlainText()
            self.taskName = self.edit_taskName.toPlainText()

            # empty 체크
            if not self.check_emptyEditText(self.pathImg, 
                                            msg_dialog="이미지 폴더 경로를 입력해주세요",
                                            msg_console="[에러] 이미지 폴더 경로 입력되지 않음"):
                return

            if not self.check_emptyEditText(self.taskName, 
                                            msg_dialog="Output 폴더명을 입력해주세요",
                                            msg_console="[에러] Output 폴더명 입력되지 않음"):
                return
            
            # config 값 업데이트
            self.config["train_params"]["task_name"] = self.edit_taskName.toPlainText()
            self.config["dataset_params"]["im_size"] = int(self.edit_imgWidth.toPlainText())

            # 체크버튼 누르기 전 초기화
            QMetaObject.invokeMethod(self.progressBar_dataset, "setEnabled", Q_ARG(bool, True))
            QMetaObject.invokeMethod(self.progressBar_dataset, "setValue", Q_ARG(int, 0))
            QMetaObject.invokeMethod(self.text_imgPathStatus, "setText", Q_ARG(str, ""))
            QMetaObject.invokeMethod(self.combo_defects, "clear")


            # 데이터 검사
            QMetaObject.invokeMethod(self, "append_to_console", Qt.QueuedConnection, Q_ARG(str, f"이미지 폴더 검사 시작..."))
            if self.checkImgPath():
                self.status_ok0_fail1(0)
                return
            else:
                self.status_ok0_fail1(1)
                return

        except Exception as e:
            print(f"[ERROR] 데이터 체크 - Fail: {e}")
            self.show_error_dialog_thread(msg=f"에러 발생: {e}")
        finally:
            # ProgressBar 비활성화
            QMetaObject.invokeMethod(self.progressBar_dataset, "setEnabled", Q_ARG(bool, False))
            # 버튼 재활성화
            QMetaObject.invokeMethod(self.btn_trainVae, "setEnabled", Q_ARG(bool, True))
            QMetaObject.invokeMethod(self.btn_trainDdpm, "setEnabled", Q_ARG(bool, True))
            QMetaObject.invokeMethod(self.btn_stopTrainVae, "setEnabled", Q_ARG(bool, True))
            QMetaObject.invokeMethod(self.btn_stopTrainDdpm, "setEnabled", Q_ARG(bool, True))
            QMetaObject.invokeMethod(self.btn_genImg, "setEnabled", Q_ARG(bool, True))
            QMetaObject.invokeMethod(self.btn_stopGenImg, "setEnabled", Q_ARG(bool, True))
            # 콤보 재활성화
            QMetaObject.invokeMethod(self.combo_defects, "setEnabled", Q_ARG(bool, True))
        

    def status_ok0_fail1(self, status: int):
        """
        0이면 OK 표시, 1이면 Fail 표시
        """

        if status == 0:
            # 데이터 체크 OK
            QMetaObject.invokeMethod(self.text_imgPathStatus, "setText", Q_ARG(str, "OK"))
            QMetaObject.invokeMethod(self.text_imgPathStatus, "setStyleSheet", Q_ARG(str, "color: green"))
            QMetaObject.invokeMethod(self, "append_to_console", Qt.QueuedConnection, Q_ARG(str, f"이미지 폴더 검사 - OK"))
        else:
            # 데이터 체크 Fail
            QMetaObject.invokeMethod(self.text_imgPathStatus, "setText", Q_ARG(str, "Fail"))
            QMetaObject.invokeMethod(self.text_imgPathStatus, "setStyleSheet", Q_ARG(str, "color: red"))
            QMetaObject.invokeMethod(self, "append_to_console", Qt.QueuedConnection, Q_ARG(str, f"이미지 폴더 검사 - Fail"))


    @pyqtSlot(list)
    def update_defect_combobox(self, defects):
        """ QComboBox에 불량명 리스트를 추가 """
        defects.sort()
        self.combo_defects.clear()
        self.combo_defects.addItems(defects)

        ## 특정 항목을 제거하고 싶을 경우:
        # index = self.combo_defects.findText("defect_name")  # "defect_name" 항목의 인덱스를 찾음
        # if index != -1:  # 항목이 존재할 경우
        #     self.combo_defects.removeItem(index)


    def get_caption_info(self):
        """
        이미지 폴더 경로에서 defect 정보 가져오기
        ver 0.1 - explainingai 버전
          - data/my_data/caption 폴더에서 parsing

        ver 0.2 - diffuser 버전
          - /data/train/dataset_dict.jsonl 파일에서 "text" 키 값 parsing
        """
        ##### ver 0.1
        dir_caption = os.path.join(self.pathImg, "caption")

        files = [os.path.join(dir_caption, f) for f in os.listdir(dir_caption) if os.path.isfile(os.path.join(dir_caption, f))]
        total_files = len(files)

        #  ProgressBar 초기화
        self.progressBar_dataset.setValue(0)
        self.progressBar_dataset.setMaximum(total_files)


        # ThreadPoolExecutor를 사용해 병렬로 파일을 읽음
        with ThreadPoolExecutor() as executor:
            results = []
            coords = []
            for i, result in enumerate(executor.map(self.parse_label, files), 1):
                results.append(result[2])
                coords.append((result[0], result[1]))
                # ProgressBar 업데이트 (메인 스레드에서 안전하게 호출)
                QMetaObject.invokeMethod(self.progressBar_dataset, "setValue", Q_ARG(int, i))

        # 불량 종류 멤버 변수로 등록
        self.defects = list(set(results))

        # config에 caption_info 등록
        self.config["dataset_params"]["caption_defects"] = results
        self.config["dataset_params"]["caption_coords"] = coords
        
        # 메모리 반환
        del results
        del coords

        ##### TODO ver 0.2 - jsonl 파일에서 "text" 키 값 parsing
        

        return 


    def parse_label(self, file_path):
        """
        label 파일 읽고 label parsing
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                # ','로 split 후 세 번째 항목을 가져옴
                content_split = content.replace("(", "").replace(")", "").split(',')
                x = int(content_split[0])
                y = int(content_split[1])
                label = content_split[2].strip()
                return x, y, label
        except Exception as e:
            self.append_to_console(f"Error processing {file_path}: {e}")
            return None


    def prep_dataset_config(self) -> bool:
        """
        1. 데이터 설정 영역 검사하고, 값을 config에 입력

        output
          - True: OK
          - False: NG
        """
        ##### 빈칸 확인 #####
        # 데이터 경로
        if not self.check_emptyEditText(self.edit_pathImg.toPlainText(), 
                                        msg_dialog="이미지 폴더 경로를 입력해주세요",
                                        msg_console="[에러] 이미지 폴더 경로 입력되지 않음"):
            self.status_ok0_fail1(1)
            return False
        self.config["dataset_params"]["im_path"] = self.edit_pathImg.toPlainText()

        # task name
        if not self.check_emptyEditText(self.edit_taskName.toPlainText(), 
                                        msg_dialog="Output 폴더명을 입력해주세요",
                                        msg_console="[에러] Output 폴더명 입력되지 않음"):
            self.status_ok0_fail1(1)
            return False
        
        self.config["train_params"]["task_name"] = self.edit_taskName.toPlainText()

        ##### 데이터 경로 유효성 확인 #####
        # 여기서 아래 에러 발생
        # QObject::setParent: Cannot set parent, new parent is in a different thread
        # QBasicTimer::start: QBasicTimer can only be used with threads started with QThread
        #   => 해결: main_thread에서 업데이트 하도록 변경 (QMetaObject.invokeMethod)
        if not self.checkImgPath():
            self.show_error_dialog_thread(msg="이미지 폴더 경로를 확인하세요")
            QMetaObject.invokeMethod(self, "append_to_console", Qt.QueuedConnection, Q_ARG(str, "[에러] 이미지 폴더 경로 유효하지 않음"))

            self.status_ok0_fail1(1)
            return False

        ##### 유효하다면 데이터 경로 변경점 검사 (기존 값과 비교해서 변경 없는지) #####
        # - 현재 config에 key가 있을때: 기존 값과 edit에서 새로 불러온 다르면 확인
        # - 현재 config에 key가 없을때: edit 값 사용 
        if "im_path" in self.config["dataset_params"]:
            # key는 있는데 현재 edit 칸의 경로 값과 다르면 사용자 확인
            if self.config["dataset_params"]["im_path"] != self.edit_pathImg.toPlainText():
                if (self.show_yes_no_dialog("이미지 폴더 경로 값이 변경된 것 같습니다.\n현재 입력 경로로 학습할까요?")):
                    self.config["dataset_params"]["im_path"] = self.edit_pathImg.toPlainText()
        else:
            self.config["dataset_params"]["im_path"] != self.edit_pathImg.toPlainText()

        self.config["dataset_params"]["im_size"] = int(self.edit_imgWidth.toPlainText())
        
        self.status_ok0_fail1(0)
        return True


    def toggle_gen_coord(self, state):
        """
        랜덤 좌표 체크박스 클릭하면 토글
        """
        if state == Qt.Checked:
            self.edit_genImgCoordX.setDisabled(True)
            self.edit_genImgCoordY.setDisabled(True)
            self.edit_coordJitterStd.setDisabled(True)
        else:
            self.edit_genImgCoordX.setDisabled(False)
            self.edit_genImgCoordY.setDisabled(False)
            self.edit_coordJitterStd.setDisabled(False)



    ########### Help 기능 실행 ############
    def click_help(self):
        try:
            # help.md 파일 읽기
            with open(path_help_md, 'r', encoding='utf-8') as f:
                md_content = f.read()
            
            # 새 창에 마크다운 렌더링하여 표시
            self.help_window = HelpWindow(md_content)
            self.help_window.show()  # exec_() (모달방식-메인UI 동시 조작 불가) 대신 
                                     # show() (비모달방식-메인UI 동시 조작 가능) 사용
        
        except Exception as e:
            self.show_error_dialog_thread(msg=f"도움말 불러오기 실패: {e}")


    def show_error_dialog_thread(self, msg):
        """
        스레드 내에서 안전하게 에러 다이얼로그를 표시하는 함수
        """
        QMetaObject.invokeMethod(self, "show_error_dialog", Qt.QueuedConnection, Q_ARG(str, msg))


    ############ 다이얼로그 기능 ############
    @pyqtSlot(str)
    def show_error_dialog(self, msg):
        """확인 버튼만 있는 에러 다이얼로그"""
        error_dialog = QMessageBox()
        error_dialog.setIcon(QMessageBox.Critical)
        error_dialog.setText(msg)
        #error_dialog.setInformativeText(msg)
        error_dialog.setWindowTitle("에러")
        error_dialog.setStandardButtons(QMessageBox.Ok)
        error_dialog.exec_()



    def show_yes_no_dialog(self, msg: str) -> bool:
        """예/아니오 선택 다이얼로그"""
        yes_no_dialog = QMessageBox()
        yes_no_dialog.setIcon(QMessageBox.Warning)
        yes_no_dialog.setText(msg)
        #yes_no_dialog.setWindowTitle("확인")
        yes_no_dialog.setStandardButtons(QMessageBox.Yes | QMessageBox.No)

        result = yes_no_dialog.exec_()

        if result == QMessageBox.Yes:
            return True
        else:
            return False



    def toggleLanguage(self):
        # 현재 텍스트 상태를 확인
        if self.btn_toggleLanguage.text() == "English":
            # 영어로 변경
            self.btn_toggleLanguage.setText("Korean")
            self.btn_checkImgPath.setText("Check Data")
            self.btn_trainVae.setText("Train")
            self.btn_stopTrainVae.setText("Stop")
            self.btn_trainDdpm.setText("Train")
            self.btn_stopTrainDdpm.setText("Stop")
            self.btn_genImg.setText("Generate")
            self.btn_stopGenImg.setText("Stop")
            self.btn_help.setText("Help")

            # 그룹박스 영어로 변경
            self.groupBox_1.setTitle("1. Data Settings")
            self.groupBox_2.setTitle("2. VAE Training")
            self.groupBox_3.setTitle("3. Diffusion Training")
            self.groupBox_4.setTitle("4. Image Generation")
            self.groupBox_4_1.setTitle("4-1. Embedding Conditions")

            # 라벨 영어로 변경
            self.label_pathimg.setText("Image Folder Path")
            self.label_taskName.setText("Output Folder")
            self.label_dim.setText("Width x Height")
            self.label_imgPathStatus.setText("Data Check Result:")
            self.label_defects.setText("Image Type")
            self.label_numGenImg.setText("No. of Images")
            self.label_genImgCoord.setText("Coordinates")
            self.label_coordJitterStd.setText("Jitter\nStrength")

            # 체크박스 영어로 변경
            self.checkBox_randomCoord.setText("Random")

        else:
            # 한국어로 복원
            self.btn_toggleLanguage.setText("English")
            self.btn_checkImgPath.setText("데이터 체크")
            self.btn_trainVae.setText("학습")
            self.btn_stopTrainVae.setText("중지")
            self.btn_trainDdpm.setText("학습")
            self.btn_stopTrainDdpm.setText("중지")
            self.btn_genImg.setText("생성")
            self.btn_stopGenImg.setText("중지")
            self.btn_help.setText("도움말")

            # 그룹박스 복원
            self.groupBox_1.setTitle("1. 데이터 설정")
            self.groupBox_2.setTitle("2. VAE 학습")
            self.groupBox_3.setTitle("3. Diffusion 학습")
            self.groupBox_4.setTitle("4. 이미지 생성")
            self.groupBox_4_1.setTitle("4-1. 생성 조건 설정 (Embedding)")

            # 라벨 복원
            self.label_pathimg.setText("이미지 폴더 경로")
            self.label_taskName.setText("Output 폴더명")
            self.label_dim.setText("가로 x 세로")
            self.label_imgPathStatus.setText("체크 결과:")
            self.label_defects.setText("생성 이미지 종류")
            self.label_numGenImg.setText("생성 이미지 개수")
            self.label_genImgCoord.setText("생성 좌표")
            self.label_coordJitterStd.setText("Jittering 강도")

            # 체크박스 복원
            self.checkBox_randomCoord.setText("랜덤 좌표")




# 마크다운을 새창에서 표시
class HelpWindow(QDialog):
    def __init__(self, md_content):
        super().__init__()
        self.setWindowTitle("Help")
        self.setGeometry(300, 300, 800, 400)

        # # 마크다운을 표시할 QTextBrowser
        layout = QVBoxLayout()
        self.text_edit = QTextEdit(self)
        self.text_edit.setReadOnly(True)
        layout.addWidget(self.text_edit)
        self.setLayout(layout)
        
        # # 마크다운 콘텐츠를 HTML로 변환하여 표시
        html_content = markdown2.markdown(md_content)
        self.text_edit.setHtml(html_content)


if __name__ == '__main__':
  app = QApplication(sys.argv)
  myWindow = WindowClass()
  myWindow.show()
  sys.exit(app.exec_())
