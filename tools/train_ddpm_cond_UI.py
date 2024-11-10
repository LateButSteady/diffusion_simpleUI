import os, sys
import shutil
import yaml
# import argparse
import numpy as np
from tqdm import tqdm
from torch.optim import Adam
from torch.utils.data import DataLoader
# import torchvision
# from torchvision.utils import make_grid

##### 모듈을 가져오기 위한 세팅 #####
# PyInstaller로 패키징된 경우 임시 폴더 경로를 추가
if hasattr(sys, '_MEIPASS'):
    try:
        sys.path.append(os.path.join(sys._MEIPASS, 'models'))
        sys.path.append(os.path.join(sys._MEIPASS, 'scheduler'))
        sys.path.append(os.path.join(sys._MEIPASS, 'utils'))
        sys.path.append(os.path.join(sys._MEIPASS, 'dataset'))
        print("sys._MEIPASS (tran_ddpm): ", sys._MEIPASS)
    except Exception as e:
        print(f"train_ddpm_text_cond_UI.py - An error occurred while setting the working directory: {e}")
        os.chdir(os.getcwd())

    # 실제 파일 시스템 경로
    dir_root = os.getcwd()

# 개발 환경에서의 일반적인 경로 설정
else:
    dir_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(os.path.join(dir_root))
#############################################
from dataset.Asdf_dataset_coord import AsdfDataset
from models.unet_cond_base_coord import Unet
from models.vqvae import VQVAE
from scheduler.linear_noise_scheduler import LinearNoiseScheduler
from tools.sample_ddpm_text_cond_UI import sample
from utils.text_utils import *
from utils.config_utils import *
from utils.diffusion_utils import *
# import pathlib
# import logging

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# path_current_file = os.path.realpath(__file__)

# # logger 세팅
# logger = logging.getLogger()
# logger.setLevel(logger.INFO)
# formatter = logging.Formatter("%(asctime)s %(levelname)s:%(message)s")

# file_handler = logging.FileHandler(filename=os.path.join(pathlib.Path(path_current_file).parent, "train_log.log"), encoding="utf-8")
# file_handler.setLevel(logging.INFO)
# file_handler.setFormatter("%(asctime)s %(levelname)s:%(message)s")

# console = logging.StreamHandler()
# console.setLevel(logging.INFO)
# console.setFormatter("%(asctime)s %(levelname)s:%(message)s")

# logger.addHandler(console)
# logger.addHandler(file_handler)
# logger.info("start")

def train(config, stop_flag=None, progress_callback=None):
    
    diffusion_config = config['diffusion_params']
    dataset_config = config['dataset_params']
    diffusion_model_config = config['ldm_params']
    autoencoder_model_config = config['autoencoder_params']
    train_config = config['train_params']
    sample_config = config['sample_params']
    continue_training = train_config['continue_training']
    continue_epoch = train_config['continue_epoch']

    path_ddpm_ckpt = os.path.join(train_config['task_name'], train_config['ldm_ckpt_name'])
    path_vqvae_ckpt = os.path.join(train_config['task_name'], train_config['vqvae_autoencoder_ckpt_name'])

    # set the desired seed value
    seed = train_config['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)

    # task 폴더 만들기
    if not os.path.exists(train_config['task_name']):
        os.mkdir(train_config['task_name'])

    ########## Create the noise scheduler #############
    scheduler = LinearNoiseScheduler(num_timesteps=diffusion_config['num_timesteps'],
                                     beta_start=diffusion_config['beta_start'],
                                     beta_end=diffusion_config['beta_end'])
    ###############################################
    
    ##### randomized randn img for sample every X epoch ######
    im_size = dataset_config["im_size"] // 2 ** sum(autoencoder_model_config['down_sample'])
    xt = torch.randn((1, autoencoder_model_config['z_channels'], im_size, im_size)).to(device)
    ##########################################################

    # Instantiate Condition related components
    text_tokenizer = None
    text_model = None
    empty_text_embed = None
    condition_types = []
    condition_config = get_config_value(diffusion_model_config, key='condition_config', default_value=None)
    if condition_config is not None:
        assert 'condition_types' in condition_config, \
            "condition type missing in conditioning config"
        condition_types = condition_config['condition_types']
        if 'text' in condition_types:
            validate_text_config(condition_config)
            with torch.no_grad():
                # Load tokenizer and text model based on config
                # Also get empty text representation
                text_tokenizer, text_model = get_tokenizer_and_model(condition_config['text_condition_config']
                                                                     ['text_embed_model'], device=device)
                empty_text_embed = get_text_representation([''], text_tokenizer, text_model, device)
            
    im_dataset_cls = {
        'Asdf': AsdfDataset,
    }.get(dataset_config['name'])
    
    im_dataset = im_dataset_cls(split='train',
                                im_path=dataset_config['im_path'],
                                im_size=dataset_config['im_size'],
                                im_channels=dataset_config['im_channels'],
                                use_latents=True,
                                latent_path=os.path.join(train_config['task_name'],
                                                         train_config['vqvae_latent_dir_name']),
                                condition_config=condition_config)
    
    data_loader = DataLoader(im_dataset,
                             batch_size=train_config['ldm_batch_size'],
                             shuffle=True)
    
    # Instantiate the unet model
    model = Unet(im_channels=autoencoder_model_config['z_channels'],
                 model_config=diffusion_model_config).to(device)
    
    # continue training
    if continue_training:
        if os.path.exists(path_ddpm_ckpt):
            print("Loading existing model checkpoint")
        else:
            print(f"[ERROR] Cannot find the DDPM model file {path_ddpm_ckpt}")
            raise FileNotFoundError 
        model.load_state_dict(torch.load(path_ddpm_ckpt, map_location=device))
        start_epoch = continue_epoch
    else:
        start_epoch = 0

    model.train()
    
    vae = None
    # Load VAE ONLY if latents are not to be saved or some are missing
    if not im_dataset.use_latents:
        print('Loading vqvae model as latents not present')
        vae = VQVAE(im_channels=dataset_config['im_channels'],
                    model_config=autoencoder_model_config).to(device)
        vae.eval()
        # Load vae if found
        if os.path.exists(path_vqvae_ckpt):
            print('Loaded vae checkpoint')
            vae.load_state_dict(torch.load(path_vqvae_ckpt, map_location=device))
        else:
            print(f"[INFO] Expected vae path: {path_vqvae_ckpt}")
            raise Exception('VAE checkpoint not found and use_latents was disabled')
    
    # Specify training parameters
    num_epochs = train_config['ldm_epochs']
    optimizer = Adam(model.parameters(), lr=train_config['ldm_lr'])
    criterion = torch.nn.MSELoss()
    
    # Load vae and freeze parameters ONLY if latents already not saved
    if not im_dataset.use_latents:
        assert vae is not None
        for param in vae.parameters():
            param.requires_grad = False
    
    # Run training
    for epoch_idx in range(start_epoch, num_epochs + 1):
        
        print(f"epoch = {epoch_idx + 1}")
        # logger.info(f"epoch = {epoch_idx + 1}")
        losses = []
        for data in tqdm(data_loader):
            cond_input = None
            if condition_config is not None:
                im, cond_input = data
            else:
                im = data
            optimizer.zero_grad()
            im = im.float().to(device)
            if not im_dataset.use_latents:
                with torch.no_grad():
                    im, _ = vae.encode(im)
                    
            ########### Handling Conditional Input ###########
            if 'text' in condition_types:
                with torch.no_grad():
                    assert 'text' in cond_input, 'Conditioning Type Text but no text conditioning input present'
                    validate_text_config(condition_config)
                    text_condition = get_text_representation(cond_input['text'],
                                                                 text_tokenizer,
                                                                 text_model,
                                                                 device)
                    text_drop_prob = get_config_value(condition_config['text_condition_config'],
                                                      'cond_drop_prob', 0.)
                    text_condition = drop_text_condition(text_condition, im, empty_text_embed, text_drop_prob)
                    cond_input['text'] = text_condition
            if 'image' in condition_types:
                assert 'image' in cond_input, 'Conditioning Type Image but no image conditioning input present'
                validate_image_config(condition_config)
                cond_input_image = cond_input['image'].to(device)
                # Drop condition
                im_drop_prob = get_config_value(condition_config['image_condition_config'],
                                                      'cond_drop_prob', 0.)
                cond_input['image'] = drop_image_condition(cond_input_image, im, im_drop_prob)
            if 'class' in condition_types:
                assert 'class' in cond_input, 'Conditioning Type Class but no class conditioning input present'
                validate_class_config(condition_config)
                class_condition = torch.nn.functional.one_hot(
                    cond_input['class'],
                    condition_config['class_condition_config']['num_classes']).to(device)
                class_drop_prob = get_config_value(condition_config['class_condition_config'],
                                                   'cond_drop_prob', 0.)
                # Drop condition
                cond_input['class'] = drop_class_condition(class_condition, class_drop_prob, im)

            # 좌표 condition (아마도 deprecate 해야할 듯)
            if 'coords' in condition_types:
                assert 'coords' in cond_input, "Conditioning Type Coords but no coords conditioning input presents"
                cond_input['coords'] = torch.tensor(cond_input['coords'], device=device).float()
            ################################################
            
            # Sample random noise
            noise = torch.randn_like(im).to(device)
            
            # Sample timestep
            t = torch.randint(0, diffusion_config['num_timesteps'], (im.shape[0],)).to(device)
            
            # Add noise to images according to timestep
            noisy_im = scheduler.add_noise(im, noise, t)
            noise_pred = model(noisy_im, t, cond_input=cond_input)
            loss = criterion(noise_pred, noise)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()

            # ****** 학습 중지 플래그 확인 ******
            if stop_flag and stop_flag():  # stop_flag가 True이면 학습 중단
                print("학습 중지 요청됨. 학습을 종료합니다.")
                release_cuda(model=model, 
                            optimizer=optimizer, 
                            vae=vae, 
                            text_model=text_model, 
                            empty_text_embed=empty_text_embed, 
                            text_tokenizer=text_tokenizer, 
                            scheduler=scheduler)
                return

        print('Finished epoch:{} | Loss : {:.4f}'.format(
            epoch_idx + 1,
            np.mean(losses)))
        torch.save(model.state_dict(), path_ddpm_ckpt)

        # #### 일정 epoch마다 이미지 샘플링 해서 확인
        # if epoch_idx % train_config['save_img_every_epoch'] == 0:
        #     print(f"[INFO] sampling an img at epoch={epoch_idx + 1}")

        #     # 모델 파일 저장
        #     shutil.copyfile(path_ddpm_ckpt, os.path.join(train_config['task_name'], f"{epoch_idx}.pth"))

        #     model.eval()
        #     with torch.no_grad():

        #         # 생성 이미지 save경로 설정
        #         dir_gen_img = os.path.join(train_config['task_name'], 'cond_text_samples')
        #         if not os.path.exists(dir_gen_img):
        #             os.mkdir(dir_gen_img)
                
                
        #         for iii in range(sample_config["num_gen_img"]):
        #             print(f"sampling cycle: {iii} / {sample_config['num_gen_img']}-------------------- ")

        #             # 첫번째 불량 (1637)
        #             text_prompt = ['(886, 631), abcd']
        #             token = text_prompt[0].replace(" ", "").replace("(", "").replace(")", "").split(",")
        #             path_img = os.path.join(dir_gen_img, f'{token[0]}_{token[1]}_{token[2]}_[{iii}]_ep{epoch_idx}.bmp')
        #             print("path_img: ", path_img)
        #             print(f"[INFO] Sampling... {text_prompt[0]}")
        #             sample(model, text_prompt, scheduler, train_config, diffusion_model_config,
        #                 autoencoder_model_config, diffusion_config, dataset_config,
        #                 vae, text_tokenizer, text_model, path_img, stop_flag)
                    
        #             # 2 (700)
        #             text_prompt = ['(525, 487), abcd']
        #             token = text_prompt[0].replace(" ", "").replace("(", "").replace(")", "").split(",")
        #             path_img = os.path.join(dir_gen_img, f'{token[0]}_{token[1]}_{token[2]}_[{iii}]_ep{epoch_idx}.bmp')
        #             print("path_img: ", path_img)
        #             print(f"[INFO] Sampling... {text_prompt[0]}")
        #             sample(model, text_prompt, scheduler, train_config, diffusion_model_config,
        #                 autoencoder_model_config, diffusion_config, dataset_config,
        #                 vae, text_tokenizer, text_model, path_img, stop_flag)


        #             # 3  (2378)
        #             text_prompt = ['(488, 416), abcd']
        #             token = text_prompt[0].replace(" ", "").replace("(", "").replace(")", "").split(",")
        #             path_img = os.path.join(dir_gen_img, f'{token[0]}_{token[1]}_{token[2]}_[{iii}]_ep{epoch_idx}.bmp')
        #             print("path_img: ", path_img)
        #             print(f"[INFO] Sampling... {text_prompt[0]}")
        #             sample(model, text_prompt, scheduler, train_config, diffusion_model_config,
        #                 autoencoder_model_config, diffusion_config, dataset_config,
        #                 vae, text_tokenizer, text_model, path_img, stop_flag)

        #             # 4
        #             text_prompt = ['(284, 506), abcd']
        #             token = text_prompt[0].replace(" ", "").replace("(", "").replace(")", "").split(",")
        #             path_img = os.path.join(dir_gen_img, f'{token[0]}_{token[1]}_{token[2]}_[{iii}]_ep{epoch_idx}.bmp')
        #             print("path_img: ", path_img)
        #             print(f"[INFO] Sampling... {text_prompt[0]}")
        #             sample(model, text_prompt, scheduler, train_config, diffusion_model_config,
        #                 autoencoder_model_config, diffusion_config, dataset_config,
        #                 vae, text_tokenizer, text_model, path_img, stop_flag)

        #             # 5 (3906)
        #             text_prompt = ['(407, 740), abcd']
        #             token = text_prompt[0].replace(" ", "").replace("(", "").replace(")", "").split(",")
        #             path_img = os.path.join(dir_gen_img, f'{token[0]}_{token[1]}_{token[2]}_[{iii}]_ep{epoch_idx}.bmp')
        #             print("path_img: ", path_img)
        #             print(f"[INFO] Sampling... {text_prompt[0]}")
        #             sample(model, text_prompt, scheduler, train_config, diffusion_model_config,
        #                 autoencoder_model_config, diffusion_config, dataset_config,
        #                 vae, text_tokenizer, text_model, path_img, stop_flag)

        #             # 6 (5328)
        #             text_prompt = ['(633, 545), abcd']
        #             token = text_prompt[0].replace(" ", "").replace("(", "").replace(")", "").split(",")
        #             path_img = os.path.join(dir_gen_img, f'{token[0]}_{token[1]}_{token[2]}_[{iii}]_ep{epoch_idx}.bmp')
        #             print("path_img: ", path_img)
        #             print(f"[INFO] Sampling... {text_prompt[0]}")
        #             sample(model, text_prompt, scheduler, train_config, diffusion_model_config,
        #                 autoencoder_model_config, diffusion_config, dataset_config,
        #                 vae, text_tokenizer, text_model, path_img, stop_flag)

        #             # 7 (6167)
        #             text_prompt = ['(172, 592), abcd']
        #             token = text_prompt[0].replace(" ", "").replace("(", "").replace(")", "").split(",")
        #             path_img = os.path.join(dir_gen_img, f'{token[0]}_{token[1]}_{token[2]}_[{iii}]_ep{epoch_idx}.bmp')
        #             print("path_img: ", path_img)
        #             print(f"[INFO] Sampling... {text_prompt[0]}")
        #             sample(model, text_prompt, scheduler, train_config, diffusion_model_config,
        #                 autoencoder_model_config, diffusion_config, dataset_config,
        #                 vae, text_tokenizer, text_model, path_img, stop_flag)

        #             # 8 (7810)
        #             text_prompt = ['(781, 814), abcd']
        #             token = text_prompt[0].replace(" ", "").replace("(", "").replace(")", "").split(",")
        #             path_img = os.path.join(dir_gen_img, f'{token[0]}_{token[1]}_{token[2]}_[{iii}]_ep{epoch_idx}.bmp')
        #             print("path_img: ", path_img)
        #             print(f"[INFO] Sampling... {text_prompt[0]}")
        #             sample(model, text_prompt, scheduler, train_config, diffusion_model_config,
        #                 autoencoder_model_config, diffusion_config, dataset_config,
        #                 vae, text_tokenizer, text_model, path_img, stop_flag)

        #             # 9 (9302)
        #             text_prompt = ['(697, 800), abcd']
        #             token = text_prompt[0].replace(" ", "").replace("(", "").replace(")", "").split(",")
        #             path_img = os.path.join(dir_gen_img, f'{token[0]}_{token[1]}_{token[2]}_[{iii}]_ep{epoch_idx}.bmp')
        #             print("path_img: ", path_img)
        #             print(f"[INFO] Sampling... {text_prompt[0]}")
        #             sample(model, text_prompt, scheduler, train_config, diffusion_model_config,
        #                 autoencoder_model_config, diffusion_config, dataset_config,
        #                 vae, text_tokenizer, text_model, path_img, stop_flag)

        #             # 10 (9886)
        #             text_prompt = ['(418, 705), abcd']
        #             token = text_prompt[0].replace(" ", "").replace("(", "").replace(")", "").split(",")
        #             path_img = os.path.join(dir_gen_img, f'{token[0]}_{token[1]}_{token[2]}_[{iii}]_ep{epoch_idx}.bmp')
        #             print("path_img: ", path_img)
        #             print(f"[INFO] Sampling... {text_prompt[0]}")
        #             sample(model, text_prompt, scheduler, train_config, diffusion_model_config,
        #                 autoencoder_model_config, diffusion_config, dataset_config,
        #                 vae, text_tokenizer, text_model, path_img, stop_flag)


        #     model.train() # train 모드로 되돌리기

        # progress bar 업데이트
        if progress_callback:
            progress_callback(epoch_idx + 1)
    
    print('Done Training ...')


def release_cuda(model, optimizer, vae, text_model, empty_text_embed, text_tokenizer, scheduler):
    """
    DDPM 학습 cuda 점유 객체 release
    """
    
    # Optimizer 해제
    optimizer = None

    # 모델 관련 객체 해제
    model.to('cpu')           # 모델을 CPU로 이동
    del model                 # 모델 객체 해제

    # VAE 모델 해제 (VAE가 사용되었을 경우)
    if vae is not None:
        vae.to('cpu')         # VAE를 CPU로 이동
        del vae               # VAE 객체 해제

    # 텍스트 모델 해제 (텍스트 모델이 사용되었을 경우)
    if text_model is not None:
        text_model.to('cpu')  # 텍스트 모델을 CPU로 이동
        del text_model        # 텍스트 모델 객체 해제

    # 텍스트 임베딩 및 토크나이저 해제
    del empty_text_embed
    del text_tokenizer

    # Noise scheduler 해제
    del scheduler
    torch.cuda.empty_cache()

    return


if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='Arguments for ddpm training')
    # parser.add_argument('--config', dest='config_path',
    #                     default='config/celebhq_text_cond_clip.yaml', type=str)
    # args = parser.parse_args()
    # train(args)

    # Read the config file
    # with open(args.config_path, 'r') as file:
    dir_now = os.path.dirname(__file__)
    path_config = os.path.join(dir_now, "..", "config", "config.yaml")

    with open(path_config, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    config["sample_params"]["jitter_std"] = 0
    config["train_params"]["ldm_epochs"] = 200
    config["train_params"]["autoencoder_epochs"] = 30
    config["train_params"]["task_name"] = 'task1'
    config["diffusion_params"]["num_timesteps"] = 1000
    # config["dataset_params"]["im_path"] = r'C:\Users\JWKim\Downloads\generic_data'
    config["dataset_params"]["im_path"] = r'G:\project\genAI\stable_diffusion_from_scratch\StableDiffusion-PyTorch\generic_data'
    config["dataset_params"]["im_size"] = 128

    print("[INFO] Completed loading config")
    ########################

    train(config=config)