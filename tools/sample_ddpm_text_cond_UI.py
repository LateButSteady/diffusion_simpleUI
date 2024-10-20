import os, sys
import numpy as np
# import pandas as pd
import torchvision
# import argparse
# import yaml
import os
from torchvision.utils import make_grid
from tqdm import tqdm
import logging
import random
import time

##### 모듈을 가져오기 위한 세팅 #####
# PyInstaller로 패키징된 경우 임시 폴더 경로를 추가
if hasattr(sys, '_MEIPASS'):
    try:
        sys.path.append(os.path.join(sys._MEIPASS, 'models'))
        sys.path.append(os.path.join(sys._MEIPASS, 'scheduler'))
        sys.path.append(os.path.join(sys._MEIPASS, 'utils'))
        print("sys._MEIPASS (sample): ", sys._MEIPASS)
    except Exception as e:
        print(f"[WARN] sample_ddpm_text_cond_UI.py - An error occurred while setting the working directory: {e}")
        os.chdir(os.getcwd())

    # 실제 파일 시스템 경로
    dir_root = os.getcwd()

# 개발 환경에서의 일반적인 경로 설정
else:
    dir_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(os.path.join(dir_root))
#############################################
from models.unet_cond_base_coord import Unet
from models.vqvae import VQVAE
from scheduler.linear_noise_scheduler import LinearNoiseScheduler
from utils.config_utils import *
from utils.text_utils import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# log 설정
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def sample(model, text_prompt, scheduler, train_config, diffusion_model_config,
           autoencoder_model_config, diffusion_config, dataset_config, vae, text_tokenizer, text_model, path_img, stop_flag=None):
    r"""
    Sample stepwise by going backward one timestep at a time.
    We save the x0 predictions
    """
    im_size = dataset_config['im_size'] // 2 ** sum(autoencoder_model_config['down_sample'])
    
    ########### Sample random noise latent ##########
    # For not fixing generation with one sample
    xt = torch.randn((1,
                      autoencoder_model_config['z_channels'],
                      im_size,
                      im_size)).to(device)
    ###############################################
    
    ############ Create Conditional input ###############
    #text_prompt = ['(282, 358), PTN_ERR'] # data로 없는 좌표
    #text_prompt = [str_pick_coord + ', ' + defect_name]
    # neg_prompt = ['']
    # empty_prompt = ['(0, 0), ']

    text_prompt_embed, coord_tensor = get_text_and_coord_representation(text_prompt, text_tokenizer, text_model, device)
    #empty_text_embed, empty_coord   = get_text_and_coord_representation(neg_prompt, text_tokenizer, text_model, device)
    #assert empty_text_embed.shape == text_prompt_embed.shape
    

    # CFG 자유도 없앤다 (neg prompt 관여 안한다)
    # uncond_input = {
    #     'text': empty_text_embed,
    #     'coords': empty_coord
    # }
    cond_input = {
        'text': text_prompt_embed,
        'coords': coord_tensor
    }
    ###############################################
    
    # By default classifier free guidance is disabled
    # Change value in config or change default value here to enable it
    # cf_guidance_scale = get_config_value(train_config, 'cf_guidance_scale', 1.0)
    
    ################# Sampling Loop ########################
    with torch.no_grad():   # 이게 없으면 VRAM 사용량 폭증
        for i in tqdm(reversed(range(diffusion_config['num_timesteps']))):
            if stop_flag():  # stop_flag가 True이면 학습 중단
                release_cuda(model, vae, text_tokenizer, text_model)
                return

            # Get prediction of noise
            #t = (torch.ones((xt.shape[0],)) * i).long().to(device)
            t = torch.full((xt.shape[0],), i, dtype=torch.long, device=device)
            noise_pred_cond = model(xt, t, cond_input)
            
            # CFG 자유도 없앤다 (neg prompt 관여 안한다)
            # if cf_guidance_scale > 1:
            #     noise_pred_uncond = model(xt, t, uncond_input)
            #     noise_pred = noise_pred_uncond + cf_guidance_scale * (noise_pred_cond - noise_pred_uncond)
            # else:
            noise_pred = noise_pred_cond
            
            # Use scheduler to get x0 and xt-1
            xt, x0_pred = scheduler.sample_prev_timestep(xt, noise_pred, t)
            
            # Save x0
            # ims = torch.clamp(xt, -1., 1.).detach().cpu()
            if i == 0:
                # Decode ONLY the final iamge to save time
                ims = vae.decode(xt)
            else:
                ims = x0_pred
        
    ims = torch.clamp(ims, -1., 1.).detach().cpu()
    ims = (ims + 1) / 2
    grid = make_grid(ims, nrow=1)
    img = torchvision.transforms.ToPILImage()(grid)
    
    # if not os.path.exists(os.path.join(train_config['task_name'], 'cond_text_samples')):
    #     os.mkdir(os.path.join(train_config['task_name'], 'cond_text_samples'))
    # dir_gen_img = os.path.join(train_config['task_name'], 'cond_text_samples', defect_name)
    # if not os.path.exists(dir_gen_img):
    #     os.mkdir(dir_gen_img)
    # img.save(os.path.join(train_config['task_name'], 'cond_text_samples', 'x0_{}.png'.format(i)))
    # img.save(os.path.join(dir_gen_img, f'x{img_ind}_{i}_({coord_tensor[0]}_{coord_tensor[1]}).png'))
    # img.close()
    ##############################################################
    img.save(path_img)
    img.close()
    del img


def gaussian_randint(std, n, return_type="np"):
    """
    randint array 만들기
    Input
      - scale: normal dist의 std
      - n: int 뽑는 개수
    """
    randomNums = np.random.normal(scale=std, size=n)
    randomInts = np.round(randomNums).astype('int')

    if return_type == "list":
        return list(randomInts)
    
    return randomInts


def infer(config, stop_flag=None, progress_callback=None):
# def infer(args):
    # Read the config file #
    # with open(args.config_path, 'r') as file:
    # with open(r"G:\project\genAI\stable_diffusion_from_scratch\StableDiffusion-PyTorch\config\A2YK_text_cond_myclass.yaml", 'r') as file:
    #     try:
    #         config = yaml.safe_load(file)
    #     except yaml.YAMLError as exc:
    #         print(exc)
    # print(config)
    ########################
    
    diffusion_config = config['diffusion_params']
    dataset_config = config['dataset_params']
    diffusion_model_config = config['ldm_params']
    autoencoder_model_config = config['autoencoder_params']
    train_config = config['train_params']
    sample_config = config['sample_params']

    #list_defect_gen = sample_config['list_defect_gen']
    defect_gen = sample_config["defect_gen"]
    num_gen_img = sample_config["num_gen_img"]
    #jitter_std = sample_config['jitter_std']
    jitter_std = sample_config["jitter_std"]
    jitter_coord = sample_config['jitter_coord']    # True
    
    #path_csv_fname_offset_class = sample_config['path_csv_fname_offset_class']
    #############################

    # Set the desired seed value
    seed = train_config['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # 폴더 준비
    if not os.path.exists(os.path.join(train_config['task_name'], 'cond_text_samples')):
        print("Creating directory for sampled images")
        os.mkdir(os.path.join(train_config['task_name'], 'cond_text_samples'))
    dir_gen_img = os.path.join(train_config['task_name'], 'cond_text_samples', sample_config["defect_gen"])
    if not os.path.exists(dir_gen_img):
        print(f"Creating directory \"{sample_config['defect_gen']}\"")
        os.mkdir(dir_gen_img)

    ########## Create the noise scheduler #############
    scheduler = LinearNoiseScheduler(num_timesteps=diffusion_config['num_timesteps'],
                                     beta_start=diffusion_config['beta_start'],
                                     beta_end=diffusion_config['beta_end'])
    ###############################################
    
    text_tokenizer = None
    text_model = None
    
    ############# Validate the config #################
    condition_config = get_config_value(diffusion_model_config, key='condition_config', default_value=None)
    assert condition_config is not None, ("This sampling script is for text conditional "
                                          "but no conditioning config found")
    condition_types = get_config_value(condition_config, 'condition_types', [])
    assert 'text' in condition_types, ("This sampling script is for text conditional "
                                        "but no text condition found in config")
    validate_text_config(condition_config)
    ###############################################
    
    ############# Load tokenizer and text model #################
    with torch.no_grad():
        # Load tokenizer and text model based on config
        # Also get empty text representation
        text_tokenizer, text_model = get_tokenizer_and_model(condition_config['text_condition_config']
                                                             ['text_embed_model'], device=device)
    ###############################################

    # ****** 학습 중지 플래그 확인 ******
    if stop_flag():  # stop_flag가 True이면 학습 중단
        print("이미지 생성 중지 요청됨. 종료합니다.")
        release_cuda(None, None, text_tokenizer, text_model)
        return

    ########## Load Unet #############
    model = Unet(im_channels=autoencoder_model_config['z_channels'],
                 model_config=diffusion_model_config).to(device)
    model.eval()
    if os.path.exists(os.path.join(train_config['task_name'],
                                   train_config['ldm_ckpt_name'])):
        print('Loaded unet checkpoint')
        model.load_state_dict(torch.load(os.path.join(train_config['task_name'],
                                                      train_config['ldm_ckpt_name']),
                                                      map_location=device))
    else:
        raise Exception('Model checkpoint "{}" not found'.format(os.path.join(train_config['task_name'],
                                                              train_config['ldm_ckpt_name'])))
    #####################################
    
    # Create output directories
    if not os.path.exists(train_config['task_name']):
        os.mkdir(train_config['task_name'])
    
    ########## Load VQVAE #############
    vae = VQVAE(im_channels=dataset_config['im_channels'],
                model_config=autoencoder_model_config).to(device)
    vae.eval()
    
    # Load vae if found
    if os.path.exists(os.path.join(train_config['task_name'],
                                   train_config['vqvae_autoencoder_ckpt_name'])):
        print('Loaded vae checkpoint')
        vae.load_state_dict(torch.load(os.path.join(train_config['task_name'],
                                                    train_config['vqvae_autoencoder_ckpt_name']),
                                       map_location=device), strict=True)
    else:
        raise Exception('VAE checkpoint {} not found'.format(os.path.join(train_config['task_name'],
                                                                          train_config['vqvae_autoencoder_ckpt_name'])))
    #####################################
    
    #with torch.no_grad():
    dict_list_str_coord_jitter = {}


    #for defect_gen in list_defect_gen:
    ########## sampling loop ###########
    ##### sampling 좌표 정하기
    # 랜덤 체크라면, 좌표 랜덤으로 뽑기
    # 1. 먼저 caption에서 좌표 최대, 최소 구하기
    # 2. 최대, 최소 안에서 이미지 생성 개수만큼 random 좌표 찍기
    if sample_config["random_coord"]:
        print("[INFO] Picking random coord")
        coords = np.array(dataset_config["caption_coords"])
        # 1. 
        x_min = np.min(coords[:, 0], axis=0)
        x_max = np.max(coords[:, 0], axis=0)
        y_min = np.min(coords[:, 1], axis=0)
        y_max = np.max(coords[:, 1], axis=0)
        print(f"x_min, x_max, y_min, y_max = {x_min} {x_max} {y_min} {y_max}")
        # 2. 
        # 랜덤 좌표의 seed는 fix하지 않는걸로
        current_time_ms = int(time.time() * 1000)
        # 밀리초의 마지막 4자리를 추출
        last_4_digits = current_time_ms % 10000
        random.seed(last_4_digits)
        print("Seed changed: ", last_4_digits)

        np_pick_x = random.choices(x_min + np.array(range(x_max - x_min)), k=num_gen_img)
        np_pick_y = random.choices(y_min + np.array(range(y_max - y_min)), k=num_gen_img)
        np_pick_xy = np.column_stack([np_pick_x, np_pick_y])
        print("Completed random pick")
        # seed 다시 복귀
        random.seed(train_config['seed'])
        print("Seed returned: ", last_4_digits)
        # print(np_pick_xy)
    # 랜덤 체크 아니라면
    # 1-1. 만약 jitter == 0이라면 text prompt 대로 sample() 시작
    # 1-2. 만약 jitter != 0이라면 생성 개수만큼 좌표 흔들기
    else:
        # 1-1.
        np_pick_xy = np.array([[sample_config["gen_coord"][0], sample_config["gen_coord"][1]]] * sample_config["num_gen_img"])
        # 1-2. 
        if jitter_std != 0:
            print(f"[INFO] Jittering the enterred coord ({sample_config['gen_coord'][0]}, {sample_config['gen_coord'][1]}) with std {jitter_std}")
            np_x_jitter = np_pick_xy[:, 0] + np.array(gaussian_randint(std=jitter_std, n=num_gen_img, return_type="list"))
            np_y_jitter = np_pick_xy[:, 1] + np.array(gaussian_randint(std=jitter_std, n=num_gen_img, return_type="list"))

            # 음수는 다시 0 이상으로 나올때까지 재시도
            cnt = 0
            while any(np_x_jitter < 0):
                if cnt < 10 :
                    print(f"[INFO] Negative value x coord ind found after adding noise. Retrying jittering")
                    # np_x_jitter = remove_neg(arr_noisy=np_x_jitter, arr_org=np_pick_xy[:, 0], std=jitter_std, n=1)
                    # loc 중심으로 std만큼 벌어진 값으로 size개만큼 뽑는다
                    np_x_jitter = np.random.normal(loc=np_pick_xy[:, 0], scale=jitter_std, size=np_pick_xy[:, 0].shape[0])
                    np_x_jitter = np.round(np_x_jitter).astype('int')
                    cnt += 1
                else:
                    raise ValueError("랜덤 좌표 생성에 실패했습니다.")

            cnt = 0
            while any(np_y_jitter < 0):
                if cnt < 10 :
                    print(f"[INFO] Negative value y coord ind found after adding noise. Retrying jittering")
                    np_y_jitter = np.random.normal(loc=np_pick_xy[:, 1], scale=jitter_std, size=np_pick_xy[:, 1].shape[0])
                    np_y_jitter = np.round(np_y_jitter).astype('int')
                    cnt += 1
                else:
                    raise ValueError("랜덤 좌표 생성에 실패했습니다.")
            
            np_pick_xy = np.column_stack((np_x_jitter, np_y_jitter))
        else:
            print(f"No jittering from the enterred coord ({sample_config['gen_coord'][0]}, {sample_config['gen_coord'][1]})")

    # 2. text prompt 만들어서 dict_list_str_coord_jitter[defect_gen]에 생성 개수만큼 추가
    dict_list_str_coord_jitter[defect_gen] = [f"({np_pick_xy[i, 0]}, {np_pick_xy[i, 1]})" for i in range(num_gen_img)]


    #df_coord = pd.read_csv(path_csv_fname_offset_class)    header = ['fname', 'class', 'x', 'y']
    #df_coord = pd.DataFrame(data={'class': [defect_gen], 'x': [sample_config["gen_coord"][0]], 'y': [sample_config["gen_coord"][1]]})


    for j in range(num_gen_img):
        # ****** 학습 중지 플래그 확인 ******
        if stop_flag():  # stop_flag가 True이면 학습 중단
            print("이미지 생성 중지 요청됨. 종료합니다.")
            release_cuda(model, vae, text_tokenizer, text_model)
            return

        # for defect_gen in list_defect_gen:
        str_pick_coord = dict_list_str_coord_jitter[defect_gen][j]

        # save할 파일명 지정
        text_prompt = [str_pick_coord + ', ' + defect_gen]
        dir_gen_img = os.path.join(train_config['task_name'], 'cond_text_samples', defect_gen)
        if not os.path.exists(dir_gen_img):
            os.mkdir(dir_gen_img)
        path_img = os.path.join(dir_gen_img, f'x[{j}]_({np_pick_xy[j, 0]}_{np_pick_xy[j, 1]})_{defect_gen}.bmp')

        print(f"[INFO] Sampling... {j+1} / {num_gen_img} // {text_prompt[0]} // {train_config['ldm_ckpt_name']}")
        sample(model, text_prompt, scheduler, train_config, diffusion_model_config,
                autoencoder_model_config, diffusion_config, dataset_config,
                #vae, text_tokenizer, text_model, j, defect_gen, fname_pth.replace(".pth", ""), str_pick_coord)
                #vae, text_tokenizer, text_model, j, defect_gen, str_pick_coord) 
                vae, text_tokenizer, text_model, path_img, stop_flag)
        
        # progress bar 업데이트
        if progress_callback:
            progress_callback(j + 1)

    # sampling 끝나면 release
    release_cuda(model, vae, text_tokenizer, text_model)



def release_cuda(model, vae, text_tokenizer, text_model):
    """
    Efficiently releases GPU memory by sending models to CPU and clearing cache
    """
    if model is not None:
        model.to('cpu')
        del model

    if vae is not None:
        vae.to('cpu')
        del vae

    if text_model is not None:
        text_model.to('cpu')
        del text_model

    if text_tokenizer is not None:
        del text_tokenizer

    torch.cuda.empty_cache()

    return


if __name__ == '__main__':
    import yaml
    # Read the config file #
    with open(r"G:\project\genAI\stable_diffusion_from_scratch\StableDiffusion-PyTorch\config\Asdf_text_cond_coord_UI.yaml", 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(f"[ERROR] yaml file load error: {exc}")

    try:
        infer(config)
    except Exception as e:
        logging.critical(f"[ERROR] Unhandled exception: {e}", exc_info=True)
