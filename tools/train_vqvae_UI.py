import os, sys
# import yaml
import argparse
import random
import numpy as np
from tqdm import tqdm
import torch, torchvision
from torch.utils.data.dataloader import DataLoader
from torch.optim import Adam
from torchvision.utils import make_grid

##### 모듈을 가져오기 위한 세팅 #####
# PyInstaller로 패키징된 경우 임시 폴더 경로를 추가
if hasattr(sys, '_MEIPASS'):
    try:
        sys.path.append(os.path.join(sys._MEIPASS, 'models'))
        sys.path.append(os.path.join(sys._MEIPASS, 'dataset'))
        print("sys._MEIPASS (train_vqvae): ", sys._MEIPASS)
    except Exception as e:
        print(f"train_vqvae_UI.py - An error occurred while setting the working directory: {e}")
        os.chdir(os.getcwd())

    # 실제 파일 시스템 경로
    dir_root = os.getcwd()

# 개발 환경에서의 일반적인 경로 설정
else:
    dir_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(os.path.join(dir_root))
#############################################
from models.vqvae import VQVAE
from models.lpips import LPIPS
from models.discriminator import Discriminator
from dataset.Asdf_dataset_coord import AsdfDataset
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(config, stop_flag, progress_callback=None):
    # # Read the config file #
    # with open(args.config_path, 'r') as file:
    # with open(r"G:\project\genAI\stable_diffusion_from_scratch\StableDiffusion-PyTorch\config\Asdf_text_cond_coord.yaml", 'r') as file:
    #     try:
    #         config = yaml.safe_load(file)
    #     except yaml.YAMLError as exc:
    #         print(exc)
    # print(config)

    dataset_config = config['dataset_params']
    autoencoder_config = config['autoencoder_params']
    train_config = config['train_params']
    
    # Set the desired seed value #
    seed = train_config['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)
    #############################
    ##### 여기서 아래 경고 발생
    # QBasicTimer::start: Timers cannot be started from another thread
    # QObject::connect: Cannot queue arguments of type 'QItemSelection'
    # (Make sure 'QItemSelection' is registered using qRegisterMetaType().)

    # Create the model and dataset
    model = VQVAE(im_channels=dataset_config['im_channels'],
                  model_config=autoencoder_config).to(device)

    # Create the dataset
    im_dataset_cls = {
        'Asdf': AsdfDataset,
    }.get(dataset_config['name'])
    
    im_dataset = im_dataset_cls(split='train',
                                im_path=dataset_config['im_path'],
                                im_size=dataset_config['im_size'],
                                im_channels=dataset_config['im_channels'])
                                #condition_config=config['ldm_params']['condition_config'])
    
    data_loader = DataLoader(im_dataset,
                             batch_size=train_config['autoencoder_batch_size'],
                             shuffle=True)
    
    # Create output directories
    if not os.path.exists(train_config['task_name']):
        os.mkdir(train_config['task_name'])
        
    num_epochs = train_config['autoencoder_epochs']

    # L1/L2 loss for Reconstruction
    recon_criterion = torch.nn.MSELoss()
    # Disc Loss can even be BCEWithLogits
    disc_criterion = torch.nn.MSELoss()
    
    # No need to freeze lpips as lpips.py takes care of that
    lpips_model = LPIPS().eval().to(device)
    discriminator = Discriminator(im_channels=dataset_config['im_channels']).to(device)
    
    optimizer_d = Adam(discriminator.parameters(), lr=train_config['autoencoder_lr'], betas=(0.5, 0.999))
    optimizer_g = Adam(model.parameters(), lr=train_config['autoencoder_lr'], betas=(0.5, 0.999))
    
    disc_step_start = train_config['disc_start']
    step_count = 0
    
    # This is for accumulating gradients incase the images are huge
    # And one cant afford higher batch sizes
    acc_steps = train_config['autoencoder_acc_steps']
    image_save_steps = train_config['autoencoder_img_save_steps']
    img_save_count = 0

    # 중간 학습 재개를 위한 가중치 로드 코드
    continue_training = train_config['continue_training_vae']
    continue_epoch = train_config['continue_epoch_vae']
    
    if continue_training:
        print(f'VAE training - continue_epoch is enabled')
        vqvae_ckpt_path = os.path.join(train_config['task_name'], train_config['vqvae_autoencoder_ckpt_name'])
        discriminator_ckpt_path = os.path.join(train_config['task_name'], train_config['vqvae_discriminator_ckpt_name'])
        if os.path.exists(vqvae_ckpt_path) and os.path.exists(discriminator_ckpt_path):
            print("Loading existing model and discriminator checkpoints")
            model.load_state_dict(torch.load(vqvae_ckpt_path, map_location=device))
            discriminator.load_state_dict(torch.load(discriminator_ckpt_path, map_location=device))
            start_epoch = continue_epoch
        else:
            print(f"Checkpoint not found. Starting from scratch.")
            start_epoch = 0
    else:
        start_epoch = 0

    print(f'VAE training - start epoch: {start_epoch}')

    model.train()


    for epoch_idx in range(start_epoch, num_epochs + 1):
        recon_losses = []
        codebook_losses = []
        #commitment_losses = []
        perceptual_losses = []
        disc_losses = []
        gen_losses = []
        losses = []
        
        optimizer_g.zero_grad()
        optimizer_d.zero_grad()
        
        for im in tqdm(data_loader):
            step_count += 1
            im = im.float().to(device)
            
            # Fetch autoencoders output(reconstructions)
            model_output = model(im)
            output, z, quantize_losses = model_output
            
            # Image Saving Logic
            if step_count % image_save_steps == 0 or step_count == 1:
                sample_size = min(8, im.shape[0])
                save_output = torch.clamp(output[:sample_size], -1., 1.).detach().cpu()
                save_output = ((save_output + 1) / 2)
                save_input = ((im[:sample_size] + 1) / 2).detach().cpu()
                
                grid = make_grid(torch.cat([save_input, save_output], dim=0), nrow=sample_size)
                img = torchvision.transforms.ToPILImage()(grid)
                if not os.path.exists(os.path.join(train_config['task_name'],'vqvae_autoencoder_samples')):
                    os.mkdir(os.path.join(train_config['task_name'], 'vqvae_autoencoder_samples'))
                img.save(os.path.join(train_config['task_name'],'vqvae_autoencoder_samples',
                                      'current_autoencoder_sample_{}.png'.format(img_save_count)))
                img_save_count += 1
                img.close()
            
            ######### Optimize Generator ##########
            # L2 Loss
            recon_loss = recon_criterion(output, im) 
            recon_losses.append(recon_loss.item())
            recon_loss = recon_loss / acc_steps
            g_loss = (recon_loss +
                      (train_config['codebook_weight'] * quantize_losses['codebook_loss'] / acc_steps) +
                      (train_config['commitment_beta'] * quantize_losses['commitment_loss'] / acc_steps))
            codebook_losses.append(train_config['codebook_weight'] * quantize_losses['codebook_loss'].item())
            # Adversarial loss only if disc_step_start steps passed
            if step_count > disc_step_start:
                disc_fake_pred = discriminator(model_output[0])
                disc_fake_loss = disc_criterion(disc_fake_pred,
                                                torch.ones(disc_fake_pred.shape,
                                                           device=disc_fake_pred.device))
                gen_losses.append(train_config['disc_weight'] * disc_fake_loss.item())
                g_loss += train_config['disc_weight'] * disc_fake_loss / acc_steps
            lpips_loss = torch.mean(lpips_model(output, im)) / acc_steps
            perceptual_losses.append(train_config['perceptual_weight'] * lpips_loss.item())
            g_loss += train_config['perceptual_weight']*lpips_loss / acc_steps
            losses.append(g_loss.item())
            g_loss.backward()
            #####################################
            
            ######### Optimize Discriminator #######
            if step_count > disc_step_start:
                fake = output
                disc_fake_pred = discriminator(fake.detach())
                disc_real_pred = discriminator(im)
                disc_fake_loss = disc_criterion(disc_fake_pred,
                                                torch.zeros(disc_fake_pred.shape,
                                                            device=disc_fake_pred.device))
                disc_real_loss = disc_criterion(disc_real_pred,
                                                torch.ones(disc_real_pred.shape,
                                                           device=disc_real_pred.device))
                disc_loss = train_config['disc_weight'] * (disc_fake_loss + disc_real_loss) / 2
                disc_losses.append(disc_loss.item())
                disc_loss = disc_loss / acc_steps
                disc_loss.backward()
                if step_count % acc_steps == 0:
                    optimizer_d.step()
                    optimizer_d.zero_grad()
            #####################################
            
            if step_count % acc_steps == 0:
                optimizer_g.step()
                optimizer_g.zero_grad()

            # ****** 학습 중지 플래그 확인 ******
            if stop_flag():  # stop_flag가 True이면 학습 중단
                print("학습 중지 요청됨. 학습을 종료합니다.")
                release_cuda(model=model, 
                             lpips_model=lpips_model, 
                             discriminator=discriminator, 
                             optimizer_d=optimizer_d, 
                             optimizer_g=optimizer_g)
                return
        optimizer_d.step()
        optimizer_d.zero_grad()
        optimizer_g.step()
        optimizer_g.zero_grad()
        if len(disc_losses) > 0:
            print(
                'Finished epoch: {} | Recon Loss : {:.4f} | Perceptual Loss : {:.4f} | '
                'Codebook : {:.4f} | G Loss : {:.4f} | D Loss {:.4f}'.
                format(epoch_idx + 1,
                       np.mean(recon_losses),
                       np.mean(perceptual_losses),
                       np.mean(codebook_losses),
                       np.mean(gen_losses),
                       np.mean(disc_losses)))
        else:
            print('Finished epoch: {} | Recon Loss : {:.4f} | Perceptual Loss : {:.4f} | Codebook : {:.4f}'.
                  format(epoch_idx + 1,
                         np.mean(recon_losses),
                         np.mean(perceptual_losses),
                         np.mean(codebook_losses)))
        
        torch.save(model.state_dict(), os.path.join(train_config['task_name'],
                                                    train_config['vqvae_autoencoder_ckpt_name']))
        torch.save(discriminator.state_dict(), os.path.join(train_config['task_name'],
                                                            train_config['vqvae_discriminator_ckpt_name']))
        
        # progress bar 업데이트
        if progress_callback:
            progress_callback(epoch_idx + 1)

    # 학습 완료 이후 model을 VRAM으로부터 release
    print('Releasing model from GPU')
    
    release_cuda(model=model, 
                 lpips_model=lpips_model, 
                 discriminator=discriminator,
                 optimizer_d=optimizer_d,
                 optimizer_g=optimizer_g)
    ############ VRAM release 부분 #########
    
    print('Done Training...')


def release_cuda(model, lpips_model, discriminator, optimizer_d=None, optimizer_g=None):
    """
    VAE 학습 cuda 점유 객체 release
     - optimizer는 모델 파라미터와 연관된 state를 가지고 있으므로 해제 필요
    """
    optimizer_d = None
    optimizer_g = None

    # 모델과 관련된 모든 객체들을 해제해야 합니다.
    model.to('cpu')           # 모델을 CPU로 이동
    model = None              # 모델 객체 해제

    lpips_model.to('cpu')     # LPIPS 모델을 CPU로 이동
    lpips_model = None        # LPIPS 모델 객체 해제

    discriminator.to('cpu')   # 판별기를 CPU로 이동
    discriminator = None      # 판별기 객체 해제
    torch.cuda.empty_cache()

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for vq vae training')
    parser.add_argument('--config', dest='config_path',
                        default='config/mnist.yaml', type=str)
    args = parser.parse_args()
    train(args)
