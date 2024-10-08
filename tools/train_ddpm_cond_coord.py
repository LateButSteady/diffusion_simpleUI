import sys
sys.path.append(r"")
import shutil
import yaml
import argparse
import numpy as np
from tqdm import tqdm
from torch.optim import Adam
# from dataset.mnist_dataset import MnistDataset
# from dataset.celeb_dataset import CelebDataset
from dataset.Asdf_dataset_coord import AsdfDataset
from torch.utils.data import DataLoader
import torchvision
from torchvision.utils import make_grid
from models.unet_cond_base_coord import Unet
from models.vqvae import VQVAE
from scheduler.linear_noise_scheduler import LinearNoiseScheduler
from tools.sample_ddpm_text_cond import sample
from utils.text_utils import *
from utils.config_utils import *
from utils.diffusion_utils import *
import pathlib
import logging

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
path_current_file = os.path.realpath(__file__)

# logger 세팅
logger = logging.getLogger()
logger.setLevel(logger.INFO)
formatter = logging.Formatter("%(asctime)s %(levelname)s:%(message)s")

file_handler = logging.FileHandler(filename=os.path.join(pathlib.Path(path_current_file).parent, "train_log.log"), encoding="utf-8")
file_handler.setLevel(logging.INFO)
file_handler.setFormatter("%(asctime)s %(levelname)s:%(message)s")

console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter("%(asctime)s %(levelname)s:%(message)s")

logger.addHandler(console)
logger.addHandler(file_handler)
logger.info("start")

def train():
    # Read the config file
    # with open(args.config_path, 'r') as file:
    with open(r"G:\project\genAI\stable_diffusion_from_scratch\StableDiffusion-PyTorch\config\Asdf_text_cond_myclass.yaml", 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    # print(config)
    ########################
    
    # Set the desired random seed
    seed = train_config['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)


    diffusion_config = config['diffusion_params']
    dataset_config = config['dataset_params']
    diffusion_model_config = config['ldm_params']
    autoencoder_model_config = config['autoencoder_params']
    train_config = config['train_params']
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
        # 'mnist': MnistDataset,
        # 'celebhq': CelebDataset,
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
    for epoch_idx in range(start_epoch, num_epochs):
        print(f"epoch = {epoch_idx + 1}")
        logger.info(f"epoch = {epoch_idx + 1}")
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
        print('Finished epoch:{} | Loss : {:.4f}'.format(
            epoch_idx + 1,
            np.mean(losses)))
        torch.save(model.state_dict(), path_ddpm_ckpt)

        if epoch_idx % train_config['save_img_every_epoch'] == 0:
            print(f"[INFO] sampling an img at epoch={epoch_idx + 1}")

            # 모델 파일 저장
            shutil.copyfile(path_ddpm_ckpt, os.path.join(train_config['task_name'], f"{epoch_idx}.pth"))

            model.eval()
            with torch.no_grad():

                # 생성 이미지 save경로 설정
                dir_gen_img = os.path.join(train_config['task_name'], 'cond_text_samples')
                if not os.path.exists(dir_gen_img):
                    os.mkdir(dir_gen_img)
                
                # 첫번째 불량
                text_prompt = ['(422, 279), defect_A']
                path_img = os.path.join(dir_gen_img, f'{text_prompt[0]}_ep_{epoch_idx}.bmp')

                print(f"[INFO] Sampling... {text_prompt[0]}")
                sample(model, text_prompt, scheduler, train_config, diffusion_model_config,
                       autoencoder_model_config, diffusion_config, dataset_config,
                       vae, text_tokenizer, text_model, path_img)
                
                # 같은 위치 두번째 불량
                text_prompt = ['(422, 279), defect_B']
                path_img = os.path.join(dir_gen_img, f'{text_prompt[0]}_ep_{epoch_idx}.bmp')

                print(f"[INFO] Sampling... {text_prompt[0]}")
                sample(model, text_prompt, scheduler, train_config, diffusion_model_config,
                       autoencoder_model_config, diffusion_config, dataset_config,
                       vae, text_tokenizer, text_model, path_img)

            model.train()
    print('Done Training ...')


if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='Arguments for ddpm training')
    # parser.add_argument('--config', dest='config_path',
    #                     default='config/celebhq_text_cond_clip.yaml', type=str)
    # args = parser.parse_args()
    # train(args)
    train()