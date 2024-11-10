import glob
import os, sys
import pickle

import torch
import torchvision
import yaml
from torch.utils.data.dataloader import DataLoader
from torchvision.utils import make_grid
from tqdm import tqdm

dir_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(dir_root))

from dataset.Asdf_dataset_coord import AsdfDataset
from models.vqvae import VQVAE

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def infer(config):
    dataset_config = config['dataset_params']
    autoencoder_config = config['autoencoder_params']
    train_config = config['train_params']
    
    # Create the dataset
    im_dataset_cls = {
        'Asdf': AsdfDataset,
    }.get(dataset_config['name'])
    
    im_dataset = im_dataset_cls(split='train',
                                im_path=dataset_config['im_path'],
                                im_size=dataset_config['im_size'],
                                im_channels=dataset_config['im_channels'])
    
    # This is only used for saving latents. Which as of now
    # is not done in batches hence batch size 1
    data_loader = DataLoader(im_dataset,
                             batch_size=1,
                             shuffle=False)

    num_images = train_config['num_samples']
    ngrid = train_config['num_grid_rows']
    
    idxs = torch.randint(0, len(im_dataset) - 1, (num_images,))
    ims = torch.cat([im_dataset[idx][None, :] for idx in idxs]).float()
    ims = ims.to(device)
    print(f"[INFO] Selected {num_images} images to test VAE")

    model = VQVAE(im_channels=dataset_config['im_channels'],
                  model_config=autoencoder_config).to(device)
    model.load_state_dict(torch.load(os.path.join(train_config['task_name'],
                                                  train_config['vqvae_autoencoder_ckpt_name']),
                          map_location=device))
    print(f"[INFO] Loaded VAE model")

    model.eval()
    
    with torch.no_grad():
        
        encoded_output, _ = model.encode(ims)
        decoded_output = model.decode(encoded_output)
        encoded_output = torch.clamp(encoded_output, -1., 1.)
        encoded_output = (encoded_output + 1) / 2
        decoded_output = torch.clamp(decoded_output, -1., 1.)
        decoded_output = (decoded_output + 1) / 2
        ims = (ims + 1) / 2

        encoder_grid = make_grid(encoded_output.cpu(), nrow=ngrid)
        decoder_grid = make_grid(decoded_output.cpu(), nrow=ngrid)
        input_grid = make_grid(ims.cpu(), nrow=ngrid)
        encoder_grid = torchvision.transforms.ToPILImage()(encoder_grid)
        decoder_grid = torchvision.transforms.ToPILImage()(decoder_grid)
        input_grid = torchvision.transforms.ToPILImage()(input_grid)
        
        input_grid.save(os.path.join(train_config['task_name'], 'input_samples.png'))
        encoder_grid.save(os.path.join(train_config['task_name'], 'encoded_samples.png'))
        decoder_grid.save(os.path.join(train_config['task_name'], 'reconstructed_samples.png'))
        
        print(f"[INFO] Completed reconstructing VAE images")

        if train_config['save_latents']:
            # save Latents (but in a very unoptimized way)
            latent_path = os.path.join(train_config['task_name'], train_config['vqvae_latent_dir_name'])
            latent_fnames = glob.glob(os.path.join(train_config['task_name'], train_config['vqvae_latent_dir_name'],
                                                   '*.pkl'))
            assert len(latent_fnames) == 0, 'Latents already present. Delete all latent files and re-run'
            if not os.path.exists(latent_path):
                os.mkdir(latent_path)
            print('Saving Latents for {}'.format(dataset_config['name']))
            
            fname_latent_map = {}
            part_count = 0
            count = 0
            for idx, im in enumerate(tqdm(data_loader)):
                encoded_output, _ = model.encode(im.float().to(device))
                fname_latent_map[im_dataset.images[idx]] = encoded_output.cpu()
                # Save latents every 1000 images
                if (count+1) % 1000 == 0:
                    pickle.dump(fname_latent_map, open(os.path.join(latent_path,
                                                                    '{}.pkl'.format(part_count)), 'wb'))
                    part_count += 1
                    fname_latent_map = {}
                count += 1
            if len(fname_latent_map) > 0:
                pickle.dump(fname_latent_map, open(os.path.join(latent_path,
                                                   '{}.pkl'.format(part_count)), 'wb'))
            print('Done saving latents')


if __name__ == '__main__':

    dir_now = os.path.dirname(__file__)
    path_config = os.path.join(dir_now, "..", "config", "config.yaml")

    with open(path_config, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    # config["sample_params"]["jitter_std"] = 0
    # config["train_params"]["ldm_epochs"] = 200
    # config["train_params"]["autoencoder_epochs"] = 30
    config["train_params"]["task_name"] = 'task1'
    # config["diffusion_params"]["num_timesteps"] = 1000
    config["dataset_params"]["im_path"] = r'C:\Users\JWKim\Downloads\generic_data'
    config["dataset_params"]["im_size"] = 128

    print("[INFO] Completed loading config")

    # parser = argparse.ArgumentParser(description='Arguments for vq vae inference')
    # parser.add_argument('--config', dest='config_path',
    #                     default='config/mnist.yaml', type=str)
    # args = parser.parse_args()
    # infer(args)
    infer(config)
