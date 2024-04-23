import os
import yaml
import wandb
import argparse
import numpy as np
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from train.model import PCAModel
from train.data_loader import PCADataModule
torch.set_default_dtype(torch.float64)


if __name__ == '__main__':
    # Parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', type=str, required=False, default='config.yaml',
                        help='Configuration file containing all model hyperparameters.')
    parser.add_argument('-checkpoint', type=str, required=False, default='/glade/work/btremblay/PROPCA/',
                        help='Path to checkpoint.')
    parser.add_argument('-data', type=str, required=False, default='/glade/work/mmolnar/PCA_inversions/',
                        help='Path to data.')
    parser.add_argument('-name', type=str, required=False, default='PCA',
                        help='Wandb logging instance name.')
    args = parser.parse_args()
    with open(args.config, 'r') as stream:
        config_data = yaml.load(stream, Loader=yaml.SafeLoader)

    # Paths
    checkpoint = args.checkpoint
    data_path = args.data
    wandb_name = args.name

    # Seed: For reproducibility
    seed = config_data['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Initialize data loader
    data_loader = PCADataModule(data_path,
                                num_workers=os.cpu_count() // 2)
    # Generate training/validation/test sets
    data_loader.setup()

    # Initialize model  # TODO: Modify
    output_props = np.load(data_path + "dbase_output_properties.npy", allow_pickle=True)
    model = PCAModel(config_data['n_input'],
                     config_data['n_output'],
                     norm=output_props,
                     n_hidden_layers=config_data['n_hidden_layers'],
                     n_hidden_neurons=config_data['n_hidden_neurons'],
                     lr=config_data['lr'])

    # Initialize logger
    wandb_logger = WandbLogger(entity=config_data['wandb']['entity'],
                               project=config_data['wandb']['project'],
                               group=config_data['wandb']['group'],
                               job_type=config_data['wandb']['job_type'],
                               tags=config_data['wandb']['tags'],
                               name=wandb_name,
                               notes=config_data['wandb']['notes'],
                               config=config_data)

    # Plot callback  # TODO: Modify
    # total_n_valid = len(data_loader.valid_ds)
    # plot_data = [data_loader.valid_ds[i] for i in range(0, total_n_valid, total_n_valid // 4)]
    # plot_images = torch.stack([image for image, eve in plot_data])
    # plot_eve = torch.stack([eve for image, eve in plot_data])
    # eve_wl = np.load(eve_wl, allow_pickle=True)
    # image_callback = ImagePredictionLogger(plot_images, plot_eve, eve_wl, config_data[instrument])

    # Checkpoint callback
    checkpoint_path = os.path.split(checkpoint)[0]
    checkpoint_callback = ModelCheckpoint(dirpath=checkpoint_path,
                                          monitor='valid_loss', mode='min', save_top_k=1,
                                          filename=checkpoint)

    # Initialize trainer
    trainer = Trainer(default_root_dir=checkpoint_path,
                      accelerator="gpu",
                      devices=1 if torch.cuda.is_available() else None,
                      max_epochs=config_data['max_epochs'],
                      callbacks=[checkpoint_callback],
                      logger=wandb_logger,
                      log_every_n_steps=10
                      )

    # Train the model ⚡
    trainer.fit(model, data_loader)

    # Save
    save_dictionary = config_data
    save_dictionary['model'] = model
    full_checkpoint_path = f"{checkpoint}.ckpt"
    torch.save(save_dictionary, full_checkpoint_path)

    # Evaluate on test set
    # Load model from checkpoint
    state = torch.load(full_checkpoint_path)
    model = state['model']
    trainer.test(model, data_loader)

    # Finalize logging
    wandb.finish()
