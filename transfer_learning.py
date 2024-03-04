import argparse
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from bayesian_torch.models.dnn_to_bnn import dnn_to_bnn, bnn_linear_layer, get_kl_loss

from models import ResNet54
from models import BNN_PRIOR_PARAMS


AUDIOSET_N_CLASS = 527
XFER_N_CLASS     = 10  # Number of classes in the new classification task


def cce(output_dict, target_dict):
    return F.cross_entropy(output_dict['embedding'], target_dict['target'])


def main(args):
    assert torch.cuda.is_available()
    device = torch.device('cuda')

    # Load deterministic model
    deterministic_model = ResNet54(
        sample_rate = args.sample_rate, 
        window_size = args.window_size, 
        hop_size    = args.hop_size, 
        mel_bins    = args.mel_bins, 
        fmin        = args.fmin, 
        fmax        = args.fmax, 
        classes_num = AUDIOSET_N_CLASS
        )
    checkpoint_deterministic = torch.load(args.checkpoint_deterministic, map_location=device)
    deterministic_model.load_state_dict(checkpoint_deterministic['model'])
    # Replace classification head with one for new classification task
    deterministic_model.fc1         = nn.Linear(2048, 2048, bias=True)
    deterministic_model.fc_audioset = nn.Linear(2048, XFER_N_CLASS, bias=True)
    # Freeze all layers except the new classification head
    for name, param in deterministic_model.named_parameters():
        if not (("fc1" in name) or ("fc_audioset" in name)):
            param.requires_grad = False
    deterministic_model.to(device)
    optimizer_deterministic = optim.Adam(deterministic_model.parameters(), lr=0.001)


    # Load flipout model
    flipout_model = ResNet54(
        sample_rate = args.sample_rate, 
        window_size = args.window_size, 
        hop_size    = args.hop_size, 
        mel_bins    = args.mel_bins, 
        fmin        = args.fmin, 
        fmax        = args.fmax, 
        classes_num = AUDIOSET_N_CLASS
        )
    for name, _ in list(flipout_model._modules.items()):
        if not (name == 'spectrogram_extractor' or name == "logmel_extractor" or name == 'spec_augmenter'):
            dnn_to_bnn(flipout_model._modules[name], BNN_PRIOR_PARAMS)
    checkpoint_flipout = torch.load(args.checkpoint_flipout, map_location=device)
    flipout_model.load_state_dict(checkpoint_flipout['model'])
    flipout_model.fc1         = bnn_linear_layer(BNN_PRIOR_PARAMS, flipout_model.fc1        )
    flipout_model.fc_audioset = bnn_linear_layer(BNN_PRIOR_PARAMS, flipout_model.fc_audioset)
    for name, param in flipout_model.named_parameters():
        if not (("fc1" in name) or ("fc_audioset" in name)):
            param.requires_grad = False
    flipout_model.to(device)
    optimizer_flipout = optim.Adam(flipout_model.parameters(), lr=0.001)


    batch_size = 32
    epoch = 0
    N = 1000  # Number of training samples
    while epoch < 5:
        fake_wav_batch = (2*torch.rand(batch_size, 32000)-1).to(device)
        fake_lbl_batch = torch.tensor(np.eye(XFER_N_CLASS)[np.random.choice(XFER_N_CLASS,batch_size)]).to(device)
        
        deterministic_model.train()
        deterministic_predictions = deterministic_model(fake_wav_batch)
        deterministic_loss = cce(deterministic_predictions, {'target': torch.argmax(fake_lbl_batch, dim=1)})
        deterministic_loss.backward()
        optimizer_deterministic.step()
        optimizer_deterministic.zero_grad()

        flipout_model.train()
        flipout_predictions = flipout_model(fake_wav_batch)
        flipout_loss = cce(flipout_predictions, {'target': torch.argmax(fake_lbl_batch, dim=1)})
        kl_loss = (get_kl_loss(flipout_model.fc1) + get_kl_loss(flipout_model.fc_audioset)) * 1/N
        flipout_loss += kl_loss
        flipout_loss.backward()
        optimizer_flipout.step()
        optimizer_flipout.zero_grad()

        epoch += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--checkpoint_deterministic', 
        type=str,   
        required=False,
        help="location of deterministic checkpoint"
        )   
    parser.add_argument(
        '--checkpoint_flipout', 
        type=str,   
        required=False,
        help="location of flipout checkpoint"
        )
    parser.add_argument('--sample_rate', type=int, default=32000)
    parser.add_argument('--window_size', type=int, default=1024)
    parser.add_argument('--hop_size'   , type=int, default=320)
    parser.add_argument('--mel_bins'   , type=int, default=64)
    parser.add_argument('--fmin'       , type=int, default=50)
    parser.add_argument('--fmax'       , type=int, default=14000)

    args = parser.parse_args()

    main(args)


