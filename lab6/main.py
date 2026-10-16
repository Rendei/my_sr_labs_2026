# Laboratory work 6. Training speaker embedding extractors based on transformer pretrains
# Run from the ./lab6 directory: python main.py [--exp base|freeze|scratch|mismatch]
#
# Experiments:
#   base     - Whisper tiny encoder is unfrozen, weights are pretrained (section 2 of the lab)
#   freeze   - extra task 2: Whisper tiny encoder is frozen (freeze_feats=True)
#   scratch  - extra task 1: Whisper tiny encoder is trained from random initialization (load_weight=False)
#   mismatch - extra task 3: 400 frames to train, 1000 frames to validate and test
#
# Training is resumable: the latest checkpoint of the experiment is loaded automatically.


# Import of modules
import os
import sys
import glob
import json
import argparse
import itertools

sys.path.append(os.path.realpath('..'))

import torch
from torch.utils.data import DataLoader

from common import test_dataset_loader as test_dataset_loader_verification
from common import extract_features, compute_scores
from common import get_eer
from exercises_blank import train_dataset_loader, test_dataset_loader, train_network, test_network, tar_imp_hists
from exercises_blank import WhisperEncoder, TDNNSimple, StatPoolLayer, MaxoutSegmentLevel, SSLDownstream, MainModel
from whisper_fbanks import ExtractWhisperFbanks80
from LossFunction import AAMSoftmaxLoss
from Optimizer import SGDOptimizer
from Scheduler import OneCycleLRScheduler
from load_save_pth import saveParameters, loadParameters


# Hyperparameters (the same as in the notebook)
nOut = 512
margin = 0.35
scale = 32.0

max_frames_train  = 400
train_path = '../data/voxceleb1_dev/wav'
batch_size_train = 128
num_workers_train = 8
shuffle = True

max_frames_val = 400
val_path = '../data/voxceleb1_dev/wav'
batch_size_val = 128
num_workers_val = 4

max_frames_test = 400
test_path = '../data/voxceleb1_test/wav/'
batch_size_test = 1
num_workers_test = 4
num_eval = 5

lr = 0.0001
weight_decay = 0

val_interval = 5
max_epoch = 40

musan_path = '../data/musan_split'
rir_path = '../data/RIRS_NOISES/simulated_rirs'

# Per-experiment settings
EXPERIMENTS = {
    'base':     dict(load_weight=True,  freeze_feats=False, lr=lr,   frames_eval=400),
    'freeze':   dict(load_weight=True,  freeze_feats=True,  lr=0.01, frames_eval=400),
    'scratch':  dict(load_weight=False, freeze_feats=False, lr=lr,   frames_eval=400),
    'mismatch': dict(load_weight=True,  freeze_feats=False, lr=lr,   frames_eval=1000),
}


def make_lists():
    # Generate train/validation data lists for close-set speaker identification
    train_list = []
    val_list   = []

    with open('../data/voxceleb1_test/iden_split.txt', 'r') as f:
        lines = f.readlines()

    black_list = os.listdir('../data/voxceleb1_test/wav')   # exclude speaker IDs from VoxCeleb1 test set
    num_train_spk = []

    for line in lines:
        line   = line.strip().split(' ')
        spk_id = line[1].split('/')[0]

        if not (spk_id in black_list):
            num_train_spk.append(spk_id)

        else:
            continue

        if (line[0] == '1'):
            train_list.append(' '.join([spk_id, line[1]]))

        elif (line[0] == '2'):
            val_list.append(' '.join([spk_id, line[1]]))

    return train_list, val_list, len(set(num_train_spk))


def latest_checkpoint(path):
    files = sorted(glob.glob(os.path.join(path, 'lab6_model_*.pth')))

    return files[-1] if files else None


def main(args):
    cfg = EXPERIMENTS[args.exp]
    model_dir = '../data/lab6_models' if args.exp == 'base' else '../data/lab6_models_' + args.exp
    results_path = 'results_%s.json' % args.exp

    frames_val  = cfg['frames_eval']
    frames_test = cfg['frames_eval']
    epochs      = args.epochs

    train_list, val_list, num_train_spk = make_lists()
    print('Experiment %s: %d train, %d val files, %d speakers' % (args.exp, len(train_list), len(val_list), num_train_spk), flush=True)

    train_dataset = train_dataset_loader(train_list=train_list,
                                         max_frames=max_frames_train,
                                         train_path=train_path,
                                         augment=True,
                                         musan_path=musan_path,
                                         rir_path=rir_path)
    train_loader  = DataLoader(train_dataset, batch_size=batch_size_train, pin_memory=False,
                               num_workers=num_workers_train, shuffle=shuffle, persistent_workers=True)

    val_dataset = test_dataset_loader(test_list=val_list, max_frames=frames_val, test_path=val_path)
    val_loader  = DataLoader(val_dataset, batch_size=batch_size_val, num_workers=num_workers_val, persistent_workers=True)

    with open('../data/voxceleb1_test/veri_test2.txt', 'r') as f:
        verif_protocol_lines = f.readlines()

    files = list(itertools.chain(*[x.strip().split()[-2:] for x in verif_protocol_lines]))
    setfiles = sorted(set(files))

    test_dataset = test_dataset_loader_verification(setfiles, test_path=test_path, eval_frames=frames_test, num_eval=num_eval)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size_test, shuffle=False,
                              num_workers=num_workers_test, drop_last=False, sampler=None)

    preprocessor        = ExtractWhisperFbanks80(pad_data=False)
    feat_extractor      = WhisperEncoder.build_model(model_name='tiny', load_weight=cfg['load_weight'], n_layer=-1)
    frame_level_block   = TDNNSimple(n_layers=1, in_ch=384, out_ch=1024)
    pooling_block       = StatPoolLayer.by_string_mode(mode='MV')
    segment_level_block = MaxoutSegmentLevel(input_dim=2048, output_dim=nOut, enable_batch_norm=True)

    whisper_sr_model    = SSLDownstream(preproc=preprocessor,
                                        feat_extractor=feat_extractor,
                                        frame_level_block=frame_level_block,
                                        pooling_block=pooling_block,
                                        segment_level_block=segment_level_block,
                                        freeze_feats=cfg['freeze_feats'])
    trainfunc           = AAMSoftmaxLoss(nOut=nOut, nClasses=num_train_spk, margin=margin, scale=scale)
    main_model          = MainModel(whisper_sr_model, trainfunc).cuda()

    if cfg['freeze_feats']:
        feat_extractor.requires_grad_(False)

    params    = [p for p in main_model.parameters() if p.requires_grad]
    optimizer = SGDOptimizer(params, lr=cfg['lr'], weight_decay=weight_decay)
    scheduler = OneCycleLRScheduler(optimizer,
                                    pct_start=0.05,
                                    cycle_momentum=False,
                                    max_lr=cfg['lr'],
                                    div_factor=10,
                                    final_div_factor=10000,
                                    total_steps=epochs*len(train_loader))

    results = {'experiment': args.exp, 'lr': cfg['lr'], 'epochs': []}
    start_epoch = 0

    ckpt = latest_checkpoint(model_dir)

    if ckpt is not None:
        start_epoch = loadParameters(main_model, optimizer, scheduler, path=ckpt) + 1
        print('Resumed from %s, start epoch %d' % (ckpt, start_epoch), flush=True)

        if os.path.exists(results_path):
            with open(results_path) as f:
                results = json.load(f)

            # Drop records of epochs that were trained after the last checkpoint
            results['epochs'] = [r for r in results['epochs'] if r['epoch'] < start_epoch]

    for num_epoch in range(start_epoch, epochs):
        train_loss, train_top1 = train_network(train_loader, main_model, optimizer, scheduler, num_epoch, verbose=args.verbose)

        print("Epoch {:1.0f}, Loss (train set) {:f}, Accuracy (train set) {:2.3f}%".format(num_epoch, train_loss, train_top1), flush=True)
        record = {'epoch': num_epoch, 'train_loss': train_loss, 'train_acc': train_top1}

        if (num_epoch + 1) % val_interval == 0 or num_epoch + 1 == epochs:
            _, val_top1 = test_network(val_loader, main_model)
            print("Epoch {:1.0f}, Accuracy (validation set) {:2.3f}%".format(num_epoch, val_top1), flush=True)

            feats = extract_features(main_model, test_loader)
            all_scores, all_labels, all_trials = compute_scores(feats, verif_protocol_lines)
            tar_scores, imp_scores = tar_imp_hists(all_scores, all_labels)
            EER, _ = get_eer(tar_scores, imp_scores)

            print("Epoch {:1.0f}, EER (test set) {:2.3f}%".format(num_epoch, EER), flush=True)
            record.update({'val_acc': val_top1, 'eer': EER})

            saveParameters(main_model, optimizer, scheduler, num_epoch, path=model_dir)

        results['epochs'].append(record)

        with open(results_path, 'w') as f:
            json.dump(results, f, indent=1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', default='base', choices=list(EXPERIMENTS))
    parser.add_argument('--epochs', type=int, default=max_epoch)
    parser.add_argument('--verbose', action='store_true')

    main(parser.parse_args())
