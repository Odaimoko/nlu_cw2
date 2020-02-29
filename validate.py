import os
import logging
import argparse
import numpy as np
from tqdm import tqdm
from collections import OrderedDict

import torch
import torch.nn as nn

from seq2seq import models, utils
from seq2seq.data.dictionary import Dictionary
from seq2seq.data.dataset import Seq2SeqDataset, BatchSampler
from seq2seq.models import ARCH_MODEL_REGISTRY, ARCH_CONFIG_REGISTRY


def get_args():
    """ Defines training-specific hyper-parameters. """
    parser = argparse.ArgumentParser('Sequence to Sequence Model')
    
    # Add data arguments
    parser.add_argument('--data', default = 'europarl_prepared',
                        help = 'path to data directory')
    parser.add_argument('--source-lang', default = 'de', help = 'source language')
    parser.add_argument('--target-lang', default = 'en', help = 'target language')
    parser.add_argument('--max-tokens', default = None, type = int,
                        help = 'maximum number of tokens in a batch')
    parser.add_argument('--batch-size', default = 10, type = int,
                        help = 'maximum number of sentences in a batch')
    parser.add_argument('--train-on-tiny', action = 'store_true',
                        help = 'train model on a tiny dataset')
    
    # Add model arguments
    parser.add_argument('--arch', default = 'lstm',
                        choices = ARCH_MODEL_REGISTRY.keys(), help = 'model architecture')
    parser.add_argument('--cuda', default = False, help = 'Use a GPU')
    
    # Add optimization arguments
    parser.add_argument('--max-epoch', default = 100, type = int,
                        help = 'force stop training at specified epoch')
    parser.add_argument('--clip-norm', default = 4.0,
                        type = float, help = 'clip threshold of gradients')
    parser.add_argument('--lr', default = 0.0003,
                        type = float, help = 'learning rate')
    parser.add_argument('--patience', default = 10, type = int,
                        help = 'number of epochs without improvement on validation set before early stopping')
    
    # Add checkpoint arguments
    parser.add_argument('--log-file', default = None, help = 'path to save logs')
    parser.add_argument('--save-dir', default = 'checkpoints',
                        help = 'path to save checkpoints')
    parser.add_argument('--restore-file', default = 'checkpoint_last.pt',
                        help = 'filename to load checkpoint')
    parser.add_argument('--save-interval', type = int, default = 1,
                        help = 'save a checkpoint every N epochs')
    parser.add_argument('--no-save', action = 'store_true',
                        help = 'don\'t save models or checkpoints')
    parser.add_argument('--epoch-checkpoints',
                        action = 'store_true', help = 'store all epoch checkpoints')
    
    # Parse twice as model arguments are not known the first time
    args, _ = parser.parse_known_args()
    model_parser = parser.add_argument_group(
        argument_default = argparse.SUPPRESS)
    ARCH_MODEL_REGISTRY[args.arch].add_args(model_parser)
    args = parser.parse_args()
    ARCH_CONFIG_REGISTRY[args.arch](args)
    return args


def main(args):
    """
        Calc loss and perplexity on training and validation set
    """
    logging.info('Commencing Validation!')
    torch.manual_seed(42)
    np.random.seed(42)
    
    utils.init_logging(args)
    
    # Load dictionaries [for each language]
    src_dict = Dictionary.load(os.path.join(
        args.data, 'dict.{:s}'.format(args.source_lang)))
    logging.info('Loaded a source dictionary ({:s}) with {:d} words'.format(
        args.source_lang, len(src_dict)))
    tgt_dict = Dictionary.load(os.path.join(
        args.data, 'dict.{:s}'.format(args.target_lang)))
    logging.info('Loaded a target dictionary ({:s}) with {:d} words'.format(
        args.target_lang, len(tgt_dict)))
    
    # Load datasets
    def load_data(split):
        return Seq2SeqDataset(
            src_file = os.path.join(
                args.data, '{:s}.{:s}'.format(split, args.source_lang)),
            tgt_file = os.path.join(
                args.data, '{:s}.{:s}'.format(split, args.target_lang)),
            src_dict = src_dict, tgt_dict = tgt_dict)
    
    train_dataset = load_data(
        split = 'train') if not args.train_on_tiny else load_data(split = 'tiny_train')
    valid_dataset = load_data(split = 'valid')
    
    # Build model and optimization criterion
    model = models.build_model(args, src_dict, tgt_dict)
    logging.info('Built a model with {:d} parameters'.format(
        sum(p.numel() for p in model.parameters())))
    criterion = nn.CrossEntropyLoss(
        ignore_index = src_dict.pad_idx, reduction = 'sum')
    
    if torch.cuda.is_available() and args.cuda:
        model = model.cuda()
    # Instantiate optimizer and learning rate scheduler
    optimizer = torch.optim.Adam(model.parameters(), args.lr)
    
    # Load last checkpoint if one exists
    state_dict = utils.load_checkpoint(args, model, optimizer)  # lr_scheduler
    
    
    train_loader = \
        torch.utils.data.DataLoader(train_dataset, num_workers = 1, collate_fn = train_dataset.collater,
                                    batch_sampler = BatchSampler(train_dataset, args.max_tokens, args.batch_size, 1,
                                                                 0, shuffle = True, seed = 42))
    
    # Calculate validation loss
    train_perplexity = validate(
        args, model, criterion, train_dataset, 0)
        
    valid_perplexity = validate(
        args, model, criterion, valid_dataset, 0)
        

def validate(args, model, criterion, valid_dataset, epoch):
    """ Validates model performance on a held-out development set. """
    valid_loader = \
        torch.utils.data.DataLoader(valid_dataset, num_workers = 1, collate_fn = valid_dataset.collater,
                                    batch_sampler = BatchSampler(valid_dataset, args.max_tokens, args.batch_size, 1, 0,
                                                                 shuffle = False, seed = 42))
    model.eval()
    stats = OrderedDict()
    stats['valid_loss'] = 0
    stats['num_tokens'] = 0
    stats['batch_size'] = 0
    if torch.cuda.is_available() and args.cuda:
        model = model.cuda()
    # Iterate over the validation set
    for i, sample in enumerate(valid_loader):
        if len(sample) == 0:
            continue
        
        if torch.cuda.is_available() and args.cuda:
            for k in sample:
                if type(sample[k]) == torch.Tensor:
                    sample[k] = sample[k].cuda()
        with torch.no_grad():
            # Compute loss
            output, attn_scores = model(
                sample['src_tokens'], sample['src_lengths'], sample['tgt_inputs'])
            loss = criterion(output.view(-1, output.size(-1)),
                             sample['tgt_tokens'].view(-1))
        # Update tracked statistics
        stats['valid_loss'] += loss.item()
        stats['num_tokens'] += sample['num_tokens']
        stats['batch_size'] += len(sample['src_tokens'])
    
    # Calculate validation perplexity
    stats['valid_loss'] = stats['valid_loss'] / stats['num_tokens']
    perplexity = np.exp(stats['valid_loss'])
    stats['num_tokens'] = stats['num_tokens'] / stats['batch_size']
    
    logging.info(
        'Epoch {:03d}: {}'.format(epoch, ' | '.join(key + ' {:.3g}'.format(value) for key, value in stats.items())) +
        ' | valid_perplexity {:.3g}'.format(perplexity))
    
    return perplexity


if __name__ == '__main__':
    args = get_args()
    args.device_id = 0
    
    # Set up logging to file
    logging.basicConfig(filename = args.log_file, filemode = 'a', level = logging.INFO,
                        format = '%(levelname)s: %(message)s')
    if args.log_file is not None:
        # Logging to console
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        logging.getLogger('').addHandler(console)
    
    main(args)
