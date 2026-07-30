from trainer import train
import json
import argparse

config_map = {
    'DVS128_10':     'pretrained_models/DVS128_10_24ms/all_params.json',
    'DVS128_11':     'pretrained_models/DVS128_11_24ms/all_params.json',
    'SLAnimals_3s':  'pretrained_models/SLAnimals_3s_48ms/all_params.json',
    'SLAnimals_4s':  'pretrained_models/SLAnimals_4s_48ms/all_params.json',
    'ASL_DVS':       'pretrained_models/ASL_DVS/all_params.json',
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train EventTransformer classifier')
    parser.add_argument('--dataset', type=str, default='DVS128_10',
                        choices=list(config_map.keys()),
                        help='Dataset to train on')
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--num_workers', type=int, default=None)
    parser.add_argument('--precision', type=int, default=None,
                        help='Training precision: 16 (FP16), 32 (default)')
    args = parser.parse_args()

    train_params = json.load(open(config_map[args.dataset], 'r'))

    train_params['logger_params']['csv']['save_dir'] = '{}'
    for k, v in train_params['callbacks_params']:
        if k != 'model_chck': continue
        v['dirpath'] = '{}/weights/'
        v['filename'] = '{epoch}-{val_loss_total:.5f}-{val_loss_clf:.5f}-{val_acc:.5f}'

    if args.batch_size is not None:
        train_params['data_params']['batch_size'] = args.batch_size
    if args.num_workers is not None:
        train_params['data_params']['num_workers'] = args.num_workers
    if args.precision is not None:
        train_params['training_params']['precision'] = args.precision

    path_results = 'pretrained_models'

    train('', path_results,
          data_params      = train_params['data_params'],
          backbone_params  = train_params['backbone_params'],
          clf_params       = train_params['clf_params'],
          training_params  = train_params['training_params'],
          optim_params     = train_params['optim_params'],
          callback_params  = train_params['callbacks_params'],
          logger_params    = train_params['logger_params'])
