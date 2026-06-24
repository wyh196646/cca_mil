from __future__ import print_function
import argparse
import os
from utils.file_utils import save_pkl
from utils.utils import *
from utils.core_utils import train
from datasets.dataset_generic import Generic_MIL_Dataset
import torch
import pandas as pd
import numpy as np

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = '/data2/yuhaowang/WSIFew'
DEFAULT_RESULTS_DIR = '/data2/yuhaowang/cca-mil-result/results'

DEFAULT_DATASET_CSVS = {
    'task_tcga_rcc_subtyping': os.path.join(ROOT_DIR, 'dataset_csv', 'RCC.csv'),
    'task_tcga_lung_subtyping': os.path.join(ROOT_DIR, 'dataset_csv', 'LUAD_LUSC.csv'),
    'task_UBC-OCEAN_subtyping': os.path.join(ROOT_DIR, 'dataset_csv', 'UBC-OCEAN.csv'),
    'task_camelyon_subtyping': os.path.join(ROOT_DIR, 'dataset_csv', 'camelyon.csv'),
    'task_TUPAC16_subtyping': os.path.join(DATA_ROOT, 'dataset_csv', 'TUPAC16.csv'),
}

DEFAULT_CONCEPT_BANKS = {
    'task_tcga_rcc_subtyping': os.path.join(ROOT_DIR, 'text_prompt', 'concept_bank', 'tcga_rcc.json'),
    'task_tcga_lung_subtyping': os.path.join(ROOT_DIR, 'text_prompt', 'concept_bank', 'tcga_nsclc.json'),
    'task_camelyon_subtyping': os.path.join(ROOT_DIR, 'text_prompt', 'concept_bank', 'camelyon.json'),
    'task_UBC-OCEAN_subtyping': os.path.join(ROOT_DIR, 'text_prompt', 'concept_bank', 'ubc_ocean.json'),
}

# Generic training settings
parser = argparse.ArgumentParser(description='Configurations for WSI Training')
parser.add_argument('--data_root_dir', type=str, default=None, help='data directory')
parser.add_argument('--data_folder_s', type=str, default=None, help='dir under data directory' )
parser.add_argument('--data_folder_l', type=str, default=None, help='dir under data directory' )
parser.add_argument('--max_epochs', type=int, default=80, help='maximum number of epochs to train (default: 80)')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 0.0001)')
parser.add_argument('--label_frac', type=float, default=1.0, help='fraction of training labels (default: 1.0)')
parser.add_argument('--seed', type=int, default=1, help='random seed for reproducible experiment (default: 1)')
parser.add_argument('--k', type=int, default=5, help='number of folds (default: 5)')
parser.add_argument('--k_start', type=int, default=-1, help='start fold (default: -1, last fold)')
parser.add_argument('--k_end', type=int, default=-1, help='end fold (default: -1, first fold)')
parser.add_argument('--results_dir', default=DEFAULT_RESULTS_DIR, help='results directory (default: {})'.format(DEFAULT_RESULTS_DIR))
parser.add_argument('--split_dir', type=str, default=None)
parser.add_argument('--log_data', action='store_true', default=False, help='log data using tensorboard')
parser.add_argument('--testing', action='store_true', default=False, help='debugging tool')
parser.add_argument('--early_stopping', action='store_true', default=False, help='enable early stopping')
parser.add_argument('--early_stopping_patience', type=int, default=15, help='early stopping patience (default: 15)')
parser.add_argument('--early_stopping_stop_epoch', type=int, default=0, help='minimum epoch before early stopping can trigger (default: 0)')
parser.add_argument('--opt', type=str, choices = ['adam', 'sgd'], default='adam')
parser.add_argument('--drop_out', action='store_true', default=False, help='enabel dropout (p=0.25)')
parser.add_argument('--model_type', type=str, choices=['ViLa_MIL', 'FOCUS', 'CCA_MIL'], default='CCA_MIL', help='few-shot model for WSI classification')
parser.add_argument('--mode', type=str, choices=['transformer'], default='transformer')
parser.add_argument('--exp_code', type=str, help='experiment code for saving results')
parser.add_argument('--weighted_sample', action='store_true', default=False, help='enable weighted sampling')
parser.add_argument('--reg', type=float, default=1e-5, help='weight decay (default: 1e-5)')
parser.add_argument('--bag_loss', type=str, choices=['svm', 'ce', 'focal'], default='ce')
parser.add_argument('--task', type=str)
parser.add_argument('--csv_path', type=str, default=None, help='dataset CSV path; overrides the task default')
parser.add_argument("--text_prompt", type=str, default=None)
parser.add_argument("--text_prompt_path", type=str, default=None)
parser.add_argument("--prototype_number", type=int, default=16)
parser.add_argument("--window_size", type=int, default=8)
parser.add_argument("--sim_threshold", type=float, default=0.8)
parser.add_argument("--max_context_length", type=int, default=8192)
parser.add_argument("--conch_ckpt_path", type=str, default=os.path.join(ROOT_DIR, "ckg", "pytorch_model.bin"))
parser.add_argument("--concept_bank_path", type=str, default=None)
parser.add_argument("--num_visual_prototypes", type=int, default=6)
parser.add_argument("--proto_tau", type=float, default=0.1)
parser.add_argument("--ot_epsilon", type=float, default=0.05)
parser.add_argument("--sinkhorn_iter", type=int, default=20)
parser.add_argument("--uot_rho_a", type=float, default=0.5)
parser.add_argument("--uot_rho_b", type=float, default=0.5)
parser.add_argument("--concept_pooling", type=str, choices=["mean", "learnable", "attention"], default="attention")
parser.add_argument("--lambda_contrast", type=float, default=None)
parser.add_argument("--contrast_tau", type=float, default=None)
parser.add_argument("--cluster_k", type=int, default=8, help="legacy alias; no longer used by CCA_MIL")
parser.add_argument("--kmeans_iters", type=int, default=10, help="legacy alias; no longer used by CCA_MIL")
parser.add_argument("--min_cluster_size", type=int, default=5, help="legacy alias; no longer used by CCA_MIL")
parser.add_argument("--selection_top_r", type=int, default=3, help="legacy alias; no longer used by CCA_MIL")
parser.add_argument("--concept_alpha", type=float, default=0.5, help="legacy alias; no longer used by CCA_MIL")
parser.add_argument("--common_concept_weight", type=float, default=0.3)
parser.add_argument("--lambda_con", type=float, default=0.1, help="legacy alias for --lambda_contrast")
parser.add_argument("--lambda_div", type=float, default=0.01)
parser.add_argument("--tau", type=float, default=0.07, help="legacy alias for --contrast_tau")
parser.add_argument("--no_normalize_kmeans", action="store_true", default=False)
parser.add_argument("--train_concept_prompt", action="store_true", default=True)
parser.add_argument("--freeze_concept_prompt", action="store_false", dest="train_concept_prompt")
parser.add_argument("--concept_prompt_n_ctx", type=int, default=4)
parser.add_argument("--concept_prompt_template_count", type=int, default=4)
parser.add_argument("--max_train_patches", type=int, default=4096)
parser.add_argument("--max_eval_patches", type=int, default=0)
parser.add_argument("--concept_logit_weight", type=float, default=0.5)
parser.add_argument("--concept_logit_tau", type=float, default=1.0)
parser.add_argument("--store_explanations", action="store_true", default=False)

args = parser.parse_args()


def resolve_results_dir(results_dir):
    results_dir = os.path.expanduser(str(results_dir))
    if os.path.isabs(results_dir):
        return results_dir

    normalized = os.path.normpath(results_dir)
    if normalized in ('.', ''):
        return DEFAULT_RESULTS_DIR

    parts = normalized.split(os.sep)
    if parts and parts[0] == 'results':
        parts = parts[1:]
    suffix = os.path.join(*parts) if parts else ''
    return os.path.join(DEFAULT_RESULTS_DIR, suffix) if suffix else DEFAULT_RESULTS_DIR


args.results_dir = resolve_results_dir(args.results_dir)
if args.lambda_contrast is None:
    args.lambda_contrast = args.lambda_con
if args.contrast_tau is None:
    args.contrast_tau = args.tau
args.text_prompt = None
if args.text_prompt_path is not None:
    args.text_prompt = np.array(pd.read_csv(args.text_prompt_path, header=None)).squeeze()


def resolve_csv_path(task):
    csv_path = args.csv_path if args.csv_path is not None else DEFAULT_DATASET_CSVS[task]
    csv_path = os.path.expanduser(csv_path)
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(
            'Dataset CSV not found: {}. Pass --csv_path to use a local CSV.'.format(csv_path)
        )
    args.csv_path = csv_path
    return csv_path

def seed_torch(seed=7):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_torch(args.seed)

print('\nLoad Dataset')

if args.task == 'task_tcga_rcc_subtyping':
    args.n_classes=3
    args.class_names = ['KICH', 'KIRC', 'KIRP']
    dataset = Generic_MIL_Dataset(csv_path = resolve_csv_path(args.task),
                                  mode = args.mode,
                                  data_dir_s = args.data_folder_s,
                                  data_dir_l = args.data_folder_l,
                                  shuffle = False,
                                  print_info = True,
                                  label_dict = {'KICH':0, 'KIRC':1, 'KIRP':2},
                                  patient_strat= False,
                                  ignore=[])
    # #   data_dir_s = os.path.join(args.data_root_dir, args.data_folder_s),
    # data_dir_l = os.path.join(args.data_root_dir, args.data_folder_l),
                                  
elif args.task == 'task_tcga_lung_subtyping':
    args.n_classes=2
    args.class_names = ['LUAD', 'LUSC']
    dataset = Generic_MIL_Dataset(csv_path = resolve_csv_path(args.task),
                                  mode = args.mode,
                                  data_dir_s = args.data_folder_s,
                                  data_dir_l = args.data_folder_l,
                                  shuffle = False,
                                  print_info = True,
                                  label_dict = {'LUAD':0, 'LUSC':1},
                                  patient_strat= False,
                                  ignore=[])

elif args.task == 'task_UBC-OCEAN_subtyping':
    args.n_classes=5
    args.class_names = ['CC', 'HGSC', 'LGSC', 'EC', 'MC']
    dataset = Generic_MIL_Dataset(csv_path = resolve_csv_path(args.task),
                            mode = args.mode,
                            data_dir_s = args.data_folder_s,
                            data_dir_l = args.data_folder_l,
                            shuffle = False, 
                            print_info = True,
                            label_dict = {'CC': 0, 'HGSC': 1, 'LGSC': 2, 'EC': 3, 'MC': 4},
                            patient_strat=False,
                            ignore=[])
    
elif args.task == 'task_camelyon_subtyping':
    args.n_classes=2
    args.class_names = ['normal', 'tumor']
    dataset = Generic_MIL_Dataset(csv_path = resolve_csv_path(args.task),
                            mode = args.mode,
                            data_dir_s = args.data_folder_s,
                            data_dir_l = args.data_folder_l,
                            shuffle = False, 
                            print_info = True,
                            label_dict = {'normal':0, 'tumor':1},
                            patient_strat=False,
                            ignore=[])
    
elif args.task == 'task_TUPAC16_subtyping':
    args.n_classes=3
    args.class_names = ['subtype_1', 'subtype_2', 'subtype_3']
    dataset = Generic_MIL_Dataset(csv_path = resolve_csv_path(args.task),
                            mode = args.mode,
                            data_dir_s = args.data_folder_s,
                            data_dir_l = args.data_folder_l,
                            shuffle = False, 
                            print_info = True,
                            label_dict = {'subtype_1':0, 'subtype_2':1, 'subtype_3':2},
                            patient_strat=False,
                            ignore=[])

else:
    raise NotImplementedError

if args.model_type in ['ViLa_MIL', 'FOCUS'] and args.text_prompt is None:
    raise ValueError('--text_prompt_path is required for {}'.format(args.model_type))

if args.model_type == 'CCA_MIL':
    if args.concept_bank_path is None:
        if args.task not in DEFAULT_CONCEPT_BANKS:
            raise ValueError('--concept_bank_path is required for {}'.format(args.task))
        args.concept_bank_path = DEFAULT_CONCEPT_BANKS[args.task]
    if not os.path.isfile(args.concept_bank_path):
        raise FileNotFoundError('Concept bank not found: {}'.format(args.concept_bank_path))

settings = {'num_splits': args.k,
            'k_start': args.k_start,
            'k_end': args.k_end,
            'task': args.task,
            'max_epochs': args.max_epochs,
            'results_dir': args.results_dir,
            'lr': args.lr,
            'experiment': args.exp_code,
            'label_frac': args.label_frac,
            'seed': args.seed,
            'early_stopping_patience': args.early_stopping_patience,
            'early_stopping_stop_epoch': args.early_stopping_stop_epoch,
            'model_type': args.model_type,
            'mode': args.mode,
            'csv_path': args.csv_path,
            "use_drop_out": args.drop_out,
            'weighted_sample': args.weighted_sample,
            'opt': args.opt,
            'class_names': getattr(args, 'class_names', None),
            'text_prompt_path': args.text_prompt_path,
            'conch_ckpt_path': args.conch_ckpt_path,
            'max_context_length': args.max_context_length,
            'window_size': args.window_size,
            'sim_threshold': args.sim_threshold,
            'concept_bank_path': args.concept_bank_path,
            'num_visual_prototypes': args.num_visual_prototypes,
            'proto_tau': args.proto_tau,
            'ot_epsilon': args.ot_epsilon,
            'sinkhorn_iter': args.sinkhorn_iter,
            'uot_rho_a': args.uot_rho_a,
            'uot_rho_b': args.uot_rho_b,
            'concept_pooling': args.concept_pooling,
            'common_concept_weight': args.common_concept_weight,
            'lambda_contrast': args.lambda_contrast,
            'lambda_div': args.lambda_div,
            'contrast_tau': args.contrast_tau,
            'train_concept_prompt': args.train_concept_prompt,
            'concept_prompt_n_ctx': args.concept_prompt_n_ctx,
            'concept_prompt_template_count': args.concept_prompt_template_count,
            'max_train_patches': args.max_train_patches,
            'max_eval_patches': args.max_eval_patches,
            'concept_logit_weight': args.concept_logit_weight,
            'concept_logit_tau': args.concept_logit_tau}

if not os.path.exists(args.results_dir):
    os.makedirs(args.results_dir)

args.results_dir = os.path.join(args.results_dir, str(args.exp_code) + '_s{}'.format(args.seed))
if not os.path.exists(args.results_dir):
    os.makedirs(args.results_dir)

if args.split_dir is None:
    args.split_dir = os.path.join('splits', args.task+'_{}'.format(int(args.label_frac*100)))
else:
    args.split_dir = os.path.join('splits', args.split_dir)

print('split_dir: ', args.split_dir)
assert os.path.isdir(args.split_dir)

settings.update({'split_dir': args.split_dir})


with open(args.results_dir + '/experiment_{}.txt'.format(args.exp_code), 'w') as f:
    print(settings, file=f)
f.close()

print("################# Settings ###################")
for key, val in settings.items():
    print("{}:  {}".format(key, val))


def main(args):
    if args.k_start == -1:
        start = 0
    else:
        start = args.k_start
    if args.k_end == -1:
        end = args.k
    else:
        end = args.k_end

    all_test_auc = []
    all_val_auc = []
    all_test_acc = []
    all_val_acc = []
    all_test_f1 = []
    all_val_f1 = []
    folds = np.arange(start, end)
    for i in folds:
        seed_torch(args.seed)
        train_dataset, val_dataset, test_dataset = dataset.return_splits(from_id=False, csv_path='{}/splits_{}.csv'.format(args.split_dir, i)) 
        datasets = (train_dataset, val_dataset, test_dataset)
        results, test_auc, val_auc, test_acc, val_acc, _, test_f1, val_f1 = train(datasets, i, args)

        all_test_auc.append(test_auc)
        all_val_auc.append(val_auc)
        all_test_f1.append(test_f1)
        all_val_f1.append(val_f1)
        all_test_acc.append(test_acc)
        all_val_acc.append(val_acc)
        filename = os.path.join(args.results_dir, 'split_{}_results.pkl'.format(i))
        save_pkl(filename, results)

    final_df = pd.DataFrame({
        'folds': folds,
        'val_auc': all_val_auc,
        'val_acc': all_val_acc,
        'val_f1': all_val_f1,
        'test_auc': all_test_auc,
        'test_acc': all_test_acc,
        'test_f1': all_test_f1,
    })
    result_df = pd.DataFrame({'metric': ['mean', 'var'],
                              'val_auc': [np.mean(all_val_auc), np.std(all_val_auc)],
                              'val_f1': [np.mean(all_val_f1), np.std(all_val_f1)],
                              'val_acc': [np.mean(all_val_acc), np.std(all_val_acc)],
                              'test_auc': [np.mean(all_test_auc), np.std(all_test_auc)],
                              'test_f1': [np.mean(all_test_f1), np.std(all_test_f1)],
                              'test_acc': [np.mean(all_test_acc), np.std(all_test_acc)],
                              })

    if len(folds) != args.k:
        save_name = 'summary_partial_{}_{}.csv'.format(folds[0], folds[-1])
        result_name = 'result_partial_{}_{}.csv'.format(folds[0], folds[-1])
    else:
        save_name = 'summary.csv'
        result_name = 'result.csv'

    result_df.to_csv(os.path.join(args.results_dir, result_name), index=False)
    final_df.to_csv(os.path.join(args.results_dir, save_name))


if __name__ == "__main__":
    results = main(args)
    print("finished!")
    print("end script")


