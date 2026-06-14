import numpy as np
import torch
from models.model_mil import MIL_fc, MIL_fc_mc
import pandas as pd
from utils.utils import *
from utils.core_utils import Accuracy_Logger
from sklearn.metrics import roc_auc_score, roc_curve, auc, f1_score
from sklearn.preprocessing import label_binarize


def initiate_model(args, ckpt_path):
    print('Init Model')    
    model_dict = {"dropout": args.drop_out, 'n_classes': args.n_classes}
    
    if getattr(args, 'model_size', None) is not None and args.model_type in ['clam_sb', 'clam_mb']:
        model_dict.update({"size_arg": args.model_size})
    
    if args.model_type == 'ViLa_MIL':
        import ml_collections
        from models.model_ViLa_MIL import ViLa_MIL_Model
        config = ml_collections.ConfigDict()
        config.input_size = 512
        config.hidden_size = 192
        config.text_prompt = args.text_prompt
        config.prototype_number = getattr(args, 'prototype_number', 16)
        model_dict = {'config': config, 'num_classes':args.n_classes}
        model = ViLa_MIL_Model(**model_dict)

    elif args.model_type == 'FOCUS':
        import ml_collections
        from models.model_FOCUS import FOCUS
        config = ml_collections.ConfigDict()
        config.input_size = 512
        config.hidden_size = 192
        config.text_prompt = args.text_prompt
        config.prototype_number = getattr(args, 'prototype_number', 16)
        config.max_context_length = getattr(args, 'max_context_length', 8192)
        config.window_size = getattr(args, 'window_size', 8)
        config.sim_threshold = getattr(args, 'sim_threshold', 0.8)
        config.conch_ckpt_path = getattr(args, 'conch_ckpt_path', 'ckg/pytorch_model.bin')
        model_dict = {'config': config, 'num_classes': args.n_classes}
        model = FOCUS(**model_dict)

    elif args.model_type == 'CCA_MIL':
        import ml_collections
        from models.cca_mil import CCA_MIL
        config = ml_collections.ConfigDict()
        config.input_size = 512
        config.feature_dim = 512
        config.concept_bank_path = args.concept_bank_path
        config.class_names = getattr(args, 'class_names', None)
        config.cluster_k = getattr(args, 'cluster_k', 8)
        config.kmeans_iters = getattr(args, 'kmeans_iters', 10)
        config.normalize_kmeans = not getattr(args, 'no_normalize_kmeans', False)
        config.min_cluster_size = getattr(args, 'min_cluster_size', 5)
        config.selection_top_r = getattr(args, 'selection_top_r', 3)
        config.concept_alpha = getattr(args, 'concept_alpha', 0.5)
        config.common_concept_weight = getattr(args, 'common_concept_weight', 0.3)
        config.lambda_con = getattr(args, 'lambda_con', 0.1)
        config.lambda_div = getattr(args, 'lambda_div', 0.01)
        config.tau = getattr(args, 'tau', 0.07)
        config.train_concept_prompt = getattr(args, 'train_concept_prompt', False)
        config.store_explanations = getattr(args, 'store_explanations', False)
        config.conch_ckpt_path = getattr(args, 'conch_ckpt_path', 'ckg/pytorch_model.bin')
        model_dict = {'config': config, 'num_classes': args.n_classes}
        model = CCA_MIL(**model_dict)

    else: # args.model_type == 'mil'
        if args.n_classes > 2:
            model = MIL_fc_mc(**model_dict)
        else:
            model = MIL_fc(**model_dict)

    print_network(model)

    ckpt = torch.load(ckpt_path)
    ckpt_clean = {}
    for key in ckpt.keys():
        if 'instance_loss_fn' in key:
            continue
        ckpt_clean.update({key.replace('.module', ''):ckpt[key]})
    model.load_state_dict(ckpt_clean, strict=True)

    if hasattr(model, "relocate"):
        model.relocate()
    else:
        model = model.to(torch.device('cuda'))
        # pass
    model.eval()
    return model

def eval(mode, dataset, args, ckpt_path):
    model = initiate_model(args, ckpt_path)
    
    print('Init Loaders')
    loader = get_simple_loader(dataset, mode=args.mode)
    patient_results, test_error, auc, test_f1, df, acc_logger = summary(mode, model, loader, args)
    print('test_error: ', test_error)
    print('auc: ', auc)
    print('f1: ', test_f1)

    each_class_acc = []
    for i in range(args.n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        each_class_acc.append(acc)

    return model, patient_results, test_error, auc, test_f1, df, each_class_acc

def summary(mode, model, loader, args):
    acc_logger = Accuracy_Logger(n_classes=args.n_classes)
    model.eval()
    test_loss = 0.
    test_error = 0.
    test_f1 = 0.

    all_probs = np.zeros((len(loader), args.n_classes))
    all_labels = np.zeros(len(loader))
    all_preds = np.zeros(len(loader))

    all_pred = []
    all_label = []

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}

    if(mode == 'transformer'):
        for batch_idx, (data_s, data_l, label) in enumerate(loader):
            data_s, data_l, label = data_s.to(device), data_l.to(device), label.to(device)
            slide_id = slide_ids.iloc[batch_idx]
            with torch.no_grad():
                Y_prob, Y_hat, loss = model(data_s, data_l, label)

            acc_logger.log(Y_hat, label)
            probs = Y_prob.cpu().numpy()
            all_probs[batch_idx] = probs
            all_labels[batch_idx] = label.item()
            all_preds[batch_idx] = Y_hat.item()
            all_pred.append(Y_hat.item())
            all_label.append(label.item())
            patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'prob': probs, 'label': label.item()}})
            error = calculate_error(Y_hat, label)
            test_error += error

        test_f1 = f1_score(all_label, all_pred, average='macro')
        test_error /= len(loader)

        aucs = []
        if len(np.unique(all_labels)) == 1:
            auc_score = -1

        else:
            if args.n_classes == 2:
                auc_score = roc_auc_score(all_labels, all_probs[:, 1])
            else:
                binary_labels = label_binarize(all_labels, classes=[i for i in range(args.n_classes)])
                for class_idx in range(args.n_classes):
                    if class_idx in all_labels:
                        fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx])
                        aucs.append(auc(fpr, tpr))
                    else:
                        aucs.append(float('nan'))
                if args.micro_average:
                    binary_labels = label_binarize(all_labels, classes=[i for i in range(args.n_classes)])
                    fpr, tpr, _ = roc_curve(binary_labels.ravel(), all_probs.ravel())
                    auc_score = auc(fpr, tpr)
                else:
                    auc_score = np.nanmean(np.array(aucs))

        results_dict = {'slide_id': slide_ids, 'Y': all_labels, 'Y_hat': all_preds}
        for c in range(args.n_classes):
            results_dict.update({'p_{}'.format(c): all_probs[:,c]})
        df = pd.DataFrame(results_dict)
        return patient_results, test_error, auc_score, test_f1, df, acc_logger 
