#!/usr/bin/env python3
"""Finetune affinity predictor by optionally freezing the ligand/epitope branch"""
import sys
import random
sys.path.insert(1,"/home/xp/ATPnet")
import argparse
import json
import logging
import os
import sys
from copy import deepcopy
from time import time
from tqdm import tqdm

import numpy as np
import pandas as pd
import torch
from paccmann_predictor.models.bimodal_mca_transformer import BimodalMCA
from paccmann_predictor.models.mydataset_onion import ProteinSmileDataset
from transformers import BertModel, BertTokenizer
from transformers import RobertaTokenizer, RobertaModel
from sklearn.metrics import (
    auc, average_precision_score, precision_recall_curve, roc_curve
)
from torch.optim.lr_scheduler import ReduceLROnPlateau
def init_seeds(seed=0,cuda_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda_deterministic:  # slower, more reproducible
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:  # faster, less reproducible
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

def main( train_affinity_filepath, test_affinity_filepath, receptor_filepath,
        ligand_filepath,output_dir,training_name,pretrain_weights_path):
    # 初始化进程组
    torch.distributed.init_process_group(backend="nccl")
    local_rank = int(os.environ['LOCAL_RANK'])
    device = torch.device("cuda", local_rank)
    torch.cuda.empty_cache()
    # 初始化随机数
    init_seeds(123456+local_rank)
    # 设置log
    logging.basicConfig(level=logging.INFO if local_rank in [-1, 1] else logging.WARN)
    # Create model directory and dump files
    training_dir = os.path.join(output_dir, training_name)
    if not os.path.exists(training_dir) and local_rank==1:
        os.makedirs(training_dir)
    # Process and dump parameter file:
    params = {}
    with open(os.path.join(output_dir, 'model_params.json')) as fp:
        params.update(json.load(fp))
    with open(os.path.join(training_dir, 'model_params.json'), 'w') as fp:
        json.dump(params, fp, indent=4)

    # Restore model
    logging.info("Restore model...")
    model = BimodalMCA(params)

    model.load(pretrain_weights_path, map_location=device)
    # logging.info("SyncBN")
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = model.to(device)

    # using DP
    #model=torch.nn.DataParallel(model,device_ids=[0,1])
    logging.info("DDP training")
    model = torch.nn.parallel.DistributedDataParallel(model,
                                                      device_ids=[local_rank],
                                                      output_device=local_rank)

    num_params = sum(p.numel() for p in model.parameters())
    num_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f'Number of parameters: {num_params}, trainable: {num_train}')
    # Assemble datasets

    Protein_model_name="Rostlab/prot_bert_bfd"
    SMILES_model_name = "DeepChem/ChemBERTa-77M-MLM"
    # AA model
    Protein_tokenizer = BertTokenizer.from_pretrained(Protein_model_name, do_lower_case=False, local_files_only=True)
    Protein_model = BertModel.from_pretrained(Protein_model_name, torch_dtype=torch.float16, local_files_only=True)
    Protein_model = Protein_model.to(device)

    # SMILES model
    SMILES_tokenizer = RobertaTokenizer.from_pretrained(SMILES_model_name, local_files_only=True)
    SMILES_model = RobertaModel.from_pretrained(SMILES_model_name, torch_dtype=torch.float16,local_files_only=True)
    SMILES_model = SMILES_model.to(device)

    Protein_num_params = sum(p.numel() for p in Protein_model.parameters())
    SMILES_num_params = sum(p.numel() for p in SMILES_model.parameters())
    logging.info(f'Number of parameters Protein: {Protein_num_params}, SMILES: {SMILES_num_params}')

    train_dataset = ProteinSmileDataset(
                 affinity_filepath=train_affinity_filepath,
                 receptor_filepath= receptor_filepath,
                 Protein_model=Protein_model,
                 Protein_tokenizer=Protein_tokenizer,
                 Protein_padding=params.get("receptor_padding_length"),
                 ligand_filepath=ligand_filepath,
                 SMILES_model=SMILES_model,
                 SMILES_tokenizer=SMILES_tokenizer,
                 SMILES_padding=params.get("ligand_padding_length"),
                 SMILES_argument=True,
                 SMILES_Canonicalization=False,
                 device=device
                 )
    train_sample = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=params['batch_size'],
        # shuffle=True,
        drop_last=False,
        num_workers=params.get('num_workers', 0),
        sampler=train_sample,
        pin_memory=False
    )
    test_dataset = ProteinSmileDataset(
        affinity_filepath=test_affinity_filepath,
        receptor_filepath=receptor_filepath,
        Protein_model=Protein_model,
        Protein_tokenizer=Protein_tokenizer,
        Protein_padding=params.get("receptor_padding_length"),
        ligand_filepath=ligand_filepath,
        SMILES_model=SMILES_model,
        SMILES_tokenizer=SMILES_tokenizer,
        SMILES_padding=params.get("ligand_padding_length"),
        SMILES_argument=False,
        SMILES_Canonicalization=True,
        device=device
    )
    test_sample = torch.utils.data.distributed.DistributedSampler(test_dataset)
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=params['batch_size'],
        # shuffle=False,
        drop_last=False,
        num_workers=params.get('num_workers', 0),
        sampler=test_sample,
        pin_memory=False
    )

    logging.info(
        f'Training dataset has {len(train_dataset)} samples, '
        f'Testset  dataset has {len(test_dataset)} samples.'
    )
    logging.info(f'batchsize:{params["batch_size"]}')
    logging.info(f'num_workers:{params["num_workers"]}')
    logging.info(
        f'Loader length: Train - {len(train_loader)}, test - {len(test_loader)}'
    )

    # Define optimizer
    min_loss, max_roc_auc = 100, 0
    num_epochs = params.get('epochs')
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=params.get('lr'),
                                  weight_decay=params.get('weight_decay'),
                                  amsgrad=params.get('amsgrad'))
    # scheduler = ReduceLROnPlateau(
    #     optimizer,
    #     mode='max',
    #     factor=0.8,
    #     patience=3,
    #     min_lr=1e-07,
    #     verbose=True
    # )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=params.get('lr'), pct_start=0.4,
                                                    steps_per_epoch=len(train_loader),
                                                    epochs=num_epochs, verbose=False,three_phase=True)

    save_top_model = os.path.join(training_dir, '{}_{}_{}.pt')
    def save(local_rank, path, metric, typ, val=None):
        """Routine to save model"""
        if local_rank == 1:
            save_name=path.format(typ, metric, round(val,4))
            model.module.save(save_name)
            if typ == 'best':
                logging.info(
                    f'\t New best performance in "{metric}"'
                    f' with value : {val} in epoch: {epoch}'
                )

    # Start training
    logging.info('Training about to start...\n')
    t = time()
    result = []
    learning_rates=[params.get("lr")]
    for epoch in range(num_epochs):
        logging.info(f"== Epoch [{epoch}/{num_epochs}] ==")
        t = time()
        # Now training
        model.train()
        train_loader.sampler.set_epoch(epoch)
        train_loss = 0
        for ind, (receptor_seq,ligand_AA_seq,ligand_SMILES_seq,receptor_embedding,SMILES_embedding,y) in enumerate(tqdm(train_loader)):
            y_hat, pred_dict = model(SMILES_embedding.to(device), receptor_embedding.to(device))
            loss = model.module.loss(y_hat, y.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # change LR
            scheduler.step()
            learning_rates.append(optimizer.param_groups[0]["lr"])
            train_loss += loss.item()

        logging.info(
            "\t **** TRAINING ****   "
            f"Epoch [{epoch + 1}/{num_epochs}], "
            f"loss: {train_loss / len(train_loader):.5f}. "
            f"This took {time() - t:.1f} secs."
        )

        #test
        model.eval()
        test_loader.sampler.set_epoch(epoch)
        with torch.no_grad():
            test_loss = 0
            predictions = []
            labels = []
            for ind, (receptor_seq,ligand_AA_seq,ligand_SMILES_seq,receptor_embedding,SMILES_embedding,y) in enumerate(test_loader):
                y_hat, pred_dict = model(
                    SMILES_embedding.to(device), receptor_embedding.to(device)
                )
                predictions.append(y_hat)
                labels.append(y.clone())
                loss = model.module.loss(y_hat, y.to(device))
                test_loss += loss.item()
        predictions = torch.cat(predictions, dim=0).flatten().cpu().numpy()
        labels = torch.cat(labels, dim=0).flatten().cpu().numpy()

        test_loss = test_loss / len(test_loader)
        fpr, tpr, _ = roc_curve(labels, predictions)
        test_roc_auc = auc(fpr, tpr)

        # scheduler.step(test_roc_auc)
        # calculations for visualization plot
        precision, recall, _ = precision_recall_curve(labels, predictions)
        avg_precision = average_precision_score(labels, predictions)

        logging.info(
            f"\t **** TESTING **** Epoch [{epoch + 1}/{num_epochs}], "
            f"loss: {test_loss:.5f}, ROC-AUC: {test_roc_auc:.3f}, "
            f"Average precision: {avg_precision:.3f}."
        )

        if test_roc_auc > max_roc_auc:
            max_roc_auc = test_roc_auc
            save(local_rank, save_top_model, 'ROC-AUC', 'best', max_roc_auc)
            ep_roc = epoch
            roc_auc_loss = test_loss
            roc_auc_pr = avg_precision

        if test_loss < min_loss:
            min_loss = test_loss
            save(local_rank, save_top_model, 'loss', 'best', min_loss)
            ep_loss = epoch
            loss_roc_auc = test_roc_auc


        train_loss = train_loss / len(train_loader)
        result.append([epoch, test_loss, test_roc_auc, avg_precision, train_loss])

    #总结训练
    logging.info(
        'Overall best performances are: \n \t'
        f'Loss = {min_loss:.4f} in epoch {ep_loss} '
        f'\t (ROC-AUC was {loss_roc_auc:4f}) \n \t'
        f'ROC-AUC = {max_roc_auc:.4f} in epoch {ep_roc} '
        f'\t (Loss was {roc_auc_loss:4f})'
    )


    logging.info('Done with training, models saved, shutting down.')
    result = pd.DataFrame(result, columns=["epoch", "test_loss", "test_roc_auc", "avg_precision", "train_loss"])
    max_result = pd.DataFrame([[ep_roc, max_roc_auc, roc_auc_loss], [ep_loss, loss_roc_auc, min_loss]],
                              columns=["epoch", "ROC_AUC", "loss"])
    learning_rates=pd.DataFrame(learning_rates)

    writer = pd.ExcelWriter(os.path.join(training_dir, 'overview.xlsx'), mode='w', engine='openpyxl')
    result.to_excel(writer, "result")
    max_result.to_excel(writer, "max_result")
    learning_rates.to_excel(writer,"learning_rates")
    writer.save()
    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--train_affinity_filepath', type=str,
        help='Path to the affinity data.'
    )
    parser.add_argument(
        '--test_affinity_filepath', type=str,
        help='Path to the affinity data.'
    )
    parser.add_argument(
        '--receptor_filepath', type=str,
        help='Path to the receptor aa data (.csv)'
    )
    parser.add_argument(
        '--ligand_filepath', type=str,
        help='Path to the ligand data (.smi)'
    )
    parser.add_argument(
        '--outputdir', type=str,
        help='Directory to store  model in'
    )
    parser.add_argument(
        '--training_name', type=str,
        help='Name for the training.'
    )
    args = parser.parse_args()
    main(
        args.train_affinity_filepath, args.test_affinity_filepath,
        args.receptor_filepath, args.ligand_filepath,
        args.pretrained_model_path, args.finetuned_model_path,
        args.training_name, args.model_type, args.params_filepath,
        args.smile_augment
    )

    #CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --nnodes=1 --nproc_per_node=2 --node_rank=0  pretrain.py