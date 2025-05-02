#!/usr/bin/env python3
#
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


import argparse
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_, clip_grad_value_
from pytorch_pretrained_bert import BertAdam
import pickle

from src.data.helpers import get_data_loaders
from src.models import get_model,ce_loss, emc_loss
from src.utils.logger import create_logger
from src.utils.utils import *
from scheduler import GradualWarmupScheduler
from collections import defaultdict

from torch.utils.tensorboard import SummaryWriter

def get_args(parser):
    parser.add_argument("--batch_sz", type=int, default=128)
    parser.add_argument("--bert_model", type=str, default="bert-base-uncased")#, choices=["bert-base-uncased", "bert-large-uncased"])
    parser.add_argument("--data_path", type=str, default="/path/to/data_dir/")
    parser.add_argument("--drop_img_percent", type=float, default=0.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--embed_sz", type=int, default=300)
    parser.add_argument("--freeze_img", type=int, default=0)
    parser.add_argument("--freeze_txt", type=int, default=0)
    parser.add_argument("--glove_path", type=str, default="./datasets/glove_embeds/glove.840B.300d.txt")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=24)
    parser.add_argument("--hidden", nargs="*", type=int, default=[])
    parser.add_argument("--hidden_sz", type=int, default=768)
    parser.add_argument("--img_embed_pool_type", type=str, default="avg", choices=["max", "avg"])
    parser.add_argument("--img_hidden_sz", type=int, default=2048)
    parser.add_argument("--include_bn", type=int, default=True)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_factor", type=float, default=0.5)
    parser.add_argument("--lr_patience", type=int, default=2)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--n_workers", type=int, default=12)
    parser.add_argument("--name", type=str, default="nameless")
    parser.add_argument("--num_image_embeds", type=int, default=1)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--savedir", type=str, default="/path/to/save_dir/")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--task_type", type=str, default="multilabel", choices=["multilabel", "classification"])
    parser.add_argument("--task", type=str, default="MVSA_Single", choices=["CrisisMMD/damage", "CrisisMMD/humanitarian", "CrisisMMD/informative",
                                                                            "N24News/abstract", "N24News/caption", "N24News/headline",
                                                                            "food101","MVSA_Single"])
    parser.add_argument("--warmup", type=float, default=0.1)
    parser.add_argument("--clip_grad", type=float, default=0)
    parser.add_argument("--n_estimators", type=int, default=500)
    parser.add_argument("--regressor", type=int, default=0)

    parser.add_argument("--lr_text_enc", type=float, default=5e-5)
    parser.add_argument("--lr_img_enc", type=float, default=5e-5)
    parser.add_argument("--lr_text_cls", type=float, default=1e-4)
    parser.add_argument("--lr_img_cls", type=float, default=1e-4)
    parser.add_argument("--weight_decay_enc", type=float, default=1e-4)
    parser.add_argument("--weight_decay_cls", type=float, default=1e-4)

    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--transform", type=str, default="vit")
    parser.add_argument("--lr_detail", type=str, default="n")

    parser.add_argument("--LOAD_SIZE", type=int, default=256)
    parser.add_argument("--FINE_SIZE", type=int, default=224)
    parser.add_argument("--vision_model", type=str, default="resnet")
    parser.add_argument("--text_model", type=str, default="transforms_bert")


def model_forward(i_epoch, model, args, criterion, batch):
    txt, segment, mask, img, tgt,idx = batch
    txt, img = txt.cuda(), img.cuda()
    mask, segment = mask.cuda(), segment.cuda()
    txt_img, txt_img_alpha = model(txt, mask, segment, img)
    tgt = tgt.cuda()
    loss = emc_loss(txt_img, txt_img_alpha, tgt)
    return loss, txt_img_alpha, tgt, txt_img

def get_criterion(args):
    criterion = nn.CrossEntropyLoss()
    return criterion


def get_optimizer(model, args):
    if args.lr_detail == "y":
        text_enc_param = list(model.txtclf.text_encoder.named_parameters())
        text_clf_param = list(model.txtclf.clf.parameters())

        img_enc_param = list(model.imgclf.image_encoder.parameters())
        img_clf_param = list(model.imgclf.clf.parameters())

        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

        lr_text_tfm = args.lr_text_enc #5e-5 #2e-5,
        lr_img_tfm = args.lr_img_enc #5e-5#
        lr_text_cls = args.lr_text_cls #1e-4 # 5e-5,
        lr_img_cls = args.lr_img_cls #1e-4
        weight_decay_tfm = args.weight_decay_enc #1e-4
        weight_decay_other = args.weight_decay_cls #1e-4

        optimizer_grouped_parameters = [
            {"params": [p for n, p in text_enc_param if not any(nd in n for nd in no_decay)],
             "weight_decay": weight_decay_tfm, 'lr': lr_text_tfm},
            {"params": [p for n, p in text_enc_param if any(nd in n for nd in no_decay)], "weight_decay": 0.0,
             'lr': lr_text_tfm},
            {"params": text_clf_param, "weight_decay": weight_decay_other, 'lr': lr_text_cls},
            {"params": img_enc_param, "weight_decay": weight_decay_tfm, 'lr': lr_img_tfm},
            {"params": img_clf_param, "weight_decay": weight_decay_other, 'lr': lr_img_cls},
        ]
        optimizer = optim.Adam(optimizer_grouped_parameters)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)#1e-5
    return optimizer


def get_scheduler(optimizer, args):
    return optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "max", patience=args.lr_patience, verbose=True, factor=args.lr_factor
    )


def train(args):
    set_seed(args.seed)
    args.savedir = os.path.join(args.savedir, args.name)
    os.makedirs(args.savedir, exist_ok=True)

    train_loader, val_loader, test_loaders = get_data_loaders(args)
    model = get_model(args)
    criterion = get_criterion(args)
    optimizer = get_optimizer(model, args)
    scheduler = get_scheduler(optimizer, args)
    if args.warmup >= 1:
        scheduler = GradualWarmupScheduler(optimizer, total_epoch=int(args.warmup), after_scheduler=scheduler)

    writer = SummaryWriter(args.savedir)
    logger = create_logger("%s/logfile.log" % args.savedir, args)
    model.wmodel.logger = logger

    model.cuda()

    torch.save(args, os.path.join(args.savedir, "args.pt"))

    start_epoch, global_step, n_no_improve, best_metric, best_train_metric, train_no_improve, best_all_metric = 0, 0, 0, -np.inf, -np.inf, 0, 0
    best_epoch = 0

    if os.path.exists(os.path.join(args.savedir, "checkpoint.pt")):
        checkpoint = torch.load(os.path.join(args.savedir, "checkpoint.pt"))
        i_epoch = start_epoch = checkpoint["epoch"]
        n_no_improve = checkpoint["n_no_improve"]
        best_metric = checkpoint["best_metric"]
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if args.warmup >= 1:
            scheduler.after_scheduler.load_state_dict(checkpoint["scheduler"])
            scheduler.last_epoch = start_epoch - 1
        scheduler.load_state_dict(checkpoint["scheduler"])

    logger.info("Training..")
    for i_epoch in range(start_epoch, args.max_epochs):
        train_losses = []
        model.train()
        optimizer.zero_grad()

        for batch in tqdm(train_loader, total=len(train_loader)):
            loss, _, _, _ = model_forward(i_epoch, model, args, criterion, batch)
            train_losses.append(loss.item())
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            loss.backward()
            global_step += 1
            if global_step % args.gradient_accumulation_steps == 0:
                if args.clip_grad > 0:
                    clip_grad_value_(model.parameters(), clip_value=args.clip_grad)
                optimizer.step()
                optimizer.zero_grad()

        logger.info(f"[{i_epoch + 1}/{args.max_epochs}] Train Loss: {np.mean(train_losses):.4f}")

        writer.add_scalars('Log/Loss', {"train": np.mean(train_losses).item()}, i_epoch + 1)

        model.eval()

        metrics = model_eval(i_epoch, val_loader, model, args, criterion)
        logger.info("Val  : " + logger_str(metrics))
        log_metrics("Val  ", metrics, args, logger)

        writer.add_scalars('Log/Loss', {"val": metrics["loss"]}, i_epoch + 1)
        writer.add_scalars('Log/Accuracy', {"val": metrics["acc"]}, i_epoch + 1)

        tuning_metric = (
            metrics["micro_f1"] if args.task_type == "multilabel" else metrics["acc"]
        )
        scheduler.step(tuning_metric)
        is_improvement = (tuning_metric > best_metric)
        if is_improvement:
            best_metric = tuning_metric
            n_no_improve = 0
            best_epoch = i_epoch + 1
        else:
            n_no_improve += 1

        save_checkpoint(
            {
                "epoch": i_epoch + 1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "n_no_improve": n_no_improve,
                "best_metric": best_metric,
                "best_all_metric":best_all_metric,
        },
            is_improvement,
            args.savedir,
        )
        if n_no_improve >= args.patience:
            logger.info("No improvement. Breaking out of loop.")
            break
    logger.info(f"Reload best model epoch: {best_epoch}")
    load_checkpoint(model, os.path.join(args.savedir, "model_best.pt"))

    model.eval()
    metrics = model_eval(i_epoch + 1, train_loader, model, args, criterion, store_preds=True, type_="train")
    logger.info("Train: " + logger_str(metrics))
    log_metrics("Train", metrics, args, logger)
    train = metrics["data"]

    metrics = model_eval(i_epoch + 1, val_loader, model, args, criterion, store_preds=True, type_="val")
    logger.info("Val  : " + logger_str(metrics))
    log_metrics("Val  ", metrics, args, logger)
    val = metrics["data"]

    outs = {}
    for key in val["outs"].keys():
        outs[key] = np.row_stack((train["outs"][key], val["outs"][key]))
    val = {
        "tgts": np.hstack((train["tgts"], val["tgts"])),
        "outs": outs,
    }

    for test_name, test_loader in test_loaders.items():
        test_metrics = model_eval(
            i_epoch + 1, test_loader, model, args, criterion, store_preds=True
        )
        logger.info("Test : " + logger_str(test_metrics))
        log_metrics(f"Test - {test_name}", test_metrics, args, logger)
        print(logger_str(test_metrics))
        if args.regressor:
            test = test_metrics["data"]
            preds = model.wmodel.train_predict(val, test)
            preds = preds.argmax(-1)
            acc = accuracy_score(test["tgts"], preds)
            print(f"test acc: {acc:.4f}")
        else:
            print(f"test acc: {test_metrics['acc']:.4f}")


def model_eval(i_epoch, data, model, args, criterion, store_preds=False, type_="test"):
    with torch.no_grad():
        losses, preds, tgts  = [], [], []
        preds_ = defaultdict(list)
        outs = defaultdict(list)
        for batch in data:
            loss, out, tgt, out_ = model_forward(i_epoch, model, args, criterion, batch)
            losses.append(loss.item())
            pred = torch.nn.functional.softmax(out, dim=1).argmax(dim=1).cpu().detach().numpy()
            preds.append(pred)
            for i in range(len(out_)):
                pred_ = torch.nn.functional.softmax(out_[i], dim=1).argmax(dim=1).cpu().detach().numpy()
                preds_[i].append(pred_)
                outs[i].append(out_[i].cpu().detach().numpy())

            tgt = tgt.cpu().detach().numpy()
            tgts.append(tgt)

    metrics = {"loss": np.mean(losses).item()}
    if args.task_type == "multilabel":
        tgts = np.vstack(tgts)
        preds = np.vstack(preds)
        metrics["macro_f1"] = f1_score(tgts, preds, average="macro")
        metrics["micro_f1"] = f1_score(tgts, preds, average="micro")
    else:
        tgts = [l for sl in tgts for l in sl]
        preds = [l for sl in preds for l in sl]
        metrics["acc"] = accuracy_score(tgts, preds)
        accs, accs_, accs_m, accs_a, acc_dict = [], None, None, 0, {}
        tgts = np.hstack(tgts)
        for i in preds_.keys():
            pred_ = np.hstack(preds_[i])
            acc_ = pred_ == tgts
            accs_ = acc_ if accs_ is None else accs_ & acc_
            accs_m = acc_ if accs_m is None else accs_m | acc_
            accs.append(acc_.sum() / len(acc_))
            accs_a += accs[i]
            outs[i] = np.row_stack(outs[i])
        acc_dict["MI"] = accs
        acc_dict["AM"] = accs_a
        acc_dict["MIN"] = accs_.sum() / len(accs_)
        acc_dict["MAX"] = accs_m.sum() / len(accs_m)
        metrics["accs"] = acc_dict

        save_data = {}
        save_data["tgts"] = tgts
        save_data["outs"] = outs
        if store_preds:
            path = f"{args.savedir}/{args.name}_{type_}_{args.noise}.pkl"
            print(f"Save to {path}")
            with open(path, "wb") as f:
                pickle.dump(save_data, f)
        metrics["data"] = save_data

    if store_preds:
        store_preds_to_disk(tgts, preds, args)

    return metrics

def logger_str(metrics):
    logger_ = "Acc: "
    for i in range(len(metrics["accs"]['MI'])):
        logger_ += f"M{i+1}: {metrics['accs']['MI'][i]:.4f} "
    logger_ += f"AM: {metrics['accs']['AM']:.4f} "
    logger_ += f"Min: {metrics['accs']['MIN']:.4f} "
    logger_ += f"Max: {metrics['accs']['MAX']:.4f} "
    logger_ += f"M: {metrics['acc']:.4f} "

    return logger_


def cli_main():
    parser = argparse.ArgumentParser(description="Train Models")
    get_args(parser)
    args, remaining_args = parser.parse_known_args()
    args.annealing_epoch=10
    assert remaining_args == [], remaining_args
    train(args)


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    cli_main()
