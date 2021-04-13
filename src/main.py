import os
import os.path as osp
import time

import torch
from loguru import logger
from pytorch_metric_learning import losses, miners

from config import get_config
from data import get_dataloader
from model import get_model
from utils import get_map, init_train_env
import wandb


def main(args):
    # Init
    init_train_env(args.work_dir, args.seed, args.gpu)

    # Logger
    if args.wandb:
        wandb.init(config=args, project="Hash-Pretrain", dir=args.work_dir)
    params = []
    for k, v in vars(args).items():
        params.append(f"{k}: {v}")
    logger.info("\n" + "\n".join(params))

    # Get dataloader
    ilsvrc_train_loader, cifar_query_loader, cifar_gallery_loader, nuswide_query_loader, nuswide_gallery_loader = get_dataloader(
        args.ilsvrc_data_dir,
        args.cifar_data_dir,
        args.nuswide_data_dir,
        args.train_batch_size,
        args.test_batch_size,
        args.samples_per_class,
        args.num_workers,
    )

    # Get model
    model = get_model(args.arch, args.bits).cuda()

    # Get loss function
    loss_function = losses.TripletMarginLoss(margin=args.margin)

    # Get hard examples miner
    miner = miners.TripletMarginMiner(margin=args.margin, type_of_triplets="semihard")

    # Get optimizer and scheduler
    #optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.max_epochs * (num_train // args.batch_size))
    last_layer_params = list(map(id, model.fc.parameters()))
    base_params = filter(lambda p: id(p) not in last_layer_params,
                        model.parameters())
    optimizer = torch.optim.AdamW(
        [
            {'params': base_params},
            {'params': model.fc.parameters(), 'lr': args.lr * 10}
        ],
        lr=args.lr, 
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.3, patience=3, verbose=True, threshold=0.01, min_lr=args.lr/1e4)

    # FP16 training
    if args.fp16:
        from torch.cuda.amp import GradScaler, autocast
        scaler = GradScaler()

    checkpoint_name = None
    cifar_best_mAP = nuswide_best_mAP = 0.
    running_loss = 0.
    start_time = time.time()
    cur_iter = 0
    train_iter = iter(ilsvrc_train_loader)
    while cur_iter < args.max_iters:
        try:
            data, labels = train_iter.next()
        except StopIteration:
            train_iter = iter(ilsvrc_train_loader)
            data, labels = train_iter.next()
        cur_iter += 1
        model.train()
        optimizer.zero_grad()
        data, labels = data.cuda(non_blocking=True), labels.cuda(non_blocking=True)

        if args.fp16:
            with autocast():
                embeddings = model(data)
                mining_pairs = miner(embeddings, labels)
                loss = loss_function(embeddings, labels, mining_pairs)
                running_loss += loss.item()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            embeddings = model(data)
            mining_pairs = miner(embeddings, labels)
            loss = loss_function(embeddings, labels, mining_pairs)
            running_loss += loss.item()
            loss.backward()
            optimizer.step()
            
        if cur_iter % args.log_step == 0:
            running_loss /= args.log_step
            logger.info("Iter: {:>2d} Time: {:>.2f} Loss: {:>.4f}".format(cur_iter, time.time()-start_time, running_loss))
            if args.wandb:
                wandb.log({
                    "Loss/total": running_loss,
                    "Iter": cur_iter,
                })
            running_loss = 0.
            
        if cur_iter % args.eval_step == 0:
            logger.info(f"Iter: {cur_iter}, Evaluating...")

            cifar_mAP = get_map(
                cifar_query_loader, 
                cifar_gallery_loader, 
                model, 
                args.bits, 
                -1,
            )
            cifar_best_mAP = max(cifar_mAP, cifar_best_mAP)
            logger.info("CIFAR-10")
            logger.info("current iter MAP: {:.4f}\tbest MAP: {:.4f}".format(cifar_mAP, cifar_best_mAP))

            nuswide_mAP = get_map(
                nuswide_query_loader, 
                nuswide_gallery_loader, 
                model, 
                args.bits, 
                -1,
            )
            nuswide_best_mAP = max(nuswide_mAP, nuswide_best_mAP)
            logger.info("NUS-WIDE")
            logger.info("current iter MAP: {:.4f}\tbest MAP: {:.4f}".format(nuswide_mAP, nuswide_best_mAP))

            scheduler.step(cifar_mAP)

            if args.wandb:
                wandb.log({
                    "Metric/CIFAR-10": cifar_mAP,
                    "Metric/NUS-WIDE": nuswide_mAP,
                    "Iter": cur_iter,
                })
            if not args.no_save:
                checkpoint_name = "Iter_{}_cifar_mAP_{:.4f}_nuswide_mAP_{:.4f}.ckpt".format(cur_iter, cifar_mAP, nuswide_mAP)
                checkpoint_pth = osp.join(args.work_dir, checkpoint_name)
                torch.save(model.state_dict(), checkpoint_pth)


if __name__ == "__main__":
    args = get_config()
    main(args)
