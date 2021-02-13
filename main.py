import os
import os.path as osp
import time

import torch
from loguru import logger
from pytorch_metric_learning import losses, miners

from config import get_config
from data import get_train_dataloader, get_test_dataloader
from model import get_model
from utils import get_map, init_train_env


def main(args):
    init_train_env(args.work_dir, args.seed)

    params = []
    for k, v in vars(args).items():
        params.append(f"{k}: {v}")
    logger.info("\n" + "\n".join(params))

    train_dataloader, num_train = get_train_dataloader(
        args.ilsvrc_data_dir,
        args.batch_size,
        args.sample_per_class,
        args.num_workers,
        args.gpu,
    )
    ilsvrc_query_loader, \
    ilsvrc_gallery_loader, \
    cifar_query_loader, \
    cifar_gallery_loader, \
    nuswide_query_loader, \
    nuswide_gallery_loader = get_test_dataloader(
        args.ilsvrc_data_dir, 
        args.cifar_data_dir, 
        args.nuswide_data_dir, 
        args.batch_size, 
        args.num_workers,
    )

    model = get_model(args.arch, args.bits).to(args.device)
    loss_function = losses.TripletMarginLoss(margin=args.margin)
    miner = miners.TripletMarginMiner(margin=args.margin, type_of_triplets="semihard")
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.max_epochs * (num_train // args.batch_size))

    if args.fp16:
        from torch.cuda.amp import GradScaler, autocast
        scaler = GradScaler()
    
    if args.tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(osp.join(args.work_dir, "tensorboard"))

    checkpoint_name = None
    current_step = 0
    ilsvrc_best_mAP = cifar_best_mAP = nuswide_best_mAP = 0.
    running_loss = 0.
    start_time = time.time()
    for epoch in range(1, args.max_epochs+1):
        model.train()
        train_dataloader.reset()
        for batch in train_dataloader:
            current_step += 1
            data = batch[0]["data"].to(args.device)
            labels = batch[0]["label"].to(args.device).squeeze()
            optimizer.zero_grad()

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
            scheduler.step()
            
            if current_step % args.log_step == 0:
                running_loss /= args.log_step
                logger.info("Epoch: {:>2d} Step: {:>4d} Time: {:>.2f} Loss: {:>.2f}".format(epoch, current_step, time.time()-start_time, running_loss))
                if args.tensorboard:
                    writer.add_scalar("Loss/total", running_loss, current_step)
                running_loss = 0.
            
        if epoch % args.eval_step == 0:
            ilsvrc_mAP = get_map(ilsvrc_query_loader, ilsvrc_gallery_loader, model, args.bits, args.device, 1000)
            ilsvrc_best_mAP = max(ilsvrc_mAP, ilsvrc_best_mAP)

            cifar_mAP = get_map(cifar_query_loader, cifar_gallery_loader, model, args.bits, args.device, -1)
            cifar_best_mAP = max(cifar_mAP, cifar_best_mAP)
            logger.debug(cifar_mAP)

            nuswide_mAP = get_map(nuswide_query_loader, nuswide_gallery_loader, model, args.bits, args.device, 5000)
            nuswide_best_mAP = max(nuswide_mAP, nuswide_best_mAP)

            logger.info("\ncurrent epoch: {}\nilsvrc map: {:.4f}\tilsvrc best map: {:.4f}\ncifar map: {:.4f}\tcifar best map: {:.4f}\nnus-wide map: {:.4f}\tnus-wide best map: {:.4f}".format(epoch, ilsvrc_mAP, ilsvrc_best_mAP, cifar_mAP, cifar_best_mAP, nuswide_mAP, nuswide_best_mAP))

            if args.tensorboard:
                writer.add_scalar("mAP/ILSVRC2012", ilsvrc_mAP, epoch)
                writer.add_scalar("mAP/CIFAR-10", cifar_mAP, epoch)
                writer.add_scalar("mAP/NUS-WIDE", nuswide_mAP, epoch)
            if not args.no_save:
                checkpoint_name = "Epoch_{}_ilsvrc_mAP_{:.4f}_cifar_mAP_{:.4f}_nuswide_mAP_{:.4f}.ckpt".format(epoch, ilsvrc_mAP, cifar_mAP, nuswide_mAP)
                checkpoint_pth = osp.join(args.work_dir, checkpoint_name)
                torch.save(model.state_dict(), checkpoint_pth)


if __name__ == "__main__":
    args = get_config()
    for bits in [128, 256, 512, 1024]:
        args.bits = bits
        main(args)
