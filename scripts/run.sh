python src/main.py\
    --bits 128\
    --ilsvrc_data_dir /dataset/ILSVRC2012\
    --cifar_data_dir /dataset/cifar-10\
    --nuswide_data_dir /dataset/nus-wide\
    --train_batch_size 128\
    --samples_per_class 8\
    --eval_step 2000\
    --num_workers 6\
    --gpu 0\
    --fp16\
    --lr 0.00001\
    --wandb \
    --margin 0.1
