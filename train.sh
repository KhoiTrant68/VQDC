accelerate launch --multi_gpu train.py --base configs/stage1/dqvae_dual_feat_imagenet.yml --epochs 6 --mode feat --with_tracking
accelerate launch --multi_gpu train.py --base configs/stage1/dqvae_dual_entropy_imagenet.yml --epochs 6 --mode entropy --with_tracking
accelerate launch --multi_gpu train.py --base configs/stage1/dqvae_triple_feat_imagenet.yml --epochs 6 --mode feat --with_tracking
# accelerate launch --multi_gpu train.py --base configs/stage1/dqvae_triple_entropy_imagenet.yml --epochs 6 --mode entropy --with_tracking


