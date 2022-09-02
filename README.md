Pytorch implementation of COPLE-Net with the proposed margin calibration method.

Running dependencies:
Python 3.7 (Anaconda)
Pytorch 1.1
tqdm

Before running the code, please edit the values of `root` in datasets/covid19_lesion.py and datasets/robotic_instrument.py, which are the image/data directories.

To train the segmentation model, simply run the command with the arguments like follows:

python -W ignore train.py \
    --dataset robotic_instrument \
    --task parts \
    --margin_loss \
    --date 0907 \
    --batch_size 6 \
    --max_epoch 50 \
    --adamw \
    --lr 1e-4 \
    --exp robotic_instrument_parts_mg 
    

To evaluate the model, just run:

python -W ignore eval.py \
    --dataset robotic_instrument\
    --task parts \
    --dump_imgs \
    --method mg \
    --snapshot <MODEL_CHECKPOINT_PATH>

If you feel this work useful to your research, please kindly cite the paper as follows:

@article{yu2022distribution,
 title={Distribution-Aware Margin Calibration for Semantic Segmentation in Images},
 author={Yu, Litao and Li, Zhibin and Xu, Min and Gao, Yongsheng and Luo, Jiebo and Zhang, Jian},
 journal={International Journal of Computer Vision},
 volume={130},
 number={1},
 pages={95--110},
 year={2022},
 publisher={Springer}
}
