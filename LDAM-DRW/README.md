### Training 

```bash
!python drive/MyDrive/Colab/LDAM-DRW/cifar_train.py --dataset mnist --gpu 0 --imb_type exp --imb_factor 0.01 --loss_type LDAM --train_rule DRW --epochs 1
```

### Reference

If you find our paper and repo useful, please cite as

```
@inproceedings{cao2019learning,
  title={Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss},
  author={Cao, Kaidi and Wei, Colin and Gaidon, Adrien and Arechiga, Nikos and Ma, Tengyu},
  booktitle={Advances in Neural Information Processing Systems},
  year={2019}
}
```
