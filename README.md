# Stimulative training of residual networks: A social psychology perspective of loafing (NeurIPS'22) [[NIPS22]](https://proceedings.neurips.cc/paper_files/paper/2022/hash/1757af1fe1429801bdf3abf5600f8bba-Abstract-Conference.html)
Abstract: Residual networks have shown great success and become indispensable in today’s deep models. In this work, we aim to re-investigate the training process of residual networks from a novel social psychology perspective of loafing, and further propose a new training strategy to strengthen the performance of residual networks. As residual networks can be viewed as ensembles of relatively shallow networks (i.e., unraveled view) in prior works, we also start from such view and consider that the final performance of a residual network is co-determined by a group of sub-networks. Inspired by the social loafing problem of social psychology, we find that residual networks invariably suffer from similar problem, where sub-networks in a residual network are prone to exert less effort when working as part of the group compared to working alone. We define this previously overlooked problem as network loafing. As social loafing will ultimately cause the low individual productivity and the reduced overall performance, network loafing will also hinder the performance of a given residual network and its sub-networks. Referring to the solutions of social psychology, we propose stimulative training, which randomly samples a residual sub-network and calculates the KL-divergence loss between the sampled sub-network and the given residual network, to act as extra supervision for sub-networks and make the overall goal consistent. Comprehensive empirical results and theoretical analyses verify that stimulative training can well handle the loafing problem, and improve the performance of a residual network by improving the performance of its sub-networks.

# Stimulative training++: Go beyond the performance limits of residual networks [[arxiv]](https://arxiv.org/abs/2305.02507)
Abstract: Residual networks have shown great success and become indispensable in recent deep neural network models. In this work, we aim to re-investigate the training process of residual networks from a novel perspective of loafing, and further propose a new training scheme as well as three improved strategies for boosting residual networks beyond their performance limits. Previous research has suggested that residual networks can be considered as ensembles of shallow networks, which implies that the final performance of a residual network is influenced by a group of subnetworks. Furthermore, we identify a previously overlooked problem, where subnetworks within a residual network are prone to exert less effort when working as part of a group compared to working alone. We define this problem as network loafing. Since network loafing may inevitably cause the sub-par performance of the residual network, we propose a novel training scheme called stimulative training, which randomly samples a residual subnetwork and calculates the KL divergence loss between the sampled subnetwork and the given residual network for extra supervision. In order to unleash the potential of stimulative training, we further propose three simple-yet-effective strategies, including a novel KL- loss that only aligns the network logits direction, random smaller inputs for subnetworks, and inter-stage sampling rules. Comprehensive experiments and analysis verify the effectiveness of stimulative training as well as its three improved strategies. For example, the proposed method can boost the performance of ResNet50 on ImageNet to 80.5% Top1 accuracy without using any extra data, model, trick, or changing the structure. With only uniform augment, the performance can be further improved to 81.0% Top1 accuracy, better than the best training recipes provided by Timm library and PyTorch official version. We also verify its superiority on various typical models, datasets, and tasks and give some theoretical analysis. As such, we advocate utilizing the proposed method as a general and next-generation technology to train residual networks.

# Run
1. ImageNet experiments are conducted on 2 A100 80G GPUs.

To train ResNet-50 with common training
```
python train.py --exp 'CT_1_0_50_E200_2G' --main_coef=1.0 --kl_coef=0 --epoch=200 
```

To train ResNet-50 with stimulative training
```
python train.py --exp 'ST_1_1_50_E200_2G' --epoch=200 
```

To train ResNet-50 with stimulative training + KD-
```
python train.py --exp 'ST_1_1_50_E200_2G_norm_A1_T1' --epoch=200 --norm_kd --amplitude=1 --Temp=1 
```

To train ResNet-50 with stimulative training + KD-(snet5)  
```
python train.py --exp 'ST_1_1_50_E200_2G_norm_Snet5_A1_T1' --epoch=200 --norm_kd --multi_Snet=5 --amplitude=1 --Temp=1 
```

To train ResNet-50 with stimulative training + KD-(snet5) + random smaller inputs for subnets
```
python train.py --exp 'ST_1_1_50_E200_2G_norm_Snet5_A1_T1_Dtrans7' --epoch=200 --norm_kd --multi_Snet=5 --amplitude=1 --Temp=1 --Snet_Dtrans --Dtrans='Dtrans7'
```

To train ResNet-50 with stimulative training + KD-(snet5) + random smaller inputs for subnets
```
python train.py --exp 'ST_1_1_50_E200_2G_norm_Snet5_A1_T1_Dtrans7' --epoch=200 --norm_kd --multi_Snet=5 --amplitude=1 --Temp=1 --Snet_Dtrans --Dtrans='Dtrans7'
```

<!-- To test a pre-trained model,

Modify `test_only: False` to `test_only: True` in .yml file to enable testing. 

Modify `pretrained: /PATH/TO/YOUR/WEIGHTS` to assign pre-trained weights. -->

<!-- # Results
1. ImageNet classification accuacy.  -->



# Citation
If you find this useful in your work, please consider citing,
```
@article{ye2023stimulative,
  title={Stimulative training++: Go beyond the performance limits of residual networks},
  author={Ye, Peng and He, Tong and Tang, Shengji and Li, Baopu and Chen, Tao and Bai, Lei and Ouyang, Wanli},
  journal={arXiv preprint arXiv:2305.02507},
  year={2023}
}

@article{ye2022stimulative,
  title={Stimulative training of residual networks: A social psychology perspective of loafing},
  author={Ye, Peng and Tang, Shengji and Li, Baopu and Chen, Tao and Ouyang, Wanli},
  journal={Advances in Neural Information Processing Systems},
  volume={35},
  pages={3596--3608},
  year={2022}
}

```
