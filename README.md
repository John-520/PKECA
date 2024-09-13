# PKECA
Code for PKECA


# [COMIND 2025] PKECA code

This is the source code for "<b>Prior knowledge embedding convolutional autoencoder: A single-source domain generalized fault diagnosis framework under small samples</b>". 

## Abstract
The proposed transfer learning-based fault diagnosis models have achieved good results in multi-source domain generalization (MDG) tasks. However, research on single-source domain generalization (SDG) is relatively scarce, and the limited availability of small training samples is seldom considered. Therefore, to address the insufficient feature extraction capability and poor generalization performance of existing models on single-source domain small sample data, a novel single-source domain generalization fault diagnosis (SDGFD) framework, the prior knowledge embedded convolutional autoencoder (PKECA), is proposed. During the training phase, first, single-source domain data are used to construct prior features based on the time domain, frequency domain, and time-frequency domain. Second, a prior knowledge embedding structure based on the convolutional autoencoder is built, which compresses the prior knowledge and original vibration data into a high-dimensional space of consistent dimensions, embedding the prior knowledge into the features corresponding to the vibration data using a mean squared error loss function. Subsequently, the proposed centroid-based self-supervised learning (CBSSL) strategy further constrains high-dimensional features, improving the generalization ability. The designed sparse regularized activation (SRA) function significantly enhances the regularization effect on features. During the testing phase, it is only necessary to input the data from the unknown domain to identify the fault types. The experimental results show that the proposed method achieves superior performance in fault diagnosis tasks involving cross-speed, time-varying speed, and small sample data in SDGFD, demonstrating that PKECA has strong generalizability. 

## Proposed Network

![image](https://github.com/user-attachments/assets/d22fecfb-eb18-45d0-a3cb-df8c82b861a4)





## Dataset Preparation

**You can find the dataset here:
【1】Case Western Reserve University Bearing Data Center Website [Online] Available: http://csegroups.case.edu/bearingdatacenter/home [DB]. 
【2】Huang H, Baddour N, Liang M. Multiple time-frequency curve extraction Matlab code and its application to automatic bearing fault diagnosis under time-varying speed conditions [J]. MethodsX, 2019, 6: 1415-32.https://www.sciencedirect.com/science/article/pii/S2215016119301402.
And the paper can be downloaded from my personal homepage [here](https://john-520.github.io/).**




## Contact

If you have any questions, please feel free to contact me:

- **Name:** Feiyu Lu
- **Email:** 21117039@bjtu.edu.cn
- **微信公众号:** 轴承智能故障诊断<img width="300" alt="二维码" src="https://github.com/user-attachments/assets/77a67e89-3214-4ff4-8256-01c75ec49e4b">


## Citation

If you find this paper and repository useful, please cite our paper 😊.

```
@article{LU2025104169,
title = {Prior knowledge embedding convolutional autoencoder: A single-source domain generalized fault diagnosis framework under small samples},
journal = {Computers in Industry},
volume = {164},
pages = {104169},
year = {2025},
issn = {0166-3615},
doi = {https://doi.org/10.1016/j.compind.2024.104169},
url = {https://www.sciencedirect.com/science/article/pii/S0166361524000976},
author = {Feiyu Lu and Qingbin Tong and Xuedong Jiang and Xin Du and Jianjun Xu and Jingyi Huo},
keywords = {Fault diagnosis, Single-source domain generalization, Centroid-based self-supervised learning},
}
```

```
@article{LU2024102536,
title = {Towards multi-scene learning: A novel cross-domain adaptation model based on sparse filter for traction motor bearing fault diagnosis in high-speed EMU},
journal = {Advanced Engineering Informatics},
volume = {60},
pages = {102536},
year = {2024},
issn = {1474-0346},
doi = {https://doi.org/10.1016/j.aei.2024.102536},
url = {https://www.sciencedirect.com/science/article/pii/S1474034624001848},
author = {Feiyu Lu and Qingbin Tong and Jianjun Xu and Ziwei Feng and Xin Wang and Jingyi Huo and Qingzhu Wan},
keywords = {Bearing fault diagnosis, Sparse filter, Cross-domain adaptation},
```
