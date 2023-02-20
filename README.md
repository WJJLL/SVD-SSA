# SVD-SSA
SVD-based feature decomposition atttack

![Learning Algo](/framework.pdf)

## Requirements

- python 3.8
- torch 1.8
- pretrainedmodels 0.7
- numpy 1.19
- pandas 1.2


## Implementation

- **Prepare Dataset**

The 1000 images from the NIPS 2017 ImageNet-Compatible dataset are provided in the folder ```dataset/images```, along with their metadata in  ```dataset/images.csv```. More details about this dataset can be found in its [official repository](https://github.com/cleverhans-lab/cleverhans/tree/master/cleverhans_v3.1.0/examples/nips17_adversarial_competition/dataset).

- **Prepare models**

  Download pretrained PyTorch models [here](https://github.com/ylhz/tf_to_pytorch_model), which are converted from widely used Tensorflow models. Then put these models into `./tf_models/`

- **Generate adversarial examples**

  Using `SFA.py` to implement our S<sup3</sup>I-FGSM,  you can run this attack as following
  
  ```bash
  
   #The middle layer for the SVD-based feature decomposition are Mixed-6e for Inception-v3, which is layer before the parameter 'AuxLogits' 
  CUDA_VISIBLE_DEVICES=gpuid python SFA.py --batch_size 20 --model_name inceptionv3 --layer AuxLogits
   #The middle layer for the SVD-based feature decomposition are the last layer of block3 for Resnet-152, which is layer before the parameter 'layer4' 
  CUDA_VISIBLE_DEVICES=gpuid python SFA.py --batch_size 20 --model_name resnet152 --layer layer4
   #The middle layer for the SVD-based feature decomposition are the Mixed-6a for Inception-Resnet-v2 , which is layer before the parameter 'mixed_7a' 
  CUDA_VISIBLE_DEVICES=gpuid python SFA.py --batch_size 20 --model_name inceptionresnetv2 --layer mixed_7a
   #The middle layer for the SVD-based feature decomposition are Mixed-6b for Inception-v4
  CUDA_VISIBLE_DEVICES=gpuid python SFA.py --batch_size 20 --model_name inceptionv4 --layer '17'
  ```
  where `gpuid` can be set to any free GPU ID in your machine. And adversarial examples will be generated in directory `./outputs`.
  
- **Evaluations on normally trained models**

  Running `verify.py` to evaluate the attack  success rate
  Running `verifyDef.py` to evaluate the attack  success rate against adversarial trained model
  ```bash
  CUDA_VISIBLE_DEVICES=gpuid python verify.py
  CUDA_VISIBLE_DEVICES=gpuid python verifyDef.py
  ```

- **Evaluations on defenses**

    To evaluate the attack success rates on defense models, we test eight defense models which contain three adversarial trained models (Inc-v3<sub>*ens3*</sub>, Inc-v3<sub>*ens4*</sub>, IncRes-v2<sub>*ens*</sub>) and six more advanced models (HGD, R&P, NIPS-r3, RS, JPEG, NRP).

    - [Inc-v3<sub>*ens3*</sub>,Inc-v3<sub>*ens4*</sub>,IncRes-v2<sub>*ens*</sub>](https://github.com/ylhz/tf_to_pytorch_model):  You can directly run `verify.py` to test these models.
    - [HGD](https://github.com/lfz/Guided-Denoise), [R&P](https://github.com/cihangxie/NIPS2017_adv_challenge_defense), [NIPS-r3](https://github.com/anlthms/nips-2017/tree/master/mmd): We directly run the code from the corresponding official repo.
    - [RS](https://github.com/locuslab/smoothing): noise=0.25, N=100, skip=100. Download it from corresponding official repo.
    - [JPEG](https://github.com/JHL-HUST/VT/blob/main/third_party/jpeg.py): Refer to [here](https://github.com/JHL-HUST/VT/blob/main/third_party/jpeg.py).
    - [NRP](https://github.com/Muzammal-Naseer/NRP): purifier=NRP, dynamic=True, base_model=Inc-v3<sub>*ens3*</sub>. Download it from corresponding official repo.
    
    ## References
Code depends on [SSA](https://github.com/yuyang-long/SSA). We thank them for their wonderful code base. 

   
