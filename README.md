# SVD-Attack
Boosting Adversarial Transferability via Fusing Logits of Top-1 Decomposed Feature [Link](https://arxiv.org/pdf/2305.01361.pdf)

![Learning Algo](/framework.png)

## Requirements

- python 3.9
- torch 1.8
- pretrainedmodels 0.7
- numpy 1.19
- pandas 1.2


## Implementation
- **Prepare models**

  Download pretrained PyTorch models [here](https://github.com/ylhz/tf_to_pytorch_model), which are converted from widely used Tensorflow models. Then put these models into `./models/`

- **Generate adversarial examples by SVD under inception-v3 ** -


  ```bash
  # Implement MI-FGSM, DI-FGSM, TI-FGSM or TI-DIM
  CUDA_VISIBLE_DEVICES=gpuid python MI_FGSM.py
  # Implement PI-FGSM or PI-TI-DI-FGSM
  CUDA_VISIBLE_DEVICES=gpuid python PI_FGSM.py
  # Implement SI_NI_FGSM, SI_NI_TI-DIM
  CUDA_VISIBLE_DEVICES=gpuid python SI_NI_FGSM.py
  # Implement VT_MI_FGSM
  CUDA_VISIBLE_DEVICES=gpuid python VT_MI_FGSM.py
  # Implement S2I_FGSM or S2I_TI_DIM
  CUDA_VISIBLE_DEVICES=gpuid python S2I_FGSM.py
  ```
  where `gpuid` can be set to any free GPU ID in your machine. And adversarial examples will be generated in directory `./adv_img`.
  
- **Evaluations on normally trained models**

  Running `verify.py` to evaluate the attack  success rate
  ```bash
  python verify.py
  ```

- **Evaluations on defenses**

    To evaluate the attack success rates on defense models, we test eight defense models which contain three adversarial trained models (Inc-v3<sub>*ens3*</sub>, Inc-v3<sub>*ens4*</sub>, IncRes-v2<sub>*ens*</sub>) and six more advanced models (HGD, R&P, NIPS-r3, RS, JPEG, NRP).

    - [Inc-v3<sub>*ens3*</sub>,Inc-v3<sub>*ens4*</sub>,IncRes-v2<sub>*ens*</sub>](https://github.com/ylhz/tf_to_pytorch_model):  You can directly run `verify.py` to test these models.
    - [HGD](https://github.com/lfz/Guided-Denoise), [R&P](https://github.com/cihangxie/NIPS2017_adv_challenge_defense), [NIPS-r3](https://github.com/anlthms/nips-2017/tree/master/mmd): We directly run the code from the corresponding official repo.
    - [RS](https://github.com/locuslab/smoothing): noise=0.25, N=100, skip=100. Download it from corresponding official repo.
    - [JPEG](https://github.com/JHL-HUST/VT/blob/main/third_party/jpeg.py): Refer to [here](https://github.com/JHL-HUST/VT/blob/main/third_party/jpeg.py).
    - [NRP](https://github.com/Muzammal-Naseer/NRP): purifier=NRP, dynamic=True, base_model=Inc-v3<sub>*ens3*</sub>. Download it from corresponding official repo.
    

   
