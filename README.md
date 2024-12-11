# CMI-Attack

The official repository for Collaborative Multimodal Interaction Attack (CMI-Attack).

Paper: *Improving Adversarial Transferability of Vision-Language Pre-training Models via Collaborative Multimodal Interaction* (https://arxiv.org/abs/2403.10883)

## Downloading GloVe Embeddings

To run the code, you need to download the GloVe embeddings file and place it in the main directory. You can download the file from the following link:
[Download GloVe Embeddings](https://drive.google.com/drive/folders/1djXK_lrFzPuXX7oWevokfjmkQhBFc26W?usp=sharing) .
Once downloaded, ensure the file is placed in the root directory of the project.



## About the Implementation

Our code utilizes the evaluation framework from the [SGA](https://github.com/Zoky-2020/SGA) method. To run our code, you need to replace the attack-related components in the SGA code with those from CMI-Attack. Specifically:

#### Original Code (SGA)
```python
img_attacker = ImageAttacker(images_normalize, eps=2/255, steps=10, step_size=0.5/255)
txt_attacker = TextAttacker(ref_model, tokenizer, cls=False, max_length=30, number_perturbation=1,
                            topk=10, threshold_pred_score=0.3)
attacker = Attacker(model, img_attacker, txt_attacker)
```

#### Updated Code (CMI-Attack)
```python
from CMI_Attack import Attack, CMIAttacker

multi_attacker = CMIAttacker(ref_model, tokenizer, cls=False, max_length=30, number_perturbation=1,
                             topk=10, threshold_pred_score=0.3, imgs_eps=2/255, step_size=0.5/255)
attacker = Attack(model, multi_attacker)
```

### Notes
Ensure that you apply these modifications to all relevant files where attack methods are implemented.



## Citation
If you find our paper interesting or helpful to your research, please consider citing it, and feel free to contact fujy23@m.fudan.edu.cn if you have any questions.
```
@misc{fu2024cmi,
      title={Improving Adversarial Transferability of Vision-Language Pre-training Models through Collaborative Multimodal Interaction}, 
      author={Jiyuan Fu and Zhaoyu Chen and Kaixun Jiang and Haijing Guo and Jiafeng Wang and Shuyong Gao and Wenqiang Zhang},
      year={2024},
      eprint={2403.10883},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
}
```

## Acknowledgment

Special thanks to the [SGA](https://github.com/Zoky-2020/SGA) project for providing a valuable reference and support for this work!


## License

The project is **only free for academic research purposes** but has **no authorization for commerce**. 
