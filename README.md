


# Rainmer: Learning multi-view representations for comprehensive image deraining and beyond

The official pytorch implementation of the paper **[Rainmer (ACMMM 2024)](https://dl.acm.org/doi/proceedings/10.1145/3664647)**

>    **Abstract**: We address image deraining under complex backgrounds, diverse rain scenarios, and varying illumination conditions, representing a highly practical and challenging problem. Our approach utilizes synthetic, real-world, and nighttime datasets, wherein rich backgrounds, multiple degradation types, and diverse illumination conditions coexist. The primary challenge in training models on these datasets arises from the discrepancies among them, potentially leading to conflicts or competition during the training period. To address this issue, we first align the distribution of synthetic, real-world and nighttime datasets. Then we propose a novel contrastive learning strategy to extract multi-view (multiple) representations that effectively capture image details, degradations, and illuminations, thereby facilitating training across all datasets. Regarding multiple representations as profitable prompts for deraining, we devise a prompting strategy to integrate them into the decoding process. This contributes to a potent deraining model, dubbed Rainmer. Additionally, a spatial-channel interaction module is introduced to fully exploit cues when extracting multi-view representations. Extensive experiments on synthetic, real-world, and nighttime datasets demonstrate that Rainmer outperforms current representative methods. Moreover, Rainmer achieves superior performance on the All-in-One image restoration dataset, underscoring its effectiveness. Furthermore, quantitative results reveal that Rainmer significantly improves object detection performance on both daytime and night-time rainy datasets. These observations substantiate the potential of Rainmer for practical applications.

**TLDR**: *This paper developed a representation-based method to process rainy images under synthetic, real-world, and night-time conditions.*

## Architecture

![Rainmer](assets/motivation.pdf-1-1.jpg)

## Results

**Derain** (more results can be found in the original paper)

![](https://github.com/Schizophreni/Rainmer/raw/main/assets/derain-vis.pdf-1-4.jpg)

**Detection**

![](https://github.com/Schizophreni/Rainmer/raw/main/assets/detection_more.pdf-1-3.jpg)

**Relation between derain and detection**

![](https://github.com/Schizophreni/Rainmer/raw/main/assets/psnr-ssim-map.pdf-1-2.jpg)

| Dataset (mixed) | Rain100L | Rain100H | Test100 | Test1200 | Test2800 | GT-Rain | GTAV-balance | Avg    |
| --------------- | -------- | -------- | ------- | -------- | -------- | ------- | ------------ | ------ |
| PSNR            | 36.42    | 29.38    | 29.86   | 31.98    | 33.22    | 23.17   | 35.42        | 31.35  |
| SSIM            | 0.9669   | 0.8775   | 0.9171  | 0.9377   | 0.9543   | 0.6594  | 0.9631       | 0.8966 |

## Datasets

- [Rain13K](https://github.com/kuijiang94/MSPFN) 
- [GT-Rain](https://drive.google.com/drive/folders/1NSRl954QPcGIgoyJa_VjQwh_gEaHWPb8)
- [GTAV-NightRain](https://github.com/zkawfanx/GTAV-NightRain)
- [All-in-One](https://github.com/jeya-maria-jose/TransWeather?tab=readme-ov-file)

## Training

To train the model from scratch, please refer to the configuration files in `configs` folder and training scripts `train_anyformer_noresize.py`. 

### Overall procedure

1. Download the datasets and change the `dataset_dict` in `datasets/dataset.py` or `datasets/dataset_noresize.py`:

   ```
   datasets = {
       "synrain": ("/home1/zhangsy/rh/data/derain/MultiRain/{}/rain", "/home1/zhangsy/rh/data/derain/MultiRain/{}/norain"),
       "gt-rain": ("/home1/zhangsy/rh/data/derain/GT-Rain/GT-RAIN_{}", ""),
       "gtav-balance": ("/home1/zhangsy/rh/data/derain/GTAV-balance/", "")
   } 
   # {} indicates 'train' or 'test' placehoder
   ```

2. Change the configuration in `configs` folder. Take `rainmer_resize_alpha.yaml` for example:

   ```
   datasets:
     train:
       dataloader: datasets.dataset_noresize-Anyrainloader
       dataset_names: "synrain,gt-rain,gtav-balance"
       preload: False  # whether to load all images first on storage\
       crop_size: 128
       batch_size: 4
       num_workers: 16
       mode: train
       n_neg: 4
     val:
       dataloader: datasets.dataset_noresize-AnyRainDataset
       dataset_names: "gt-rain"  # only using validation of gt-rain to select checkpoints
       preload: False
       mode: val
       crop_size: 0
   ```

   The `dataset_names` in the yaml file should be consistent to the keys in the dataset_dict

3. Run the training script:

   ```
   python train.py --config $YAML_CONFIG
   ```

### Run the benchmark models

Please refer to the `Benchmarks` folder and configuration under `configs` folder

- Other benchmarks: PromptIR, HCT-FFN, NLCL, AirNet, EfDerain. Please refer to their official repository for training details. (Only need to modify the dataset) We are also begin to integrate more models for the `Benchmarks`.

## Inference

Run the script below to test model:

```
python test.py --config $YAML_CONFIG --testset $TEST --tile 512 --tile_overlap 0 --checkpoint $MODEL_PTH
```

`testset` configuration can be found in `test.py`. Please place test images path correctly.

`testset: [test100, rain100L, rain100H, test1200, test2800, gtrain, gtavcrop, gtavset3, realint, outdoorrain, raindrop, snow100k-L]`

**Modify the input_path and target_path for testing datasets**

```
# Test100, Rain100L, Rain100H, Test1200, Test2800 (Synthetic)
def test_test100():
    input_path = "/home1/zhangsy/rh/data/derain/MultiRain/test/Test100/input"
    target_path = "/home1/zhangsy/rh/data/derain/MultiRain/test/Test100/target"
    imgs, gts = [], []
    for i in range(98):
        input_file = "{}.png".format(i+1)
        target_file = "{}.png".format(i+1)
        imgs.append(os.path.join(input_path, input_file))
        gts.append(os.path.join(target_path, target_file))
    print("Prcess Test100 | total files: [{}/{}]".format(len(imgs), len(gts)))
    return imgs, gts
```



## Citation

If Rainmer helps your research or work, please consider citing this paper.

```
@inproceedings{ran2024rainmer,
  title={Rainmer: Learning Multi-view Representations for Comprehensive Image Deraining and Beyond},
  author={Ran, Wu and Ma, Peirong and He, Zhiquan and Lu, Hong},
  booktitle={Proceedings of the ACM International Conference on Multimedia},
  pages={2766--2775},
  year={2024}
}
```



## Contact

If you have any questions, please contact [e-mail](wran21@m.fudan.edu.cn).
