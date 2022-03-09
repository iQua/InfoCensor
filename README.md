# InfoCensor
InfoCensor: An Information-Theoretic Framework against Sensitive Attribute Inference and Demographic Disparity

This codebase is written on top of pytorch (python3)

Some required packages are listed in requirement.txt

## Datasets
### German
The .csv file in german_data directory
### Health Heritage
The .csv file in health_data directory
### UTKFace
Download the UTKFace dataset from https://susanqq.github.io/UTKFace/ (Note that we use the utkface directory in the "Aligned and Cropped Faces")
Replace the data path in "load_utkface.py". Note that for UTKFace and Twitter, it is better to use cuda to accelerate training (--use-cuda)
### Twitter:
Get tokenizer from https://github.com/erikavaris/tokenizer
pip install git+https://github.com/erikavaris/tokenizer.git
You also need torchtext (find the version that matches your torch version)


## Baselines without defense (code files are in the "defenses" directory)

```commandline
python no_defense.py --dataset <dataset> --target-attr <the target attribute> --sensitive-attr <the sensitive attribute>
```

### Examples (If you do not have GPU, then do not use the argument --use-cuda):

```commandline
python no_defense.py --dataset german --target-attr credit --sensitive-attr gender
```

```commandline
python no_defense.py --dataset health --target-attr charlson --sensitive-attr age
```

```commandline
python no_defense.py --dataset utkface --num-epochs 100 --target-attr age --sensitive-attr race --use-cuda
```

### Examples for InfoCensor (It is better to add --use-cuda if you have a gpu)

```commandline
python infocensor.py --dataset health --target-attr charlson --sensitive-attr age --info-lambda 0.5 --info-beta 0.0 --fair-kappa 0.5
```


## Attacks (code files are in the "attacks" directory)
```commandline
python baseline.py --dataset <health or utkface or twitter> --defense-method <adv_censor or info_censor> --defense-model-name <the model name (record certain hyperparameters) for the specific defense> --sensitive-attr <the sensitive attribute>
```

```commandline
python decensor.py --defense-method <adv_censor or info_censor> --dataset <health or utkface>  --defense-model-name <the model name (record certain hyperparameters) for the specific defense> --sensitive-attr <the sensitive attribute>
```

### Examples (It is better to add --use-cuda if you have a gpu)
```commandline
python baseline.py --dataset health --defense-method no_defense --defense-model-name target_charlson_sensitive_age_num_features_128 --sensitive-attr age
```

```commandline
python baseline.py --dataset health --defense-method infocensor --defense-model-name target_charlson_sensitive_age_num_features_128_lambda_0.5_beta_0.0_kappa_0.5 --sensitive-attr age
```