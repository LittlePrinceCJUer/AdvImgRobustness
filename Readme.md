## Dataset Setup

The project uses MNIST and CIFAR-10 datasets. The datasets will be automatically downloaded when you run the training scripts.

## Training Models

### 1. Baseline Models

To train a baseline model:

```bash
python train_baseline.py
```

For baseline with early stopping:

```bash
python train_baseline_with_earlystop.py
```

### 2. MLP Models

To train an MLP model:

```bash
python train_mlp.py
```

### 3. ResNet Models

To train a ResNet model:

```bash
python train_res.py
```

### 4. Adversarial Training

To train models with adversarial training:

```bash
python train_adv.py        # For Cnn model
python train_adv_mlp.py    # For MLP model
python train_adv_res.py    # For ResNet model
```

## Testing Models

### 1. Testing on Clean Data

To test models on clean data:

```bash
python test_raw.py
```

### 2. Testing Against Adversarial Attacks

To test models against adversarial attacks:

```bash
python test_adv.py        # For baseline model
python test_adv_mlp.py    # For MLP model
python test_adv_res.py    # For ResNet model
```

## Generating Adversarial Examples

To generate adversarial examples:

```bash
python gen_adv.py        # For baseline model
python gen_adv_mlp.py    # For MLP model
python gen_adv_res.py    # For ResNet model
```
