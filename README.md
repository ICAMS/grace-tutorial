# Tutorials  

### Prerequisites  

This tutorial is for MLPfits workshop, hosted in Noctua2 cluster of PC2 (Paderborn Center for Parallel Computing)

### Tutorial Materials  

All materials for these tutorials are available on [GitHub](https://github.com/ICAMS/grace-tutorial). Clone the repository with:  

```bash
git clone --depth=1 https://github.com/ICAMS/grace-tutorial
```  

## Tutorial 1: Parameterization of 2-layer GRACE for Al-Li


### 1.1. Unpack and collect DFT data

Unpack data:

```bash
sh unpack_data.sh
```

Working folder for this tutorial is `1-AlLi-GRACE-2LAYER`

```bash
cd 1-AlLi-GRACE-2LAYER/0-data
```

Now, let's collect DFT data recursively, by running

```bash
grace_collect
```

This will result in `collected.pkl.gz` file, that we will be used by `gracemaker`

### 1.2. Parameterization

#### 1.2.1. Input file

Now, switch to another folder and generate input file with `gracemaker -t`:

```bash
cd ../1-fit
gracemaker -t
```

```bash
── Fit type
? Fit type: fit from scratch

── Dataset
  Tab ↹ autocompletes path  ·  ↑↓ navigates history
? Training dataset file (e.g. data.pkl.gz): ../0-data/collected.pkl.gz
  ✓ Train file: ../0-data/collected.pkl.gz
? Use a separate test dataset file? No
? Test set fraction (split from train) 0.05
  ✓ Test fraction: 0.05

── Model Details
? Model preset: GRACE_2LAYER_latest
  ✓ Preset: GRACE_2LAYER_latest
? Model complexity: small
  ✓ Complexity: small
? Cutoff radius (Å) 6
  ✓ Cutoff: 6.0 Å

── Optimizer
? Optimizer: Adam
  ✓ Optimizer: Adam

── Loss function
? Loss type: huber
  ✓ Loss type: huber
? Huber delta 0.01
? Energy loss weight 16
  ✓ Energy weight: 16
? Force loss weight 32
  ✓ Force weight: 32
? Include stress in the loss? Yes
? Stress loss weight 128.0
  ✓ Stress weight: 128.0
? Switch E/F/S weights mid-training? Yes
? Energy weight after switch (was 16) 128
? Force weight after switch (was 32) 32
? Stress weight after switch (was 128.0) 128.0

── Weighting & batch size
? Sample weighting scheme: uniform
  ✓ Weighting: uniform
? Batch size 8
  ✓ Batch size: 8  (test: 32)
? Target total updates 5000
  ✓ Total updates: 5000
```

This will produce `input.yaml` file, which you can further check and tune.
Please uncomment and set the line in order to reduce training time for this tutorial:



#### 1.2.2. Run gracemaker

Now, you can  submit fitting job to the queue with 
```bash
sbatch submit.sh
```
or run the model parameterization with:  

```bash
gracemaker
```  

During this process:  
* **Preprocessing and Data Preparation**: Tasks such as building neighbor lists will be performed.  
* **JIT Compilation**: The first epoch (or iteration) may take additional time due to JIT compilation for each training and testing bucket.  

If you do not wish to wait, you can manually terminate the process: `Ctrl+Z` and `kill %1`  

To create multiple models for an ensemble, run additional parameterizations with different seeds:  

```bash
sbatch submit-seed-2.sh
sbatch submit-seed-3.sh
```
or for local runs:
```bash
gracemaker --seed 2
gracemaker --seed 3
```  

#### 1.2.3. Analyze learning curves

Check `1-AlLi-GRACE-2LAYER/1-fit/visualize_metrics.ipynb` Jupyter notebook to analyze learning curves

#### 1.2.4. Save/export model

In order to export the model into TensorFlow's SavedModel format, do

```bash
gracemaker -r -s 
```

Check [here](../quickstart/#tensorflows-savedmodel) for more details.

### 1.3. Usage of the model

#### 1.3.1. Python/ASE

Please, check Jupyter notebook `1-AlLi-GRACE-2LAYER/1-fit/validate.ipynb` and `1-AlLi-GRACE-2LAYER/2-python-ase/Al-Li.ipynb`

#### 1.3.2. LAMMPS

You need LAMMPS with GRACE be compiled (check [here](../install/#lammps-with-grace)).

To use the GRACE potential in SavedModel format, apply the following pair_style:

```bash
pair_style grace pad_verbose
pair_coeff * * ../1-fit/seed/1/saved_model/ Al
```  
By default, this `pair_style` attempts to process the entire structure at once. However, for very large systems, this may result in Out-of-Memory (OOM) errors.

To prevent this, you can use the "chunked" versions of GRACE: `grace/1layer/chunk` or `grace/2layer/chunk`. These options process the structure in fixed-size pieces (chunks) that can be tuned to fit your GPU memory:

or 
```bash
pair_style grace/2layer/chunk chunksize 4096 
pair_coeff * * ../1-fit/seed/1/saved_model/ Al
```

The `pad_verbose` option provides detailed information about padding during the simulation.  
  

Submit the LAMMPS calculations to the queue:
```bash
cd ../3-lammps/
sbatch submit.sh
sbatch submit.big.sh
```

Or you can run it locally

```bash
cd ../3-lammps/
lmp -in in.lammps
lmp -in in.lammps.chunked
```

in order to compare the normal and chunked versions of the GRACE-2L models.

### Simulation Details  

- The simulation will first run for **20 steps** to JIT-compile the model.  
- Then, it will run for another **20 steps** to measure execution time.  

For example, on an A100 GPU, one of the final output lines might be:  

```
Loop time of 24.4206 on 1 procs for 20 steps with 108000 atoms
```  

This indicates that the current model (GRACE-2LAYER, small) achieves a performance of approximately **11 mcs/atom**, supporting simulations with up to **108k atoms**.  

---

## Tutorial 2. Parameterization of GRACE/FS for high entropy alloy HEA25S dataset

Dataset for this tutorial was taken from
["Surface segregation in high-entropy alloys from alchemical machine learning: dataset HEA25S"](https://iopscience.iop.org/article/10.1088/2515-7639/ad2983) paper.

All tutorial materials can be found in `grace-tutorial/2-HEA25S-GRACE-FS/`:

```bash
cd grace-tutorial/2-HEA25S-GRACE-FS/
```

### (optional) Dataset conversion from extxyz format

You can download complete dataset from [Materials Cloud](https://archive.materialscloud.org/record/2024.43) in an _extxyz_ format.

* Download `data.zip` file (in browser)
* Unpack the `data.zip`, go to any(all) subfolders and convert the _extxyz_ dataset with `extxyz2df`:

```bash
unzip data.zip
cd data/dataset_O_bulk_random
extxyz2df bulk_random_train.xyz
```

As a result, you will get compressed pandas DataFrame (`bulk_random_train.pkl.gz`).
Same procedure can be repeated for other files.

### 2.1. Parameterization

#### 2.1.1. Input file

Go to `1-fit` folder and run `gracemaker -t` for interactive dialogue:

```bash
cd 1-fit 
gracemaker -t
```

You have to enter following information:
```bash

── Fit type
? Fit type: fit from scratch

── Dataset
  Tab ↹ autocompletes path  ·  ↑↓ navigates history
? Training dataset file (e.g. data.pkl.gz): ../0-data/bulk_random_train.pkl.gz
  ✓ Train file: ../0-data/bulk_random_train.pkl.gz
? Use a separate test dataset file? No
? Test set fraction (split from train) 0.05
  ✓ Test fraction: 0.05

── Model Details
? Model preset: FS
  ✓ Preset: FS
? Model complexity: medium
  ✓ Complexity: medium
? Cutoff radius (Å) 7
  ✓ Cutoff: 7.0 Å

── Optimizer
  → FS from scratch: BFGS (full Hessian) is recommended for small/medium models.
  → If your FS model has many parameters (large lmax/order), prefer L-BFGS-B instead.
? Optimizer: Adam
  ✓ Optimizer: Adam

── Loss function
? Loss type: huber
  ✓ Loss type: huber
? Huber delta 0.01
? Energy loss weight 16
  ✓ Energy weight: 16
? Force loss weight 32
  ✓ Force weight: 32
? Include stress in the loss? Yes
? Stress loss weight 128.0
  ✓ Stress weight: 128.0
? Switch E/F/S weights mid-training? Yes
? Energy weight after switch (was 16) 128
? Force weight after switch (was 32) 32
? Stress weight after switch (was 128.0) 128.0

── Weighting & batch size
? Sample weighting scheme: uniform
  ✓ Weighting: uniform
? Batch size 16
  ✓ Batch size: 16  (test: 64)
? Target total updates 50000
  ✓ Total updates: 50000
```

#### 2.1.2. Run `gracemaker`

Now, you can run the model parameterization with:  
Submit job to the queue
```bash
sbatch submit.sh
```

or run locally

```bash
gracemaker input.yaml
```  

During the run, you may notice multiple warning messages starting with `ptxas warning:` or similar. These messages indicate that JIT compilation is occurring for each training and testing bucket, and they are normal. They will disappear after the first iteration/epoch.  

If you prefer, you can [reduce](../faq/#how-to-reduce-tensorflow-verbosity-level) the verbosity level of TensorFlow to minimize these messages.  

#### 2.1.3. (optional) Manual continuation of the fit with new loss function (TODO: update)

In order to continue the fit with **new** parameters, for example, add more weight onto energy in the loss function, do following steps:

* create new folder and run `gracemaker -t`:

```
── Fit type
? Fit type: continue fit

── Dataset
  Tab ↹ autocompletes path  ·  ↑↓ navigates history
? Training dataset file (e.g. data.pkl.gz): ../1-fit/seed/1/training_set.pkl.gz
  ✓ Train file: ../1-fit/seed/1/training_set.pkl.gz
? Use a separate test dataset file? Yes
? Test dataset file: ../1-fit/seed/1/test_set.pkl.gz
  ✓ Test file: ../1-fit/seed/1/test_set.pkl.gz
  ✓ Test fraction: 0.05

── Model Details
? Model config to continue from (e.g. model.yaml): ../1-fit/seed/1/model.yaml
  ✓ Found 3 checkpoints in ../1-fit/seed/1/checkpoints
? Which checkpoint to load? Auto (best test or latest)
  → Auto-selected: checkpoint.best_test_loss
  ✓ Previous model: ../1-fit/seed/1/model.yaml
  ✓ Checkpoint: ../1-fit/seed/1/checkpoints/checkpoint.best_test_loss
  → Note: reset_epoch_and_step is set to True (training will start from epoch 0)

── Optimizer
? Optimizer: Adam
  ✓ Optimizer: Adam

── Loss function
? Loss type: huber
  ✓ Loss type: huber
? Huber delta 0.01
? Energy loss weight 16
  ✓ Energy weight: 16
? Force loss weight 32
  ✓ Force weight: 32
? Include stress in the loss? Yes
? Stress loss weight 128.0
  ✓ Stress weight: 128.0
? Switch E/F/S weights mid-training? No

── Weighting & batch size
? Sample weighting scheme: uniform
  ✓ Weighting: uniform
? Batch size 16
  ✓ Batch size: 16  (test: 64)
? Target total updates 10000
  ✓ Total updates: 10000
```

submit to the queue or run as usual `gracemaker`
```  

**NOTE**: You can switch energy/forces/stress weights in the loss function during a single `gracemaker` run. To do this, you need to manually provide the `input.yaml::fit::loss::switch` option (see [here](../inputfile/#input-file-inputyaml) for more details) or provide a non-empty answer to the `Switch loss function E:F:S...` question in the `gracemaker -t` dialog. 

### 2.2. Save/Export Model

To export the model into both TensorFlow's SavedModel and GRACE/FS YAML formats, run:  

```bash
gracemaker -r -s -sf
```  

For more details, check [here](../quickstart/#gracefs).

### 2.3. Active Set Construction

To construct the active set (ASI) for uncertainty indication, run the following commands (more details [here](../quickstart/#build-active-set-for-gracefs-only)):  

```bash
cd seed/1
pace_activeset -d training_set.pkl.gz saved_model.yaml
```

### 2.4. Usage of the Model

#### 2.4.1. Python/ASE  

Please refer to the Jupyter notebook `2-HEA25S-GRACE-FS/HEA25-GRACE-FS.ipynb` for usage details.

#### 2.4.2. LAMMPS  

You need to compile LAMMPS with GRACE/FS (see [here](../install/#lammps-with-grace) for instructions).  

```bash
cd 3-lammps/grace-fs-with-extrapolation-grade/
mpirun -np 2 /path/to/lmp -in in.lammps
```  

In this simulation, the FCC(111) surface slab will be run under NPT conditions with an increasing temperature from 500K to 5000K. The extrapolation grade will be computed for each atom, and the configuration will be saved to `extrapolative_structures.dump` if the max gamma > 1.5.

To select the most representative structures for DFT calculations based on D-optimality, use the `pace_select` utility:  

```bash
pace_select extrapolative_structures.dump  -p ../../1-fit/seed/1/FS_model.yaml -a ../../1-fit/seed/1/FS_model.asi -e "Au"
```  

Find more details [here](https://pacemaker.readthedocs.io/en/latest/pacemaker/utilities/#d-optimality_structure_selection).

---

## Tutorial 3: Using Universal Machine Learning Interatomic Potentials

Universal machine learning interatomic potentials, also known as foundation models, are models capable of supporting a wide range of elements or even nearly the entire periodic table. These models are parameterized using large reference DFT datasets, such as the [Materials Project](https://next-gen.materialsproject.org/) or [Alexandria](https://alexandria.icams.rub.de/). Some of these models have been tested for high-throughput materials discovery, as demonstrated in [Matbench Discovery](https://matbench-discovery.materialsproject.org/).

We have parameterized several GRACE-1LAYER and GRACE-2LAYER models on the MPTraj dataset (relaxation trajectories from the Materials Project).

### 3.1. Overview and Download

You can view the available GRACE foundation models using the command:

```bash
grace_models list
```

To download a specific model, use:

```bash
grace_models download MP_GRACE_1L_r6_4Nov2024
```

To download all models at once, use:

```bash
grace_models download all
```

### 3.2. Usage in ASE

To load a model in ASE, use the following function:

```python
from tensorpotential.calculator.foundation_models import grace_fm, GRACEModels

calc = grace_fm("MP_GRACE_1L_r6_4Nov2024",
                pad_atoms_number=2,
                pad_neighbors_fraction=0.05, 
                min_dist=0.5)
# or 
calc = grace_fm(GRACEModels.MP_GRACE_1L_r6_4Nov2024) # for better code completion
```

Note that the additional parameters are optional. Default values are provided for `pad_atoms_number` and `pad_neighbors_fraction`.

For more details, refer to the Jupyter notebook `3-foundation-models/1-python-ase/using-grace-fm.ipynb`.

### 3.3. Usage in LAMMPS

The usage of foundation models in LAMMPS is the same as for custom-parameterized GRACE potentials. Examples are provided in the following directories:

* `3-foundation-models/2-lammps/1-Pt-surface`: Simulation of an oxygen molecule on a Pt (100) surface.
* `3-foundation-models/2-lammps/2-ethanol-water`: Simulation of ethanol and water.

**Note:** The currently available GRACE-1LAYER models do not support MPI parallelization. Updated models with MPI support will be released in the future.

---

## Tutorial 4: Fine-Tuning Foundation GRACE Models

Fine-tuning foundation GRACE models can only be performed using checkpoints and not saved models.

```yaml
...

potential:
  finetune_foundation_model: GRACE-1L-OAM
#  reduce_elements: True # select from original models only elements presented in the dataset  
fit:

  ### set small learning rate
  opt_params: {learning_rate: 0.001,  ... }
  
  ### evaluate initial metrics
  eval_init_stats: True
  
  # reset_optimizer: True
  
  ### specify trainable variables name pattern (depends on the model config)
  # trainable_variable_names: ["rho/reducing_", "Z/ChemicalEmbedding"] 
```

If you want manually specify model and checkpoint, then

```yaml
potential:
  filename: /path/to/model.yaml
  checkpoint_name: /path/to/checkpoints/checkpoint.best_test_loss
#  reduce_elements: True 
```


### Finetuning and distillation

#### Finetuning  foundation model with a new dataset

Run `gracemaker` with `-t` flag to start interactive dialogue and select the following options:

```bash
╭──────────────────────────────╮
│ GRACEmaker input.yaml wizard │
╰─ navigate with arrow keys · ─╯

── Dataset
? Training dataset file (e.g. data.pkl.gz): ../../1-AlLi-GRACE-2LAYER/0-data/collected.pkl.gz
  ✓ Train file: ../../1-AlLi-GRACE-2LAYER/0-data/collected.pkl.gz
? Use a separate test dataset file? No
? Test set fraction (split from train) 0.05
  ✓ Test fraction: 0.05

── Model
? Finetune a foundation model? Yes
? Model tier: 2L  — two message-passing layers (most accurate)
? Training dataset: OMAT  — OMat24 only (base / ft-E variants)
? Model size: medium    — larger capacity
? Fine-tuning variant: ft-E   — fine-tuned on energies
  ✓ Foundation model: GRACE-2L-OMAT-medium-ft-E

── Loss function
? Loss type: huber
  ✓ Loss type: huber
? Huber delta 0.01
? Energy loss weight 16
  ✓ Energy weight: 16
? Force loss weight 32
  ✓ Force weight: 32
? Include stress in the loss? Yes
? Stress loss weight 128.0
  ✓ Stress weight: 128.0
? Switch E/F/S weights mid-training? Yes
? Switch after (fraction 0-1, epoch number, or 'auto' = 0.75) 0.75
? LR reduction factor at switch (new LR = current_LR × factor) 0.1
? Energy weight after switch (was 16) 128
? Force weight after switch (was 32) 32
? Stress weight after switch (was 128.0) 128.0

── Weighting & batch size
? Sample weighting scheme: uniform
  ✓ Weighting: uniform
? Batch size 8
  ✓ Batch size: 8  (test: 32)


```

After that, run `gracemaker` again in order to start the training or submit the job to the cluster.

After the training is finished, you can find the final model in `seed/1/final_model` folder.
If not, then you can export the model with `gracemarker -r -s` command to `seed/1/saved_model` folder.

#### Generating distilled data

Now, you can use finetuned model to generate distilled reference data.
Check `3-foundation-models/3b-distillation/distill_data.ipynb`.
This fill create `distilled_AlLi_dataset.pkl.gz` file.

#### Training distilled model

Run `gracemaker` with `-t` flag to start interactive dialogue and select the following options:

```bash
── Dataset
? Training dataset file (e.g. data.pkl.gz): distilled_AlLi_dataset.pkl.gz
  ✓ Train file: distilled_AlLi_dataset.pkl.gz
? Use a separate test dataset file? No
? Test set fraction (split from train) 0.05
  ✓ Test fraction: 0.05

── Model
? Finetune a foundation model? No
? Model preset: FS
  ✓ Preset: FS
? Model complexity: medium
  ✓ Complexity: medium
? Cutoff radius (Å) 7
  ✓ Cutoff: 7.0 Å

── Loss function
? Loss type: huber
  ✓ Loss type: huber
? Huber delta 0.01
? Energy loss weight 16
  ✓ Energy weight: 16
? Force loss weight 32
  ✓ Force weight: 32
? Include stress in the loss? Yes
? Stress loss weight 128.0
  ✓ Stress weight: 128.0
? Switch E/F/S weights mid-training? Yes
? Switch after (fraction 0-1, epoch number, or 'auto' = 0.75) 0.75
? LR reduction factor at switch (new LR = current_LR × factor) 0.1
? Energy weight after switch (was 16) 128
? Force weight after switch (was 32) 32
? Stress weight after switch (was 128.0) 128.0

── Weighting & batch size
? Sample weighting scheme: uniform
  ✓ Weighting: uniform
? Batch size 32
  ✓ Batch size: 32  (test: 128)

```
then set manually in `input.yaml` file in order to reduce training time:
```yaml
fit:
  target_total_updates: 10000
```

and run the fit with `gracemaker input.yaml` or submit it to the cluster.

After fit is finished, you can find the final model in `seed/1/final_model` folder.
We also need to convert model to grace/fs format. Go to `seed/1/` and run
`grace_utils -p model.yaml -c checkpoints/checkpoint.best_test_loss.index export -sf`.
You will get `saved_model.yaml` file.
Then run `pace_activeset saved_model.yaml -d ../../distilled_AlLi_dataset.pkl.gz` to generate active set.

---

## Further Reading

* [Fitting Generic Tensor Properties](gen_tensor.md) — how to fit first- and second-rank tensor properties such as EFG, Born effective charges, or stress using the `TENSOR_1L`/`TENSOR_2L` presets.
