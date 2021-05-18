# AntiSplodge ![MIT License](https://img.shields.io/badge/license-MIT%20License-blue.svg) ![CC BY 4.0](https://img.shields.io/badge/license-CC%20BY%204.0-blue.svg)

More information soon.

## Installation

### From GitHuB

You can install the package directly from GitHub by running the following command:

`python -m pip install git+https://github.com/HealthML/AntiSplodge.git`

### Directly from source (this repository)

Clone the repository to a folder of your choice.

From a terminal this can be done by running:

`git clone git@github.com:HealthML/AntiSplodge.git`

Subsequently, run the following pip command from your terminal (in the root of cloned directory):

`pip install .`

## Usage

The full pipeline (see blow) assumes that you have a scRNA dataset (SC) and spatial transcriptomics dataset (ST) that both are formatted as .h5ad (AnnData data structures). Please see https://anndata.readthedocs.io/ for information about how to structure your data. Alternative you can check out the tutorial [INSERT TUTORIAL LINK] for an example on how to do this.

### Standard pipeline

```python
import AntiSplodge as AS
# SC should be the single-cell dataset formatted as .h5ad (AnnData)
Exp = AS.DeconvolutionExperiment(SC) 
Exp.setVerbosity(True)

# CELLTYPE_COLUMN should be replaced with actual column
Exp.setCellTypeColumn('CELLTYPE_COLUMN') 
# Use 80% as train data and split the rest into a 50/50 split validation and testing
Exp.splitTrainTestValidation(train=0.8, rest=0.5)

# Generate profiles, num_profiles = [#training, #validation, #testing]
# This will construct 10,000*10(CDs)=100,000, 5,000*10=50,000, 1,000*10=10,000 profiles  
# for train, validation and test (respectively)
Exp.generateTrainTestValidation(num_profiles=[10000,5000,1000], CD=[1,10])

# Load the profiles into data loaders
Exp.setupDataLoaders()

# Initialize Neural network-model and allocate it to the cuda_id specified
# Use 'cuda_id="cpu"' if you want to allocate it to a cpu
Exp.setupModel(cuda_id=6)

# Train the model using the profiles generated 
# The patience parameter determines how long it will run without fining a new better (lower) error 
# The weights found will be saved to 'NNDeconvolver.pt' and will be autoloaded once the training is complete 
stats = AS.train(Exp, save_file="NNDeconvolver.pt", patience=100)

# Check the testing accuracy
y_preds = AS.predict(Exp)

# Print test divergance as (Jensen Shannon divergance)
import numpy as np
from scipy.spatial import distance
jsds_ = []
for i in range(len(y_preds)):
    jsds_.append(distance.jensenshannon(Exp.Y_test_prop[i], y_preds[i]))
print("Mean {}".format(np.mean(jsds_)))

# Plot a boxplot of divergences for each test profile 
import seaborn as sns
import pandas as pd
pd.DataFrame({'jsds': jsds_}).to_csv("MouseBrainTestJSDS.csv")
sns.boxplot(y="JSD", data=pd.DataFrame({'JSD':jsds_}))

# Assuming we have a spatial transcriptomics dataset ST formatted in .h5ad (AnnData)
# create dataloader so that we can predict the profiles of each spot in our ST dataset
dataset_spots = AS.SingleCellDataset(torch.from_numpy(np.array(ST.X.toarray())).float(), torch.from_numpy(np.array([0]*ST.n_obs)).float())
spots_loader = DataLoader(dataset=dataset_spots,
                      batch_size=50, # batch_size doesn't matter 
)

spot_preds = AS.predict(Exp, spots_loader) # predict spots
# The results for each ST profile (spot) is now in spot_preds and can be used for further analysis 
```


### Profile generation

### Several ways of training 
The standard training procedure is 

1. Standard training
```python
# Assuming an Exp is an DeconvolutionExperiment
AS.train(experiment=Exp, patience=25, save_file=None, auto_load_model_on_finish=True) # default parameters
```
2. Several warm restarts
Do 10 warm restarts with a low patience (n=5), this will autoload the model per train call.
This will make the best model weights be loaded back onto the model and it will try again from these settings
```python
# Do 10 warm restarts 
for (i in range(10) {
    AS.train(experiment=Exp, patience=5) # default parameters
}
```

3. Lowering learning rate


4. Running on systems with reduced memory

### Use profiles and train the network with low memory and warm restarts 

### Tutorial
Check out the tutorial [INSERT TUTORIAL LINK].
See tutorials.

## Dependencies

## Documentation

The documentation will be available at https://antisplode.readthedocs.io/.

## References

## License

The source code for AntiSplodge is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Known bugs 

- AntiSplodge is prone to be affected by bad initiations. Oftentimes, this can be resolved by simply restarting the Experiment (or re-initializing the model). This seems to be more frequent when solving problems with many classes (large number of cell types). If verose is set to true, you should see output warnings during training with (`!!NaNs vectors produced!!`, these are not a problem if they only persist for a single iteration and is gone in the next).
