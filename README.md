# CA-BiLSTM
Unraveling Decision-Making through Attention Mechanism in Deep Learning: Insights from Neuronal Spikes

### Abstract

The attention mechanism embedded within this model serves the crucial purpose of adeptly localizing neurons integral to decision-making stability within the task at hand. Specifically, when applied to the reproducible electrophysiology dataset from the International Brain Laboratory (IBL), our proposed model has demonstrated a remarkable capacity for accurately forecasting decision-making behavior in mice. Consequently, this investigation furnishes a novel perspective for unraveling the intricacies of neural decision-making mechanisms.

## Data

https://int-brain-lab.github.io/iblenv/notebooks_external/data_release_repro_ephys.html

see `new_prepare_all_data.py` for pre-processing steps.

## Code

create a conda environment, and install depencies (see `requirements.txt`). The models can be run with torch versions 2.1.0 and above

use `test.py` to start training and test a model.

`Caution`: This repository is being refactored. Please contact corresponding authors for specific questions about code in this repository.