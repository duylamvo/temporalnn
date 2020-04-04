# Temporal Neural Network And Explainable AI

This is the project to use neural network to build a model for temporal data which has at least one time based dimension. This project includes 2 mains topics, temporalnn and xai. The former is one of important main generalized module while the latter is non-generalized codes but also important.

- Temporal neural network (temporalnn): module which contains blocks different kind of CNN-based model. temporalnn has also a wheel file (whl folder) which can be easy to be installed using pip:

  - models:
    - default CNN models
    - contains WaveNet, one of the modern cnn-based deep learning model in this module.
  - utils:
    - preprocess data
    - generate time series data set (`utils.ts_utils`)

- XAI algorithms, the second module, which seperately stored in XAI folder. The module attaches different current algorithms in explainable AI which were used to explain models built by temporalnn
  - Local Interpretable and Model Agnostic Explaination (LIME)
  - Other algorithms (on going)
- wavenet: a folder storing codes where we are trying to apply the module to build a temperature forecaster based on wavenet model. This could be seen as an example of using module temporalnn to build wavenet models.
- presentation: where we store

## Installation

Create an virtualenv

```bash
make virtualenv
```

Temporalnn is a tool that is developed by during building wavenet to handling time series data using neural network. The tool helps to build blocks of wavenet model. It supports also some transformation tools. Currently it is specific for data set with time series based on data frame of pandas. Exammple applied for dwd weather data.

```bash
pip install whl/temporalnn-0.0.2-py3-none-any.whl
```

if the whl file is not there, you could generate by

```bash
make setup
pip isntall temporalnn-0.0.2-py3-none-any.whl
```

## Build a WaveNet model for Univariate

To build a wavenet model implemented in temporalnn

```python
from temproalnn.temporalnn.models import temporal as ts_model

# Build A wavenet model
wn = ts_model.WaveNet()
model = wn.build_model(input_shape=(32, 1),
                        n_in_steps=32,
                        n_out_steps=1,
                        gated_activations=['relu', 'sigmoid'])

```

This is a model based on `keras`, hence you could apply any `model.compile(...)` and `model.fit(...)` to compile and train model. Please refer to keras api to see how to do.

Alternatively you could do all in one for wavenet by using `ts_util.train_on_memory` or `ts_util.train_on_generator(.)`

```python
from temporalnn.temporalnn.utils import ts_util

# To be fast you could train by feed every thing on memory
ts_util.train_on_memory(X_train, y_train, X_test, y_test,
                                input_shape,
                                x_steps, y_steps,
                                weight_file, history_file, log_dir,
                                shuffle=False,
                                batch_size=64,
                                validation_freq=10,
                                max_queue_size=20,
                                epochs=100,
                                verbose=1,
                                )
# alternatively you could use ts_util.train_on_generator(.)
# which could save more memory but bit slower.
```

## Example of Using wavenet for climate data

This example codes could be found in wavenet folder

```python
from temporalnn.temporalnn.utils import ts_util
from temproalnn.temporalnn.models import temporal as ts_model
from temproalnn.temporalnn.utils.datasets import gen_train

import pandas as pd

climate_file= "data/climate/preprocessed_climate.csv"
data = pd.read_csv(climate_file)

# columns (input features and output features from data)
target_col = "tmk"
indep_cols = ["tmk"]

# column to group by ~ each group is one uts
group_col = "stations_id"

# Time Steps of input, and output
x_steps = 32
y_steps = 1
input_shape= (x_steps, 1)

# Generate TS train data
x_set, y_set = gen_train(data,
                        indep_cols,
                        target_col,
                        group_col=group_col,
                        x_steps=x_steps,
                        y_steps=y_steps,
                        stride=1)

# K-Folds
kf = KFold(5, shuffle=True)
folds = kf.split(y_set)

# To simple, test it on one fold
train_index, test_index = folds[0]
X_train, X_test = x_set[train_index], x_set[test_index]
y_train, y_test = y_set[train_index], y_set[test_index]

output_dir = f"test-outputs/climate/train_{x_steps}_{y_steps}"
log_dir = output_dir + "/logs"
weight_file = output_dir + f"/weight/uts_{x_steps}_{y_steps}.h5.{set_id}"
history_file = output_dir + f"/history/uts_{x_steps}_{y_steps}.hist.{set_id}"

ts_util.train_on_memory(X_train, y_train, X_test, y_test,
                        input_shape,
                        x_steps, y_steps,
                        weight_file, history_file, log_dir,
                        shuffle=False,
                        batch_size=64,
                        validation_freq=10,
                        max_queue_size=20,
                        epochs=100,
                        verbose=1,
                        )

```

The project had already trained results under `wavenet_results` which will be used to apply explanable AI later in the project

## Apply Exaplanable AI to explain WaveNet Model.

### Local interpretable model-agnostic explanations (LIME)

This project has implemented LIME algorithm from scratch and be able to explain both images and regression for TS models. The project was hosted seperately and here ported codes were copied to have all-in-one-place repo under folder `xai/LIME`

To test the script simply run, (but possibly need to be modified the data folder be able be loaded)
`python lime-image.py`

The result of LIME could be found in `xai/LIME/test-outputs` or simply in our `presentation` folder
