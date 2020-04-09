# Temporal Neural Network And Explainable AI

## Introduction
Temporalnn is to build different neural network to work with temporal data, which has at least one time based dimension. This project includes 2 mains topics. Existing block of neural network and explainable ai (xai) used to explain and validate the model.

- Temporal Models supported
    - pure CNN for temporal
    - WaveNet
    - Simple LSTM 
- XAI model supported
    - LIME for time series regression

Among the model supported, WaveNet is the latest and most modern to support very long time steps of time series data. Additionally, temporalnn tries to combines building and explaining the structure of neural network into one tool. It is useful when you have both as a convenient tool

Besides the supporting models, temporalnn also support utilities in converting time series data frame into train data both univariate and multivariate time series. 
  
## Installation
Temporalnn has already simplified the installation by using makefile. If you do not use any other virtualenv tool like conda, you can easily create one by:

```bash
make virtualenv
make wheel
pip install whl/temporalnn-0.0.2-py3-none-any.whl
```

## Sample Data - DWD Climate data

Temporalnn has been built and tested on dws weather data. You could find a test data for small climate under tests/data folder. There are here also example models was built by our temporalnn Wavenet block. 

```python
import json
import pandas as pd

with open("tests/data/climate_small_schema.json") as f:
    schema = json.load(f)
climate_data = pd.read_csv("tests/data/climate_small.csv")
climate_data = climate_data.astype(schema)

df = climate_data.copy()
df.set_index("measure_date")

```
## Temporal with WaveNet
### Generate training data

You could use temporalnn to generate or convert a time-based index data frame into train and test set

```python

from temporalnn.utils import ts as ts_util
df = "... Sample Data ..."

df.set_index("measure_date")

dependent_columns = ["fx"]
independent_column = "tmk"
group_col = "stations_id"

x_steps = 7
y_steps = 2
stride = 1
x_train, x_test, y_train, y_test = ts_util.df_to_ts_numpy(
    df, dependent_columns,
    independent_column, group_col,
    x_steps, y_steps, stride,
    split_test=True, test_size=0.3, random_state=42)
```
### Build a WaveNet model for Univariate Time Series

You could build a wavenet block by simply

```python
from temporalnn.models.temporal import WaveNet


x_train, x_test, y_train, y_test = """... Previous block ..."""
x_steps = 32
y_steps = 1
x_dim = 1   # uts only 1 column
input_shape = (x_steps, x_dim)
# Build A wavenet model
wn = WaveNet()
model = wn.build_model(input_shape=input_shape,
                           x_steps=x_steps,
                           y_steps=y_steps,
                           gated_activations=['relu', 'sigmoid'],
                           n_conv_filters=32)

model.compile(optimizer='nadam',
              loss='mse',
              metrics=['mse', 'mae', 'mape', 'cosine'],
              )

validation_data = (x_test, y_test)
history = model.fit(x_train, y_train,
                    validation_data=validation_data,
                    )
```
Currently our focus is to use WaveNet as a main model to apply explainable to explains different models built by WaveNet hence we has built a function to simplify training. 

```python

from temporalnn.utils.ts_trainer import train

x_train, x_test, y_train, y_test = """... Previous block ..."""

# The train function has a WaveNet model built inside.
train(x_train, y_train, x_test, y_test)

# Alternative if you have alreday weight_file, or want to save weight file
weight_file = "...path_to_output..."
train(x_train, y_train, x_test, y_test, weight_file=weight_file)

```

## Complete Example of Using wavenet for climate data (Univariate Time Series)

The complete example of using our simple train-test set. 

```python
import json
import os
import pandas as pd
from temporalnn.utils.ts_trainer import train
from temporalnn.utils import ts as ts_util

with open("tests/data/climate_small_schema.json") as f:
    schema = json.load(f)
climate_data = pd.read_csv("tests/data/climate_small.csv")
climate_data = climate_data.astype(schema)

df = climate_data.copy()
df.set_index("measure_date")

dependent_columns = ["fx"]
independent_column = "tmk"
group_col = "stations_id"

x_steps = 7
y_steps = 2
stride = 1
x_train, x_test, y_train, y_test = ts_util.df_to_ts_numpy(
    df, dependent_columns, independent_column, group_col,
    x_steps, y_steps, stride,
    split_test=True, test_size=0.3, random_state=42)

weight_file = "./output_dir" + "/" + "uts_tmk_7_1.h5"
os.makedirs(os.path.dirname(weight_file), exist_ok=True)
train(x_train, y_train, x_test, y_test, weight_file=weight_file)

```

If you do not want to use only one train-test set but using KFold instead, you could use our df_to_ts_numpy we use ts_util to generate sample and data set. But you also could use KFold from `scikit-learn` to generate different KFold

```python
from temporalnn.utils import ts as ts_util
df, dependent_columns, independent_column, group_col, x_steps, y_steps, stride = "... previous ..."

x_train, y_train = ts_util.df_to_ts_numpy(
    df, dependent_columns, independent_column, group_col,
    x_steps, y_steps, stride, split_test=False)

# Apply K-Fold here for different training-test sets
```

More examples could be found under our tests folder like [temporal-util](tests/test_temporal_utils.py)

Notice that, if you want to run tests as interative console, then please correct the folder of data from "data/..." to "tests/data/..."

## Explainable AI with LIME

We have implemented LIME approach from scratch also for images and time series. Image classification using LIME in this project is just a demo of how LIME works. We actually use LIME to explain our time series model. Our examples could be easily found under [LIME-image](tests/test_lime_image.py) and [LIME-UTS](tests/test_lime_uts.py)

### Introduction of LIME
LIME stands for Local Interpretable and Model Agnostic Explanation. A general method to explains and validate trust of model. 

The graph as follows explaining model
![Alt text](https://g.gravizo.com/source/svg/custom_mark_01?https%3A%2F%2Fraw.githubusercontent.com%2Fdungthuapps%2Ftemporalnn%2Fmaster%2FREADME.md)

@custom_mark_01
digraph LIME {
    X [label="Instance X"]
    X_segment [label="X - Segmentation"]
    X_comma [label="X'"]
    Z_comma [label="Sampling Neighbors Z'"]
    Z [label="Z"]
    F [label="f(Z)"]
    P [label="π(x,z)"]
    G [label="g(z') = W*Z'", color="turquoise", style="filled"]
    L [label="Loss of f(z) ~ g(z') with weight π"]
    O [label="Optimizer: argmin { loss }"]
    R [label="W^ ~ Weights of features", color="turquoise", style="filled"]
    
    X -> X_segment
    X_segment -> X_comma
    X_comma -> Z_comma
    Z_comma -> Z
    Z -> F
    Z -> P [label="Similarity of x to z"]
    Z_comma -> G
    F -> L 
    G -> L 
    P -> L 
    L -> O 
    O -> R
    
    subgraph sampling_z {
        node [shape="box", style="dashed"]
        
        S [label="Sampling Neighbors Z'"]
        A [label="Z'"]
        B [label="Z'"]
        C [label="Z'"]
        
        S -> B
        S -> C
        S -> A
    }
}
@custom_mark_01


### Use LIME to explain univariate time series built by WaveNet

We have implemented and extended LIME to be able to use with time series. You could get our example models which is built by our WaveNet block under `tests/data/uts_tmk_32_1.h5` for 32 input steps and output 1 step of value.

#### Prepare data (tmk)
```python
import pandas as pd
import json

DATA_DIR = "tests/data"

def preload_uts(data_dir=DATA_DIR):
    df = pd.read_csv(f"{data_dir}/climate_small.csv")
    with open(f"{data_dir}/climate_small_schema.json") as f:
        schema = json.load(f)
        df = df.astype(schema)

    df = df.set_index("measure_date")
    tmk = df.query("stations_id == 2074")["tmk"]

    input_steps = 32
    return tmk[-input_steps:].to_numpy()

ts_original = preload_uts()
```
### Prepare predict function

Even LIME is agnostic to any model, but we need to write correct function to return what we really care.
Here I care about y_hat (or temperature of tommorow (tmk) in dwd weather data. The model is trained to do that). So we need to prepare our predict function to ensure returning a float number.

```python
from keras.models import load_model

DATA_DIR = "tests/data"
model = load_model(f"{DATA_DIR}/uts_tmk_32_1.h5")
 
def predict_uts_tmk(x, *args):
    _shape = x.shape
    if len(_shape) == 2:
        _shape = (1, _shape[0], _shape[1])
    elif len(_shape) == 1:
        _shape = (1, _shape[0], 1)
    x = x.reshape(_shape)

    y_hat = model.predict(x, *args)

    # flatten to one value
    return y_hat.ravel().item()

```

### Create an explainer for the model
To explain Time Series, you could easily use our class LIMETimeSeries to input an instance to explain. Combining with predict function which we need to write correctly to return values of temperature. 

```python
from temporalnn.explain.lime import LIMETimeSeries
preload_uts = """... previous ... """
predict_uts_tmk = """... previous ... """

ts_original = preload_uts()

xai_uts = LIMETimeSeries(x=ts_original, predict_fn=predict_uts_tmk)
xai_model = xai_uts.explain(on_offs=[1, 0])

```

### Visualization result
We provide also tool to visualize result easily
```python
from temporalnn.explain.lime import LIMETimeSeries
preload_uts = """... previous ... """
predict_uts_tmk = """... previous ... """
ts_original = preload_uts()

xai_uts = LIMETimeSeries(x=ts_original, predict_fn=predict_uts_tmk)
xai_model = xai_uts.explain(on_offs=[1, 0])

demo_dict = xai_uts.create_demo_explanation(xai_model, top_k=3)
xai_uts.viz_coefficient(xai_model)
xai_uts.viz_explain(demo_dict)
```

More examples could be found under our [LIME-test-uts](tests/test_lime_uts.py)