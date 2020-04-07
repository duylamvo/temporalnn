"""Core generators and functions used to transfrom and generate data."""

import pandas as pd
import numpy as np
import tqdm
import os
import types

from .util import UtilBase
import multiprocessing as mp
import sqlalchemy


# TODO: Encoder OHE | Word2Vector embedding
# TODO: Decoder


class DataGenerator(UtilBase):
    """General generator."""

    def __init__(self, **kwargs):
        """Initialize instance."""
        super().__init__(**kwargs)

    @staticmethod
    def split_data_set():
        """Split data into tran/dev/test set based on split rate."""
        pass

    def shuffle(self):
        pass


class TemporalDataGenerator(DataGenerator):
    """Time Distributed to generate data based on slicing of time steps."""

    def __init__(self,
                 n_input_steps=1,
                 stride=1,
                 n_output_steps=1,
                 interpolate=True,
                 encoding='ohe',
                 **kwargs):
        """Initialize method.
        :param n_input_steps:   Time steps to be the input
                                For example: time steps=2 : X(t) = a*X(t-1) + b*X(t-2)
                                input will be x[0], x[1], and output will be
                                x[2] if it available
        :param stride:          Windowing stride on the sequence. Ex if stride = 2 then x[2] ~ x[0], x[1]
                                -> and then stride for 2 for the next is x[4] ~ x[2], x[3]
        :param n_output_steps:    number of output steps will be sliced. It is also called multi-steps prediction.
                                Default is 1.
        :param interpolate:    if true then it apply
                                interpolate linear for both direction
        :param encoding:        Encoding method {None, 'ohe', 'ordinal', 'embedding'} for categorical. One method is
                                'ordinal' which consider categories as int numbers. Another is 'one-hot-encoding' vector
                                which is dummies variables of 0s and 1s. The latest encoding method is embedding which
                                convert a word /categorical to numerical vectors. Currently, we have not supported
                                word embedding yet.

        """
        self.n_output_steps = n_output_steps
        self.n_input_steps = n_input_steps
        self.stride = stride
        self.interpolate = interpolate
        self.encoding = encoding

        super().__init__(**kwargs)

    def gen_train_ts(self,
                     df,
                     time_col=None,
                     drop_time_col=True,
                     in_var=None,
                     out_var=None,
                     series_id=''):
        """Generate a single dependent time series.

        Dependent time series like X(t) = a*X(t-1) + b*X(t-2) vs Independent time series where the output column
        does not depend on previous values of input vectors. This could happen when you have already parsed your
        time series data. It could be considered as processed time series for training data. Or it the output
        columns not in the list of input columns.
        For example:
            dependent ts -> what is the temperature of tomorrow if given temperature and wind level of 2 days ago?
            independent ts -> Is tomorrow rain or not if given temperature and wind level of 2 days ago?

        It is only if all dimensions are metric except time index
        For example:
            given a ts [1, 2, 3, 4, 5, 6, 7, 8],
            if stride = 1 and output depends on 4 previous time steps
                1st sample -> x = [1, 2, 3, 4], y = [5]
                2nd sample -> x = [2, 3, 4, 5], y = [6]
                3rd sample -> x = [3, 4, 5, 6], y = [7]
                4th sample -> x = [4, 5, 6, 7], y = [8]
            if stride = 2 and output depends on 4 previous time steps
                1st sample -> x = [1, 2, 3, 4], y = [5]
                2nd sample -> x = [3, 4, 5, 6], y = [7]
                3rd sample -> x = [5, 6, 7, 8], y = [NaN] -> this invalid sample will be removed, because out of range

        If there is categorical in the data frame, it will be converted to pd.category type and encode
        to either ordinal or one-hot vector. To handle the multi categorical, all categorical variables
        are concatenated as different unique time series.

        :param df:              A data frame with having at least a column as time index
        :param time_idx_col:    Name of column with time index
        :param in_var:          Names of input variables
        :param out_var:         Names of output variables
        :param series_no:       the prefix or series id will be added to sample-idx like series-"1"+sample_idx = 11
                                this to help to make sure a sample-idx is unique. Default is 0
        :param drop_time_idx: Drop column time_idx_col if true


        :return:                Generator of list of dictionary of input and output under json format
                                Ex: [
                                        {
                                            'sample_idx':1,
                                            'input_records':[   {col1: val, col2: val}
                                                                {col1: val, col2: val}  ],
                                            'output':[  {col1: val, col2: val}  ]
                                            'meta': {'columns': ['sample_idx', 'col1', 'col2']}
                                        },
                                        ...
                                    ]
        """
        if isinstance(df, (pd.Series, types.GeneratorType)):
            df = pd.DataFrame(df)

        if not isinstance(df, pd.DataFrame):
            raise Exception("NotSupportedError"
                            "Only support a dataframe, Series or dataframe chunk"
                            )

        if isinstance(in_var, str):
            in_var = [in_var]
        if isinstance(out_var, str):
            out_var = [out_var]

        in_var = in_var or df.columns
        out_var = out_var or df.columns

        if time_col and (time_col in df.columns):
            df = df.sort_values(time_col)
            if not drop_time_col:
                if time_col not in out_var:
                    out_var = out_var + [time_col]
                if time_col not in in_var:
                    in_var = in_var + [time_col]

        if self.interpolate:
            df = df.interpolate(limit_direction='both')

        total_rows = df.shape[0]
        if total_rows < self.n_input_steps:
            self.logger.warning(
                f"Length of sequence {total_rows} smaller",
                f"thant time steps {self.n_input_steps}")
            return
        try:
            # reset index again to start with 0
            df = df.reset_index(drop=True)
            sample_idx = 0
            for in_start in range(0, total_rows, self.stride):
                # calculate slicing of input and output
                in_end = in_start + self.n_input_steps - 1
                out_start = in_end + 1
                out_end = out_start + self.n_output_steps - 1

                _s_idx = f"{series_id}-{sample_idx}"
                if out_end < total_rows:
                    # using loc will different wit iloc in indexing
                    input_records = pd.DataFrame(df.loc[in_start: in_end, in_var]) \
                        .to_dict(orient='records')
                    output_records = pd.DataFrame(df.loc[out_start: out_end, out_var]) \
                        .to_dict(orient='records')

                    yield {'sample_idx': _s_idx,
                           'input_records': input_records,
                           'output_records': output_records}

                    sample_idx += 1
        except KeyError as ke:
            self.logger.error(ke)

    def _parser_to_csv(self, gen_data, dir_path=None):
        """Parse train data to data frame.

        The function will return two data sets, train input, and train output each will have a sample index.
        Each sample is a univariate or multivariate time-based vector. The return will like this
              Month  Passengers  sample_idx                 Month  Passengers  sample_idx
            1949-01         112           0                 1949-03         132           0
            1949-02         118           0
            1949-02         118           1                 1949-04         129           1
            1949-03         132           1

        Here 'Passengers' column is a univariate ts. Each sample here will include 2 time steps, which are generated by
        function generate ts train data. We could feed the 2-steps input vector into the neural network (a conversion to
        array also be necessary in some case). The output table on the left will have the corresponding sample index with
        the input

        :param gen_data:    Generators generated by function generate ts, or list with same output format
        :param dir_path:    Dir path to save files. Default is current working dir
        :return:            Path to files. Name will be dir_path/ts_input.csv, and dir_path/ts_output.csv
        """
        dir_path = dir_path or ''
        if dir_path[-1] == '/':
            dir_path = dir_path[:-1]
        try:
            os.makedirs(dir_path)
        except FileExistsError:
            pass
        x_file = f'{dir_path}/x_{self.n_input_steps}_{self.n_output_steps}.csv'
        y_file = f'{dir_path}/y_{self.n_input_steps}_{self.n_output_steps}.csv'
        header_x = False if os.path.isfile(x_file) else True
        header_y = False if os.path.isfile(y_file) else True

        # TODO: should parallelism here to boost faster
        first_run = True
        for item in tqdm.tqdm(gen_data):
            i_df = pd.DataFrame(item["input_records"])
            i_df["sample_idx"] = item["sample_idx"]

            o_df = pd.DataFrame(item["output_records"])
            o_df["sample_idx"] = item["sample_idx"]

            i_df.to_csv(x_file, mode='a', index=False, header=header_x)
            o_df.to_csv(y_file, mode='a', index=False, header=header_y)
            if first_run:
                header_x = False
                header_y = False
                first_run = False
        return x_file, y_file

    def _parser_to_sql(self, gen_data, con):
        x_table_name = f"x_{self.n_input_steps}_{self.n_output_steps}"
        y_table_name = f"y_{self.n_input_steps}_{self.n_output_steps}"
        for item in tqdm.tqdm(gen_data):
            i_df = pd.DataFrame(item["input_records"])
            i_df["sample_idx"] = item["sample_idx"]

            o_df = pd.DataFrame(item["output_records"])
            o_df["sample_idx"] = item["sample_idx"]

            i_df.to_sql(x_table_name, con, index=False, if_exists="append", method='multi')
            o_df.to_sql(y_table_name, con, index=False, if_exists="append", method='multi')

    def _parser_to_dataframe(self, gen_data):
        """Parser generators to dataframe.

        Parsing json input generated into data frame pandas.

        :param gen_data:    Generators generated by function generate ts, or list with same output format
        :return:            Two data frames of x_train and y_train
        """
        input_df = pd.DataFrame()
        output_df = pd.DataFrame()
        for item in tqdm.tqdm(gen_data):
            i_df = pd.DataFrame(item["input_records"])
            i_df["sample_idx"] = item["sample_idx"]
            input_df = input_df.append(i_df)

            o_df = pd.DataFrame(item["output_records"])
            o_df["sample_idx"] = item["sample_idx"]
            output_df = output_df.append(o_df)

        return input_df.reset_index(drop=True), output_df.reset_index(drop=True)

    def _parser_to_numpy(self, gen_data):
        """Parser generators to dataframe."""
        inputs = []
        outputs = []
        for item in tqdm.tqdm(gen_data):
            inputs.append(pd.DataFrame(item["input_records"]).to_numpy())
            outputs.append(pd.DataFrame(item["output_records"]).to_numpy())

        return np.array(inputs), np.array(outputs)

    def parse_generators(self, gen_data, action='to_dataframe', **kwargs):
        """Parse generators to data frame, csv files or to numpy.

        :param gen_data:
        :param action: A string to describe which method will be used to parse generators.
                        {'to_csv', 'to_numpy', 'to_dataframe}. If to_csv, you could also specify dir_path
        :param dir_path: dir to file if action is 'to_csv'
        :return: file paths, dataframe or numpy type

        """
        _func = None
        if action == 'to_csv':
            _func = self._parser_to_csv
        if action == 'to_dataframe':
            _func = self._parser_to_dataframe
        if action == 'to_numpy':
            _func = self._parser_to_numpy
        if action == 'to_sql':
            _func = self._parser_to_sql

        if _func is not None:
            return _func(gen_data, **kwargs)

    @staticmethod
    def _parser_to_keras_gen(gen_data):
        """Translate generators to array and yield to generate generator for funciton fit_generator of keras."""
        for item in tqdm.tqdm(gen_data):
            yield (
                pd.DataFrame(item["input_records"]).to_numpy(),
                pd.DataFrame(item["output_records"]).to_numpy()
            )

    def yield_n_train_ts(self,
                         chunks,
                         x_vars,
                         y_vars,
                         time_col,
                         id_col=None,
                         schema=None):
        for df in chunks:
            if schema is not None:
                df = df.astype(schema)

            _ts_id = ""
            if id_col is not None:
                _id = df[id_col].unique()
                if len(_id) == 1:
                    _ts_id = str(_id[0])
                else:
                    raise ValueError(f"Not unique time series."
                                     f"It has {len(_id)} unique values: "
                                     f"{_id}"
                                     )
            yield from self.gen_train_ts(df,
                                         time_col=time_col,
                                         in_var=x_vars,
                                         out_var=y_vars,
                                         series_id=_ts_id
                                         )

    def to_sql(self,
               sql_str,
               conn_str,
               time_col,
               x_vars=None,
               y_vars=None,
               id_col=None,
               schema=None,
               drop_time_col=False,
               ):
        df = pd.read_sql(sql_str, conn_str)
        if df is None:
            raise ValueError("Cannot generate train. None data found")
        if time_col not in df.columns:
            raise KeyError(f"Cannot find {time_col} in the retrieved data frame")

        series_id = None
        if id_col in df.columns:
            _ids = df[id_col].unique()
            if len(_ids) != 1:
                raise KeyError(f"Series Id Column is not with unique value {id_col}")
            else:
                series_id = _ids[0]

        if schema is not None:
            _not_exists = [k for k in schema if k not in df.columns]
            for _k in _not_exists:
                schema.pop(_k)
            df = df.astype(schema)

        if not x_vars:
            x_vars = df.columns
        if not y_vars:
            y_vars = df.columns

        if isinstance(x_vars, str):
            x_vars = [x_vars]
        if isinstance(y_vars, str):
            y_vars = [y_vars]

        if not drop_time_col:
            if time_col not in y_vars:
                y_vars = y_vars + [time_col]
            if time_col not in x_vars:
                x_vars = x_vars + [time_col]

        df = df.sort_values(time_col)
        df = df.reset_index(drop=True)

        if self.interpolate:
            df = df.interpolate(limit_direction='both')

        total_rows = df.shape[0]

        slices = self._gen_slices(total_rows,
                                  series_id,
                                  x_vars,
                                  y_vars,
                                  self.n_input_steps,
                                  self.n_output_steps)
        with mp.Pool(mp.cpu_count()-1) as pool:
            result = []
            for sl in slices:
                sample_idx, start, end, t_vars, table_name = sl
                args = [df, sample_idx, start, end, t_vars, table_name, conn_str]
                result.append(pool.apply_async(self._slice_to_sql, args=args))

            for r in tqdm.tqdm(result):
                sample_idx, status = r.get()
                # print(f"{sample_idx}--> {status}")

    def _gen_slices(self, total_rows, series_id, x_vars, y_vars, n_input_steps, n_output_steps):
        if total_rows < self.n_input_steps:
            self.logger.warning(
                f"Length of sequence {total_rows} smaller",
                f"thant time steps {self.n_input_steps}")
            return

        x_table_name = f"x_{n_input_steps}_{n_output_steps}"
        y_table_name = f"y_{n_input_steps}_{n_output_steps}"

        # Get and check series id
        sample_idx = 0
        for in_start in range(0, total_rows, self.stride):
            # calculate slicing of input and output
            in_end = in_start + self.n_input_steps - 1
            out_start = in_end + 1
            out_end = out_start + self.n_output_steps - 1

            _s_idx = sample_idx
            if series_id is not None:
                _s_idx = f"{series_id}-{_s_idx}"

            if out_end < total_rows:
                # using loc will different wit iloc in indexing
                yield _s_idx, in_start, in_end, x_vars, x_table_name
                yield _s_idx, out_start, out_end, y_vars, y_table_name
                sample_idx += 1

    @staticmethod
    def _slice_to_sql(df, sample_idx, start, end, t_vars, table_name, conn_str):
        _df = pd.DataFrame(df.loc[start: end, t_vars])
        _df["sample_idx"] = sample_idx
        try:
            _df.to_sql(table_name,
                       conn_str,
                       index=False,
                       if_exists="append",
                       method='multi')
            return sample_idx, "done"
        except (Exception, KeyError):
            return sample_idx, "failed"
