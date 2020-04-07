"""This Module is to apply wavenet to do forecasting on dwd weather database."""
import logging
import sqlalchemy
import pandas as pd
from inspect import cleandoc
import types
import json


# TODO: Visualization module for trained neural network
# TODO: Prediction with your model - especially focus on categorical encoder


class UtilBase(object):
    """Util Base."""

    def __init__(self, *args, **kwargs):
        """Init Util Base"""
        self.logger = self.get_logger()

        super().__init__()

    @staticmethod
    def get_logger(name=__name__, level=logging.WARNING):
        """Get Logger by name of module (file)

        :param name: Name of logger, defaults to variable __name__
        :type name: string, optional
        :param level: level of logging, defaults to logging.WARNING
        :type level: string, imported directly from logging class, optional
        :return: customized logger with handlers
        :rtype: logging.Logger
        """
        logger = logging.getLogger(name)

        # Create Consoler handler
        c_handler = logging.StreamHandler()
        c_handler.setLevel(level)

        _formater = logging.Formatter(
            '%(name)s :: %(levelname)s :: %(message)s')
        c_handler.setFormatter(_formater)
        logger.addHandler(c_handler)
        return logger


class Source(UtilBase):
    """Source of Data."""

    def __init__(self, source_type=None, source=None):
        self.source_type = source_type
        self.source = source
        self.logger = self.get_logger()

    def _configure():
        pass


class CSVSource(Source):
    pass


class SQLSource(Source):
    def __init__(self):
        self.driver = "postgresql"
        self.username = "tiger"
        self.password = "tiger"
        self.host = "localhost"
        self.port = 5432
        self.database = "climatedb"
        self.conn_str = (f"{self.driver}://"
                         f"{self.username}:{self.password}"
                         f"@{self.host}:{self.port}/"
                         f"{self.database}")

        super().__init__(source_type="sql", source=self.conn_str)

        self.engine = sqlalchemy.create_engine(self.source)
        self.conn = None

    def open_conn(self):
        self.logger.debug(self.source)
        if self.conn is None:
            self.conn = self.engine.connect()

    def close_conn(self):
        if self.conn is not None:
            if not self.conn.closed:
                self.conn.close()

    def query(self, sqlstr, **kwargs):
        self.logger.debug(sqlstr)
        with self.engine.connect() as conn:
            df = pd.read_sql(sqlstr, conn, **kwargs)
            return df


class Climate(UtilBase):
    """Climate class for dwd weather data.

    [Time Series]:
        Each of time series even multivariate or single variate is with a unique
        id. For univariate, one row in table data is one series, while there are
        multi-rows in case of multivariate.

        In case of categorical data, each combination of dimensions (categorical
        variables) will generate a unique time series over multi-metric columns.

    """

    def __init__(self, *args, **kwargs):
        """Create instance of climate."""
        self.datasource = SQLSource()

        super().__init__(**kwargs)

    def get_ids(self, col, table, conditions=None):
        """Get unique time series ids.

        In the data source, the column id must be available, it could be index
        or a combination of dimension (Dimensions are categorical variables and
        metric is numerical variables).
        """

        if isinstance(conditions, list):
            conditions = " AND ".join(conditions)

        sqlstr = cleandoc(f"""SELECT
                                DISTINCT {col}
                             FROM
                                {table}
                             WHERE
                                1 = 1
                                {"" if not conditions else " AND " + conditions}
                            """
                          )

        self.logger.debug(sqlstr)
        stations_id = self.datasource.query(sqlstr)
        return stations_id[col].tolist()

    def yield_ts(self,
                 table="train_climate",
                 sel_col=None,
                 conditions=None,
                 **kwargs
                 ):
        """Yield a time series based on condition."""
        if isinstance(conditions, list):
            cond_str = " AND ".join(conditions)
            conditions = cond_str

        if isinstance(sel_col, str):
            sel_col = [sel_col]
        if isinstance(sel_col, list):
            sel_col = ", ".join(sel_col)
        if not sel_col:
            sel_col = "*"
        sqlstr = cleandoc(f"""SELECT
                                {sel_col}
                            FROM
                                {table}
                            WHERE
                                1 = 1
                                {"" if not conditions else " AND " + conditions}
                            """
                          )
        self.logger.debug(sqlstr)
        df = self.datasource.query(sqlstr, **kwargs)

        self.logger.debug(type(df))
        if isinstance(df, types.GeneratorType):
            yield from df
        if isinstance(df, pd.DataFrame):
            yield df

    def yield_n_ts(self,
                   table,
                   id_col,
                   sel_col=None,
                   conditions=None,
                   **kwargs
                   ):
        """Yield all time series based on its id.
        """

        if isinstance(conditions, str):
            conditions = [conditions]

        ids = self.get_ids(col=id_col,
                           table=table,
                           conditions=conditions
                           )

        self.logger.debug(f"""
                            Table:  {table}
                            TS ID:  {id_col}
                            Selected Cols: {sel_col}
                            No Ids: {len(ids)}
                        """)

        for _id in ids:
            _match = [f"{id_col} = {_id}"] + conditions
            yield from self.yield_ts(table=table,
                                     sel_col=sel_col,
                                     conditions=_match,
                                     **kwargs
                                     )

    def yield_sql_str(self,
                      table,
                      id_col=None,
                      sel_col=None,
                      conditions=None,
                      **kwargs
                      ):
        """Yield all time series based on its id.
        """

        if isinstance(conditions, str):
            conditions = [conditions]

        if id_col is not None:
            ids = self.get_ids(col=id_col,
                               table=table,
                               conditions=conditions
                               )

            self.logger.debug(f"""
                                Table:  {table}
                                TS ID:  {id_col}
                                Selected Cols: {sel_col}
                                No Ids: {len(ids)}
                            """)
        else:
            # if no id_col then condition will be where 1=1
            #   and other conditions
            id_col = 1
            ids = [1]

        for _id in ids:
            _match_id_str = [f"{id_col} = {_id}"]
            if isinstance(conditions, list):
                conditions.extend(_match_id_str)
            else:
                conditions = _match_id_str
            conditions = " AND ".join(conditions)

            if isinstance(sel_col, str):
                sel_col = [sel_col]
            if isinstance(sel_col, list):
                sel_col = ", ".join(sel_col)
            if not sel_col:
                sel_col = "*"
            sql_str = cleandoc(f"""SELECT
                                    {sel_col}
                                FROM
                                    {table}
                                WHERE
                                    1 = 1
                                    {"" if not conditions else " AND " + conditions}
                                """
                               )
            yield sql_str

    @staticmethod
    def _csv_to_chunks(self,
                       x_train_f,
                       y_train_f,
                       n_input_steps,
                       n_output_steps,
                       time_col
                       ):
        """Transform and load generated train data to  keras fit_generator."""
        x_train = pd.read_csv(x_train_f, chunksize=n_input_steps)
        y_train = pd.read_csv(y_train_f, chunksize=n_output_steps)
        yield from self._to_batch_chunks(x_train,
                                         y_train,
                                         time_col=time_col,
                                         n_input_steps=n_input_steps,
                                         n_output_steps=n_output_steps
                                         )

    @staticmethod
    def _sql_to_chunks(self,
                       conn,
                       x_sql,
                       y_sql,
                       n_input_steps,
                       n_output_steps,
                       time_col
                       ):
        x_train = pd.read_sql(x_sql, conn, chunksize=n_input_steps)
        y_train = pd.read_sql(y_sql, conn, chunksize=n_output_steps)
        yield from self._to_batch_chunks(x_train,
                                         y_train,
                                         time_col=time_col,
                                         n_input_steps=n_input_steps,
                                         n_output_steps=n_output_steps
                                         )

    @staticmethod
    def _to_batch_chunks(x_df, y_df, **kwargs):
        time_col = kwargs.get("time_col")
        n_input_steps = kwargs.get("n_input_steps")
        n_output_steps = kwargs.get("n_output_steps")

        for chunk in x_df:
            if time_col in chunk.columns:
                if 'sample_idx' in chunk.columns:
                    batch_x = chunk.set_index(["sample_idx", time_col])
                    n_in_cols = len(batch_x.columns)
                    batch_x = batch_x \
                        .sort_index() \
                        .reset_index(drop=True) \
                        .to_numpy()
                    batch_x = batch_x.reshape((1, n_input_steps, n_in_cols))

                    batch_y = y_df.get_chunk().set_index(
                        ["sample_idx", time_col])
                    # n_out_cols = len(batch_y.columns)
                    batch_y = batch_y \
                        .sort_index() \
                        .reset_index(drop=True) \
                        .to_numpy()
                    batch_y = batch_y.reshape(1, n_output_steps)
                    yield (batch_x, batch_y)


def save_schema(df, f_path=None):
    schema = {}
    for k, v in df.dtypes.items():
        schema.update({k: str(v)})
    if isinstance(f_path, str):
        with open(f_path, "w") as f:
            f.write(json.dumps(schema))


def load_schema(f_path):
    import numpy as np
    with open(f_path, "r") as f:
        schema = json.load(f)
        for k, v in schema.items():
            schema[k] = np.dtype(v)
        return schema
