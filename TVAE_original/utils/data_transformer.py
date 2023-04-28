"""

Reference:
[1]: https://github.com/sdv-dev/CTGAN/blob/acb8e17cbf7ac8d29b834561726ab44c860453e8/ctgan/data_transformer.py#L18

data_transformer.py
"""
#%%
from collections import namedtuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from utils.numerical import ClusterBasedNormalizer
#from utils.categorical import OneHotEncoder
from rdt.transformers import OneHotEncoder
# from rdt.transformers import ClusterBasedNormalizer, OneHotEncoder

SpanInfo = namedtuple('SpanInfo', ['dim', 'activation_fn'])
ColumnTransformInfo = namedtuple(
    'ColumnTransformInfo', [
        'column_name', 'column_type', 'transform', 'output_info', 'output_dimensions'
    ]
)

#%%
class DataTransformer(object):
    """Data Transformer.
    Model continuous columns with a BayesianGMM and normalized to a scalar [0, 1] and a vector.
    Discrete columns are encoded using a scikit-learn OneHotEncoder.
    """

    def __init__(self, max_clusters=10, weight_threshold=0.005):
        """Create a data transformer.
        Args:
            max_clusters (int):
                Maximum number of Gaussian distributions in Bayesian GMM.
            weight_threshold (float):
                Weight threshold for a Gaussian distribution to be kept.
        """
        self._max_clusters = max_clusters
        self._weight_threshold = weight_threshold

    def _fit_continuous(self, data, random_state=0):
        """Train Bayesian GMM for continuous columns.
        Args:
            data (pd.DataFrame):
                A dataframe containing a column.
        Returns:
            namedtuple:
                A ``ColumnTransformInfo`` object.
        """
        column_name = data.columns[0]
        gm = ClusterBasedNormalizer(model_missing_values=True, max_clusters=min(len(data), 10),
                                    random_state=random_state)
        gm.fit(data, column_name)
        num_components = sum(gm.valid_component_indicator)

        return ColumnTransformInfo(
            column_name=column_name, column_type='continuous', transform=gm,
            output_info=[SpanInfo(1, 'tanh'), SpanInfo(num_components, 'softmax')],
            output_dimensions=1 + num_components)

    def _fit_discrete(self, data):
        """Fit one hot encoder for discrete column.
        Args:
            data (pd.DataFrame):
                A dataframe containing a column.
        Returns:
            namedtuple:
                A ``ColumnTransformInfo`` object.
        """
        column_name = data.columns[0]
        ohe = OneHotEncoder()
        ohe.fit(data, column_name)
        num_categories = len(ohe.dummies)

        return ColumnTransformInfo(
            column_name=column_name, column_type='discrete', transform=ohe,
            output_info=[SpanInfo(num_categories, 'softmax')],
            output_dimensions=num_categories)

    def fit(self, raw_data, discrete_columns=(), random_state=0):
        """Fit the ``DataTransformer``.
        Fits a ``ClusterBasedNormalizer`` for continuous columns and a
        ``OneHotEncoder`` for discrete columns.
        This step also counts the #columns in matrix data and span information.
        """
        self.output_info_list = []
        self.output_dimensions = 0
        self.dataframe = True

        if not isinstance(raw_data, pd.DataFrame):
            self.dataframe = False
            # work around for RDT issue #328 Fitting with numerical column names fails
            discrete_columns = [str(column) for column in discrete_columns]
            column_names = [str(num) for num in range(raw_data.shape[1])]
            raw_data = pd.DataFrame(raw_data, columns=column_names)

        self._column_raw_dtypes = raw_data.infer_objects().dtypes
        self._column_transform_info_list = []
        for column_name in raw_data.columns:
            if column_name in discrete_columns:
                column_transform_info = self._fit_discrete(raw_data[[column_name]])
            else:
                column_transform_info = self._fit_continuous(raw_data[[column_name]],
                                                             random_state=random_state)

            self.output_info_list.append(column_transform_info.output_info)
            self.output_dimensions += column_transform_info.output_dimensions
            self._column_transform_info_list.append(column_transform_info)

    def _transform_continuous(self, column_transform_info, data):
        column_name = data.columns[0]
        data[column_name] = data[column_name].to_numpy().flatten()
        gm = column_transform_info.transform
        transformed = gm.transform(data)

        #  Converts the transformed data to the appropriate output format.
        #  The first column (ending in '.normalized') stays the same,
        #  but the lable encoded column (ending in '.component') is one hot encoded.
        output = np.zeros((len(transformed), column_transform_info.output_dimensions))
        output[:, 0] = transformed[f'{column_name}.normalized'].to_numpy()
        index = transformed[f'{column_name}.component'].to_numpy().astype(int)
        output[np.arange(index.size), index + 1] = 1.0

        return output

    def _transform_discrete(self, column_transform_info, data):
        ohe = column_transform_info.transform
        return ohe.transform(data).to_numpy()

    def _synchronous_transform(self, raw_data, column_transform_info_list):
        """Take a Pandas DataFrame and transform columns synchronous.
        Outputs a list with Numpy arrays.
        """
        column_data_list = []
        for column_transform_info in column_transform_info_list:
            column_name = column_transform_info.column_name
            data = raw_data[[column_name]]
            if column_transform_info.column_type == 'continuous':
                column_data_list.append(self._transform_continuous(column_transform_info, data))
            else:
                column_data_list.append(self._transform_discrete(column_transform_info, data))

        return column_data_list

    # def _parallel_transform(self, raw_data, column_transform_info_list):
    #     """Take a Pandas DataFrame and transform columns in parallel.
    #     Outputs a list with Numpy arrays.
    #     """
    #     processes = []
    #     for column_transform_info in column_transform_info_list:
    #         column_name = column_transform_info.column_name
    #         data = raw_data[[column_name]]
    #         process = None
    #         if column_transform_info.column_type == 'continuous':
    #             process = delayed(self._transform_continuous)(column_transform_info, data)
    #         else:
    #             process = delayed(self._transform_discrete)(column_transform_info, data)
    #         processes.append(process)

    #     return Parallel(n_jobs=-1)(processes)

    def transform(self, raw_data):
        """Take raw data and output a matrix data."""
        if not isinstance(raw_data, pd.DataFrame):
            column_names = [str(num) for num in range(raw_data.shape[1])]
            raw_data = pd.DataFrame(raw_data, columns=column_names)

        # Only use parallelization with larger data sizes.
        # Otherwise, the transformation will be slower.
        column_data_list = self._synchronous_transform(
            raw_data,
            self._column_transform_info_list
        )
        # if raw_data.shape[0] < 500:
        #     column_data_list = self._synchronous_transform(
        #         raw_data,
        #         self._column_transform_info_list
        #     )
        # else:
        #     column_data_list = self._parallel_transform(
        #         raw_data,
        #         self._column_transform_info_list
        #     )

        return np.concatenate(column_data_list, axis=1).astype(float)

    def _inverse_transform_continuous(self, column_transform_info, column_data, sigmas, st):
        gm = column_transform_info.transform
        data = pd.DataFrame(column_data[:, :2], columns=list(gm.get_output_sdtypes()))
        data.iloc[:, 1] = np.argmax(column_data[:, 1:], axis=1)
        if sigmas is not None:
            selected_normalized_value = np.random.normal(data.iloc[:, 0], sigmas[st])
            data.iloc[:, 0] = selected_normalized_value

        return gm.reverse_transform(data)

    def _inverse_transform_discrete(self, column_transform_info, column_data):
        ohe = column_transform_info.transform
        data = pd.DataFrame(column_data, columns=list(ohe.get_output_sdtypes()))
        return ohe.reverse_transform(data)[column_transform_info.column_name]

    def inverse_transform(self, data, sigmas=None):
        """Take matrix data and output raw data.
        Output uses the same type as input to the transform function.
        Either np array or pd dataframe.
        """
        st = 0
        recovered_column_data_list = []
        column_names = []
        for column_transform_info in self._column_transform_info_list:
            dim = column_transform_info.output_dimensions
            column_data = data[:, st:st + dim]
            if column_transform_info.column_type == 'continuous':
                recovered_column_data = self._inverse_transform_continuous(
                    column_transform_info, column_data, sigmas, st)
            else:
                recovered_column_data = self._inverse_transform_discrete(
                    column_transform_info, column_data)

            recovered_column_data_list.append(recovered_column_data)
            column_names.append(column_transform_info.column_name)
            st += dim

        recovered_data = np.column_stack(recovered_column_data_list)
        recovered_data = (pd.DataFrame(recovered_data, columns=column_names)
                          .astype(self._column_raw_dtypes))
        if not self.dataframe:
            recovered_data = recovered_data.to_numpy()

        return recovered_data

    def convert_column_name_value_to_id(self, column_name, value):
        """Get the ids of the given `column_name`."""
        discrete_counter = 0
        column_id = 0
        for column_transform_info in self._column_transform_info_list:
            if column_transform_info.column_name == column_name:
                break
            if column_transform_info.column_type == 'discrete':
                discrete_counter += 1

            column_id += 1

        else:
            raise ValueError(f"The column_name `{column_name}` doesn't exist in the data.")

        ohe = column_transform_info.transform
        data = pd.DataFrame([value], columns=[column_transform_info.column_name])
        one_hot = ohe.transform(data).to_numpy()[0]
        if sum(one_hot) == 0:
            raise ValueError(f"The value `{value}` doesn't exist in the column `{column_name}`.")

        return {
            'discrete_column_id': discrete_counter,
            'column_id': column_id,
            'value_id': np.argmax(one_hot)
        }
#%%
# class DataTransformer(object):
#     """Data Transformer.
#     Model continuous columns with a BayesianGMM and normalized to a scalar [0, 1] and a vector.
#     Discrete columns are encoded using a scikit-learn OneHotEncoder.
#     """

#     def __init__(self, max_clusters=10, weight_threshold=0.005):
#         """Create a data transformer.
#         Args:
#             max_clusters (int):
#                 Maximum number of Gaussian distributions in Bayesian GMM.
#             weight_threshold (float):
#                 Weight threshold for a Gaussian distribution to be kept.
#         """
#         self._max_clusters = max_clusters
#         self._weight_threshold = weight_threshold

#     def _fit_continuous(self, data, random_state=1):
#         """Train Bayesian GMM for continuous columns.
#         Args:
#             data (pd.DataFrame):
#                 A dataframe containing a column.
#         Returns:
#             namedtuple:
#                 A ``ColumnTransformInfo`` object.
#         """
#         column_name = data.columns[0]
#         gm = ClusterBasedNormalizer(model_missing_values=True, max_clusters=min(len(data), 10),
#                                     random_state=random_state)
#         gm.fit(data, column_name)
#         num_components = sum(gm.valid_component_indicator)

#         return ColumnTransformInfo(
#             column_name=column_name, column_type='continuous', transform=gm,
#             output_info=[SpanInfo(1, 'tanh'), SpanInfo(num_components, 'softmax')],
#             output_dimensions=1 + num_components)

#     def _fit_discrete(self, data):
#         """Fit one hot encoder for discrete column.
#         Args:
#             data (pd.DataFrame):
#                 A dataframe containing a column.
#         Returns:
#             namedtuple:
#                 A ``ColumnTransformInfo`` object.
#         """
#         column_name = data.columns[0]
#         ohe = OneHotEncoder()
#         ohe.fit(data, column_name)
#         num_categories = len(ohe.dummies)

#         return ColumnTransformInfo(
#             column_name=column_name, column_type='discrete', transform=ohe,
#             output_info=[SpanInfo(num_categories, 'softmax')],
#             output_dimensions=num_categories)

#     def fit(self, raw_data, discrete_columns=(), random_state=1):
#         """Fit the ``DataTransformer``.
#         Fits a ``ClusterBasedNormalizer`` for continuous columns and a
#         ``OneHotEncoder`` for discrete columns.
#         This step also counts the #columns in matrix data and span information.
#         """
#         self.output_info_list = []
#         self.output_dimensions = 0
#         self.dataframe = True

#         if not isinstance(raw_data, pd.DataFrame):
#             self.dataframe = False
#             # work around for RDT issue #328 Fitting with numerical column names fails
#             discrete_columns = [str(column) for column in discrete_columns]
#             column_names = [str(num) for num in range(raw_data.shape[1])]
#             raw_data = pd.DataFrame(raw_data, columns=column_names)

#         self._column_raw_dtypes = raw_data.infer_objects().dtypes
#         self._column_transform_info_list = []
#         for column_name in raw_data.columns:
#             if column_name in discrete_columns:
#                 column_transform_info = self._fit_discrete(raw_data[[column_name]])
#             else:
#                 column_transform_info = self._fit_continuous(raw_data[[column_name]],
#                                                              random_state=random_state)

#             self.output_info_list.append(column_transform_info.output_info)
#             self.output_dimensions += column_transform_info.output_dimensions
#             self._column_transform_info_list.append(column_transform_info)

#     def transform_continuous( column_transform_info, data):
#         column_name = data.columns[0]
#         data[column_name] = data[column_name].to_numpy().flatten()
#         gm = column_transform_info.transform
#         transformed = gm.transform(data)

#         #  Converts the transformed data to the appropriate output format.
#         #  The first column (ending in '.normalized') stays the same,
#         #  but the lable encoded column (ending in '.component') is one hot encoded.
#         output = np.zeros((len(transformed), column_transform_info.output_dimensions))
#         output[:, 0] = transformed[f'{column_name}.normalized'].to_numpy()
#         index = transformed[f'{column_name}.component'].to_numpy().astype(int)
#         output[np.arange(index.size), index + 1] = 1.0

#         return output
     
#     ################################################################################### 
#     def transform_discrete(column_transform_info, data):
#         ohe = column_transform_info.transform
#         return ohe.transform(data).to_numpy()
#     ###################################################################################    

#     def synchronous_transform(raw_data, column_transform_info_list):
#         """Take a Pandas DataFrame and transform columns synchronous.
#         Outputs a list with Numpy arrays.
#         """
        
#         column_transform_info = column_transform_info_list[5]   ######################
#         column_data_list = []
#         for column_transform_info in column_transform_info_list:
#             column_name = column_transform_info.column_name
#             data = raw_data[[column_name]]   #############################
#             if column_transform_info.column_type == 'continuous':
#                 column_data_list.append(transform_continuous(column_transform_info, data))
#             else:
#                 column_data_list.append(transform_discrete(column_transform_info, data))   ######################

#         return column_data_list
    
    

#     # def _parallel_transform(self, raw_data, column_transform_info_list):
#     #     """Take a Pandas DataFrame and transform columns in parallel.
#     #     Outputs a list with Numpy arrays.
#     #     """
#     #     processes = []
#     #     for column_transform_info in column_transform_info_list:
#     #         column_name = column_transform_info.column_name
#     #         data = raw_data[[column_name]]
#     #         process = None
#     #         if column_transform_info.column_type == 'continuous':
#     #             process = delayed(self._transform_continuous)(column_transform_info, data)
#     #         else:
#     #             process = delayed(self._transform_discrete)(column_transform_info, data)
#     #         processes.append(process)

#     #     return Parallel(n_jobs=-1)(processes)
#     raw_data = train
#     column_transform_info_list = transformer._column_transform_info_list
    
#     def transform(raw_data):
#         """Take raw data and output a matrix data."""
#         if not isinstance(raw_data, pd.DataFrame):
#             column_names = [str(num) for num in range(raw_data.shape[1])]
#             raw_data = pd.DataFrame(raw_data, columns=column_names)

#         # Only use parallelization with larger data sizes.
#         # Otherwise, the transformation will be slower.
#         column_data_list = synchronous_transform(
#             raw_data,
#             column_transform_info_list
#         )
#         # if raw_data.shape[0] < 500:
#         #     column_data_list = self._synchronous_transform(
#         #         raw_data,
#         #         self._column_transform_info_list
#         #     )
#         # else:
#         #     column_data_list = self._parallel_transform(
#         #         raw_data,
#         #         self._column_transform_info_list
#         #     )

#         return np.concatenate(column_data_list, axis=1).astype(float)

#     def _inverse_transform_continuous(self, column_transform_info, column_data, sigmas, st):
#         gm = column_transform_info.transform
#         data = pd.DataFrame(column_data[:, :2], columns=list(gm.get_output_sdtypes()))
#         data.iloc[:, 1] = np.argmax(column_data[:, 1:], axis=1)
#         if sigmas is not None:
#             selected_normalized_value = np.random.normal(data.iloc[:, 0], sigmas[st])
#             data.iloc[:, 0] = selected_normalized_value

#         return gm.reverse_transform(data)

#     def _inverse_transform_discrete(self, column_transform_info, column_data):
#         ohe = column_transform_info.transform
#         data = pd.DataFrame(column_data, columns=list(ohe.get_output_sdtypes()))
#         return ohe.reverse_transform(data)[column_transform_info.column_name]

#     def inverse_transform(self, data, sigmas=None):
#         """Take matrix data and output raw data.
#         Output uses the same type as input to the transform function.
#         Either np array or pd dataframe.
#         """
#         st = 0
#         recovered_column_data_list = []
#         column_names = []
#         for column_transform_info in self._column_transform_info_list:
#             dim = column_transform_info.output_dimensions
#             column_data = data[:, st:st + dim]
#             if column_transform_info.column_type == 'continuous':
#                 recovered_column_data = self._inverse_transform_continuous(
#                     column_transform_info, column_data, sigmas, st)
#             else:
#                 recovered_column_data = self._inverse_transform_discrete(
#                     column_transform_info, column_data)

#             recovered_column_data_list.append(recovered_column_data)
#             column_names.append(column_transform_info.column_name)
#             st += dim

#         recovered_data = np.column_stack(recovered_column_data_list)
#         recovered_data = (pd.DataFrame(recovered_data, columns=column_names)
#                           .astype(self._column_raw_dtypes))
#         if not self.dataframe:
#             recovered_data = recovered_data.to_numpy()

#         return recovered_data

#     def convert_column_name_value_to_id(self, column_name, value):
#         """Get the ids of the given `column_name`."""
#         discrete_counter = 0
#         column_id = 0
#         for column_transform_info in self._column_transform_info_list:
#             if column_transform_info.column_name == column_name:
#                 break
#             if column_transform_info.column_type == 'discrete':
#                 discrete_counter += 1

#             column_id += 1

#         else:
#             raise ValueError(f"The column_name `{column_name}` doesn't exist in the data.")

#         ohe = column_transform_info.transform
#         data = pd.DataFrame([value], columns=[column_transform_info.column_name])
#         one_hot = ohe.transform(data).to_numpy()[0]
#         if sum(one_hot) == 0:
#             raise ValueError(f"The value `{value}` doesn't exist in the column `{column_name}`.")

#         return {
#             'discrete_column_id': discrete_counter,
#             'column_id': column_id,
#             'value_id': np.argmax(one_hot)
#         }
# #%%
# """BaseTransformer module."""
# import abc
# import inspect
# import warnings

# import numpy as np
# import pandas as pd


# class BaseTransformer:
#     """Base class for all transformers.
#     The ``BaseTransformer`` class contains methods that must be implemented
#     in order to create a new transformer. The ``_fit`` method is optional,
#     and ``fit_transform`` method is already implemented.
#     """

#     INPUT_SDTYPE = None
#     SUPPORTED_SDTYPES = None
#     IS_GENERATOR = None

#     columns = None
#     column_prefix = None
#     output_columns = None
#     missing_value_replacement = None

#     # def __init__(self):
#     #     self.output_properties = {None: {'sdtype': 'float', 'next_transformer': None}}
#     output_properties = {None: {'sdtype': 'float', 'next_transformer': None}}


#     def _set_missing_value_replacement(default, missing_value_replacement):
#         if missing_value_replacement is None:
#             warnings.warn(
#                 "Setting 'missing_value_replacement' to 'None' is no longer supported. "
#                 f"Imputing with the '{default}' instead.", FutureWarning
#             )
#             missing_value_replacement = default
#         else:
#             missing_value_replacement = missing_value_replacement

#     @classmethod
#     def get_name(cls):
#         """Return transformer name.
#         Returns:
#             str:
#                 Transformer name.
#         """
#         return cls.__name__

#     @classmethod
#     def get_subclasses(cls):
#         """Recursively find subclasses of this Baseline.
#         Returns:
#             list:
#                 List of all subclasses of this class.
#         """
#         subclasses = []
#         for subclass in cls.__subclasses__():
#             if abc.ABC not in subclass.__bases__:
#                 subclasses.append(subclass)

#             subclasses += subclass.get_subclasses()

#         return subclasses

#     @classmethod
#     def get_input_sdtype(cls):
#         """Return the input sdtype supported by the transformer.
#         Returns:
#             string:
#                 Accepted input sdtype of the transformer.
#         """
#         return cls.INPUT_SDTYPE

#     @classmethod
#     def get_supported_sdtypes(cls):
#         """Return the supported sdtypes by the transformer.
#         Returns:
#             list:
#                 Accepted input sdtypes of the transformer.
#         """
#         return cls.SUPPORTED_SDTYPES or [cls.INPUT_SDTYPE]

#     def _get_output_to_property(property_):
#         output = {}
#         for output_column, properties in output_properties.items():
#             # if 'sdtype' is not in the dict, ignore the column
#             if property_ not in properties:
#                 continue
#             if output_column is None:
#                 output[f'{column_prefix}'] = properties[property_]
#             else:
#                 output[f'{column_prefix}.{output_column}'] = properties[property_]

#         return output

#     def get_output_sdtypes():
#         """Return the output sdtypes produced by this transformer.
#         Returns:
#             dict:
#                 Mapping from the transformed column names to the produced sdtypes.
#         """
#         return _get_output_to_property('sdtype')

#     def get_next_transformers():
#         """Return the suggested next transformer to be used for each column.
#         Returns:
#             dict:
#                 Mapping from transformed column names to the transformers to apply to each column.
#         """
#         return _get_output_to_property('next_transformer')

#     def is_generator():
#         """Return whether this transformer generates new data or not.
#         Returns:
#             bool:
#                 Whether this transformer generates new data or not.
#         """
#         return bool(IS_GENERATOR)

#     def get_input_column():
#         """Return input column name for transformer.
#         Returns:
#             str:
#                 Input column name.
#         """
#         return columns[0]

#     def get_output_columns():
#         """Return list of column names created in ``transform``.
#         Returns:
#             list:
#                 Names of columns created during ``transform``.
#         """
#         return list(_get_output_to_property('sdtype'))

#     def _store_columns(columns, data):
#         if isinstance(columns, tuple) and columns not in data:
#             columns = list(columns)
#         elif not isinstance(columns, list):
#             columns = [columns]

#         missing = set(columns) - set(data.columns)
#         if missing:
#             raise KeyError(f'Columns {missing} were not present in the data.')

#         #columns = columns

#     @staticmethod
#     def _get_columns_data(data, columns):
#         if len(columns) == 1:
#             columns = columns[0]

#         return data[columns].copy()
    
#     ################################################################
#     @staticmethod
#     def _add_columns_to_data(data, transformed_data, transformed_names):
#         """Add new columns to a ``pandas.DataFrame``.
#         Args:
#             - data (pd.DataFrame):
#                 The ``pandas.DataFrame`` to which the new columns have to be added.
#             - transformed_data (pd.DataFrame, pd.Series, np.ndarray):
#                 The data of the new columns to be added.
#             - transformed_names (list, np.ndarray):
#                 The names of the new columns to be added.
#         Returns:
#             ``pandas.DataFrame`` with the new columns added.
#         """
#         if isinstance(transformed_data, (pd.Series, np.ndarray)):
#             transformed_data = pd.DataFrame(transformed_data, columns=transformed_names)

#         if transformed_names:
#             # When '#' is added to the column_prefix of a transformer
#             # the columns of transformed_data and transformed_names don't match
#             transformed_data.columns = transformed_names
#             data = pd.concat([data, transformed_data.set_index(data.index)], axis=1)

#         return data

#     def _build_output_columns(self, data):
#         self.column_prefix = '#'.join(self.columns)
#         self.output_columns = self.get_output_columns()

#         # make sure none of the generated `output_columns` exists in the data,
#         # except when a column generates another with the same name
#         output_columns = set(self.output_columns) - set(self.columns)
#         repeated_columns = set(output_columns) & set(data.columns)
#         while repeated_columns:
#             warnings.warn(
#                 f'The output columns {repeated_columns} generated by the {self.get_name()} '
#                 'transformer already exist in the data (or they have already been generated '
#                 "by some other transformer). Appending a '#' to the column name to distinguish "
#                 'between them.'
#             )
#             self.column_prefix += '#'
#             self.output_columns = self.get_output_columns()
#             output_columns = set(self.output_columns) - set(self.columns)
#             repeated_columns = set(output_columns) & set(data.columns)

#     def __repr__(self):
#         """Represent initialization of transformer as text.
#         Returns:
#             str:
#                 The name of the transformer followed by any non-default parameters.
#         """
#         class_name = self.__class__.get_name()
#         custom_args = []
#         args = inspect.getfullargspec(self.__init__)
#         keys = args.args[1:]
#         defaults = args.defaults or []
#         defaults = dict(zip(keys, defaults))
#         instanced = {key: getattr(self, key) for key in keys}

#         if defaults == instanced:
#             return f'{class_name}()'

#         for arg, value in instanced.items():
#             if defaults[arg] != value:
#                 custom_args.append(f'{arg}={repr(value)}')

#         args_string = ', '.join(custom_args)
#         return f'{class_name}({args_string})'

#     def _fit(self, columns_data):
#         """Fit the transformer to the data.
#         Args:
#             columns_data (pandas.DataFrame or pandas.Series):
#                 Data to transform.
#         """
#         raise NotImplementedError()

#     def fit(self, data, column):
#         """Fit the transformer to a ``column`` of the ``data``.
#         Args:
#             data (pandas.DataFrame):
#                 The entire table.
#             column (str):
#                 Column name. Must be present in the data.
#         """
#         self._store_columns(column, data)
#         columns_data = self._get_columns_data(data, self.columns)
#         self._fit(columns_data)
#         self._build_output_columns(data)

#     def _transform( columns_data):
#         """Transform the data.
#         Args:
#             columns_data (pandas.DataFrame or pandas.Series):
#                 Data to transform.
#         Returns:
#             pandas.DataFrame or pandas.Series:
#                 Transformed data.
#         """
#         raise NotImplementedError()

#     def transform(self, data):
#         """Transform the `self.columns` of the `data`.
#         Args:
#             data (pandas.DataFrame):
#                 The entire table.
#         Returns:
#             pd.DataFrame:
#                 The entire table, containing the transformed data.
#         """
#         # if `data` doesn't have the columns that were fitted on, don't transform
#         if any(column not in data.columns for column in self.columns):
#             return data
        
#         data = data.copy()
#         columns_data = _get_columns_data(data, data.columns)
#         transformed_data = _transform(columns_data)
#         data = data.drop(data.columns, axis=1)
#         data = self._add_columns_to_data(data, transformed_data, self.output_columns)

#         return data
    
#     def _add_columns_to_data(data, transformed_data, transformed_names):
#         """Add new columns to a ``pandas.DataFrame``.
#         Args:
#             - data (pd.DataFrame):
#                 The ``pandas.DataFrame`` to which the new columns have to be added.
#             - transformed_data (pd.DataFrame, pd.Series, np.ndarray):
#                 The data of the new columns to be added.
#             - transformed_names (list, np.ndarray):
#                 The names of the new columns to be added.
#         Returns:
#             ``pandas.DataFrame`` with the new columns added.
#         """
#         if isinstance(transformed_data, (pd.Series, np.ndarray)):
#             transformed_data = pd.DataFrame(transformed_data, columns=transformed_names)

#         if transformed_names:
#             # When '#' is added to the column_prefix of a transformer
#             # the columns of transformed_data and transformed_names don't match
#             transformed_data.columns = transformed_names
#             data = pd.concat([data, transformed_data.set_index(data.index)], axis=1)

#         return data

#     def fit_transform(self, data, column):
#         """Fit the transformer to a `column` of the `data` and then transform it.
#         Args:
#             data (pandas.DataFrame):
#                 The entire table.
#             column (str):
#                 A column name.
#         Returns:
#             pd.DataFrame:
#                 The entire table, containing the transformed data.
#         """
#         self.fit(data, column)
#         return self.transform(data)

#     # def _reverse_transform(self, columns_data):
#     #     """Revert the transformations to the original values.
#     #     Args:
#     #         columns_data (pandas.DataFrame or pandas.Series):
#     #             Data to revert.
#     #     Returns:
#     #         pandas.DataFrame or pandas.Series:
#     #             Reverted data.
#     #     """
#     #     raise NotImplementedError()

#     # def reverse_transform(self, data):
#     #     """Revert the transformations to the original values.
#     #     Args:
#     #         data (pandas.DataFrame):
#     #             The entire table.
#     #     Returns:
#     #         pandas.DataFrame:
#     #             The entire table, containing the reverted data.
#     #     """
#     #     # if `data` doesn't have the columns that were transformed, don't reverse_transform
#     #     if any(column not in data.columns for column in self.output_columns):
#     #         return data

#     #     data = data.copy()
#     #     columns_data = self._get_columns_data(data, self.output_columns)
#     #     reversed_data = self._reverse_transform(columns_data)
#     #     data = data.drop(self.output_columns, axis=1)
#     #     data = self._add_columns_to_data(data, reversed_data, self.columns)

#     #     return data
#     #%%
# def _get_columns_data(data, columns):
#         if len(columns) == 1:
#             columns = columns[0]

#         return data[columns].copy()
# #%%
# class OneHotEncoder(BaseTransformer):
#     """OneHotEncoding for categorical data.
#     This transformer replaces a single vector with N unique categories in it
#     with N vectors which have 1s on the rows where the corresponding category
#     is found and 0s on the rest.
#     Null values are considered just another category.
#     """

#     INPUT_SDTYPE = 'categorical'
#     SUPPORTED_SDTYPES = ['categorical', 'boolean']
#     dummies = None
#     _dummy_na = None
#     _num_dummies = None
#     _dummy_encoded = False
#     _indexer = None
#     _uniques = None

#     @staticmethod
#     def _prepare_data(data):
#         """Transform data to appropriate format.
#         If data is a valid list or a list of lists, transforms it into an np.array,
#         otherwise returns it.
#         Args:
#             data (pandas.Series or pandas.DataFrame):
#                 Data to prepare.
#         Returns:
#             pandas.Series or numpy.ndarray
#         """
#         if isinstance(data1, list):
#             data1 = np.array(data1)

#         if len(data1.shape) > 2:
#             raise ValueError('Unexpected format.')
#         if len(data1.shape) == 2:
#             if data1.shape[1] != 1:
#                 raise ValueError('Unexpected format.')

#             data = data1[:, 0]

#         return data

#     def _fit(self, data):
#         """Fit the transformer to the data.
#         Get the pandas `dummies` which will be used later on for OneHotEncoding.
#         Args:
#             data (pandas.Series or pandas.DataFrame):
#                 Data to fit the transformer to.
#         """
#         data = self._prepare_data(data)

#         null = pd.isna(data).to_numpy()
#         self._uniques = list(pd.unique(data[~null]))
#         self._dummy_na = null.any()
#         self._num_dummies = len(self._uniques)
#         self._indexer = list(range(self._num_dummies))
#         self.dummies = self._uniques.copy()

#         if not np.issubdtype(data.dtype, np.number):
#             self._dummy_encoded = True

#         if self._dummy_na:
#             self.dummies.append(np.nan)

#         self.output_properties = {
#             f'value{i}': {'sdtype': 'float', 'next_transformer': None}
#             for i in range(len(self.dummies))
#         }

#     def _transform_helper(self, data):
#         if self._dummy_encoded:
#             coder = self._indexer
#             codes = pd.Categorical(data, categories=self._uniques).codes
#         else:
#             coder = self._uniques
#             codes = data

#         rows = len(data)
#         dummies = np.broadcast_to(coder, (rows, self._num_dummies))
#         coded = np.broadcast_to(codes, (self._num_dummies, rows)).T
#         array = (coded == dummies).astype(int)

#         if self._dummy_na:
#             null = np.zeros((rows, 1), dtype=int)
#             null[pd.isna(data)] = 1
#             array = np.append(array, null, axis=1)

#         return array

#     def _transform(self, data):
#         """Replace each category with the OneHot vectors.
#         Args:
#             data (pandas.Series, list or list of lists):
#                 Data to transform.
#         Returns:
#             numpy.ndarray
#         """
#         data1= data.copy()
#         data2 = _prepare_data(data1)
#         unique_data = {np.nan if pd.isna(x) else x for x in pd.unique(data)}
#         unseen_categories = unique_data - set(self.dummies)
#         if unseen_categories:
#             # Select only the first 5 unseen categories to avoid flooding the console.
#             examples_unseen_categories = set(list(unseen_categories)[:5])
#             warnings.warn(
#                 f'The data contains {len(unseen_categories)} new categories that were not '
#                 f'seen in the original data (examples: {examples_unseen_categories}). Creating '
#                 'a vector of all 0s. If you want to model new categories, '
#                 'please fit the transformer again with the new data.'
#             )

#         return self._transform_helper(data)
# #%%