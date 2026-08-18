import pandas as pd
from itertools import chain
import numpy as np
import cmath
import os
from pyomo.environ import *

PHASE_ORDER = np.array(['A', 'B', 'C'])

def filter_by_phase_to_dict(df, phase):
    """
    The use is to filter a df according to phases and pass it as a dictionary to use in pyomo model.
    """
    filtered_df = df[df['Phase'] == phase].copy()  
    filtered_df.drop(columns=['Phase'], inplace=True) 
    load_dict = dict()
    columns = filtered_df.columns
  
    for i, t in filtered_df.iterrows():
      for col in columns:
          load_dict[(col, i+1)] = t[col]
 
    return load_dict

def infer_matrix_phases(matrix):
  matrix = np.asarray(matrix, dtype=complex)
  if matrix.shape != (3, 3):
    raise ValueError("Input matrix must be a 3x3 matrix.")
  magnitude = np.abs(matrix)
  tol = max(1e-12, 1e-9 * np.max(magnitude))
  active_rows = np.any(magnitude > tol, axis=1)
  active_cols = np.any(magnitude > tol, axis=0)
  active_phases = active_rows | active_cols
  return ''.join(PHASE_ORDER[active_phases])

def phase_map_from_config(config_numpy):
   config_numpy = np.asarray(config_numpy, dtype=complex)
   matrices = config_numpy.reshape(-1, 3, 3)
   return {
      config_number: infer_matrix_phases(matrix) for config_number, matrix in enumerate(matrices, start=1)
   }

def format_phases(value):
     return ''.join([phase for phase in 'ABC' if phase in str(value)])
  
def phases_from_config(df, config_numpy=None, need_phases=False):
  df = df.copy()
  if need_phases:
    phase_map = phase_map_from_config(
       config_numpy
    )
    config_ids = pd.to_numeric(
       df['Config'], errors='raise'
    ).astype(int)

    unknown_configs = sorted(set(config_ids) - set(phase_map))

    if unknown_configs:
       raise ValueError(f"Unknown config values found: {unknown_configs}")
    df["Phases"] = config_ids.map(phase_map)

  def format_phases(value):
     phases = [phase for phase in 'ABC' if phase in str(value)]
     return ''.join(phases)
  
  df["Phases"] = df["Phases"].apply(format_phases)
  return df

def reverse_lookup(value, params):
  reverse_lookup = {v: k for k, v in model.k.items()} 
  return reverse_lookup.get(value)

def create_matrix_dict_pandas(filename):
    """
    Reads a CSV file with continuous 3x3 matrices and creates a dictionary using Pandas.

    Args:
        filename: Path to the CSV file.

    Returns:
        A dictionary where keys are integers (matrix index) and values are dictionaries representing matrices.
    """

    data = pd.read_csv(filename, header=None, names=['A', 'B', 'C'], skiprows=1)
    
    matrix_dict = {}
    current_matrix_index = 1
    matrix_df1 = pd.DataFrame()
    
    for i in range(0, len(data), 3):
        matrix = data.iloc[i:i+3].copy()
        matrix.index = ['A', 'B', 'C']
        matrix_dict[current_matrix_index] = matrix.to_dict(orient='dict')
        matrix['Config'] = current_matrix_index
        matrix = matrix.set_index('Config', append=True)
        matrix_df1 = pd.concat([matrix_df1, matrix])
        current_matrix_index += 1

    config_dict = dict()
    columns = matrix_df1.columns
    config_numpy = matrix_df1.map(lambda x: complex(x)).to_numpy()
    
    n = 1
    count = 0
    for a, (conf, row_label) in enumerate(matrix_df1.index):
        row = matrix_df1.iloc[a]
        for c, col in enumerate(columns):
            config_dict[(row_label, conf, col)] = config_numpy[a][c]
            count += 1
            if count == 9:
                count = 0
                n += 1

    return config_dict, config_numpy


def create_res_rec_from_dicts(config):
  """
    Creates a dictionary representation of resistance and reactances from the config_numpy (array) config files.
    Args:
        config: Numpy array of size 36x3

    Returns:
        2 dictionaries for resistance and reactances where keys are integers (matrix index) and values are resistances and reactance matrix for that line configuration
    """

  a = np.array([[1], [-0.5-0.866j], [-0.5+0.866j]])
  a_H = a.conj().T
  
  a_a_H = a*a_H

  dic_real = {}
  dic_imag = {}
  c_size = int((config.shape[0])/3)
  for i in range(1,(c_size*3)+1,3):
    
    dic_real[int(i/3)+1] = config[i-1:i+2][:].real
    dic_imag[int(i/3)+1] = config[i-1:i+2][:].imag

  r = {}
  x = {}
 
  for i in range(1, c_size+1):
    r[i] = np.multiply(a_a_H.real, dic_real[i]) + np.multiply(a_a_H.imag, dic_imag[i])
    x[i] = np.multiply(a_a_H.real, dic_imag[i]) - np.multiply(a_a_H.imag, dic_real[i])
  
  return r, x
  # return dic_real, dic_imag

def convert_matrix_to_dict(matrix):
  """
  Used to convert a matrix to a dictionary
  """
  return {(i, j): matrix[i, j] for i in range(matrix.shape[0]) for j in range(matrix.shape[1])}

matrix_data = create_matrix_dict_pandas("OPF_data/IEEE123TestCase/config.csv")
r,x = create_res_rec_from_dicts(matrix_data[1])

def sort_csv(filename):
  """
  Sort a csv file by first column
  """
  df = pd.read_csv(filename)
  df = df.sort_values(by=df.columns[0])
  df.to_csv(filename)
  return df

def correct_names(df, col, name):
    """
    Used to Correctly name the index of the dataframe, the OPF-D takes only Bus names as Bus1, Bus2... etc. and Gencos name as Genco1, Genco2 etc,
    """
    if col not in df.columns:
        raise KeyError(f"Column {col!r} not found")

    df[col] = df[col].astype("string")

    mask = (
        df[col].notna()
        & ~df[col].str.startswith(name, na=False)
    )

    df.loc[mask, col] = name + df.loc[mask, col]
    return df


def data_indexing(filename, col_name, buses, name):

  df = pd.read_csv(filename)
  df.set_index(col_name, inplace=True)
  df = df.reset_index()
  df = correct_names(df, col_name, name)
  df.set_index(col_name, inplace=True)
  
  return df

def cap_values_per_phase(df):
  cap_dict = {}
  for _, row in df.iterrows():
    cap_i = row['Cap_i']
    if not pd.isnull(row['Qmax_A']):
        cap_dict[(cap_i, 'A')] = row['Qmax_A']
    if not pd.isnull(row['Qmax_B']):
        cap_dict[(cap_i, 'B')] = row['Qmax_B']
    if not pd.isnull(row['Qmax_C']):
        cap_dict[(cap_i, 'C')] = row['Qmax_C']

  return cap_dict

def descending_sort_dict(dic):
  sorted_list = sorted(dic.items(), key = lambda x:x[1][0], reverse = True)
  sorted_dict = dict(sorted_list)

  return sorted_dict

def get_duals_in_numpy_vector(df):

  df.index = df.index.map(lambda x: f"{x[0]}_{x[1]}")
  row_vector = df.T
  row_array = row_vector.to_numpy()
  return row_array

def custom_sort_key(item):
    try:
        return int(item.split('_')[1])
    except:
        return item

