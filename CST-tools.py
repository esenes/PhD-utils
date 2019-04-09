print('Version 0.1')
print('Please report bugs to eugenio.senes@cern.ch\nUse at your own risk.')
import numpy as np
import pandas as pnd

#---------------------------------------------------------------------
################## general dataframe utilities #######################
#---------------------------------------------------------------------

def create_empty_dataframe(param_values, column_names):
    '''
    Create an empty pandas dataframe of np.ndarr

    Last modified: 08.04.2019 by Eugenio Senes
    '''
    return pnd.DataFrame([[np.empty(0)]*len(column_names)]*len(param_values), index=param_values, columns=column_names)


#---------------------------------------------------------------------
##################### for PARAMETRIC RESULTS #########################
#---------------------------------------------------------------------
def read_CST_param_results(fname,delimiter = '#', verbose=True):
    '''
    Read a CST ASCII output file from a parametric simulation

    Inputs:
    - fname:        filename, including path
    - delimiter:    comment delimiter (default = #)
    - verbose:      print the headers as you parse

    Outputs:
    - file_divided: the file split at each delimiter. Data are
                    numpy arrays, the rest are strings.
    - data_flag:    flag for the data

    Access the data after the split as file_divided[data_flag].
    The other lines are the headers, printed when the function
    is called.

    Last modified: 08.04.2019 by Eugenio Senes
    '''

    with open(fname,'r') as f:
        full_txt = f.readlines()
        row_n_delim = []
        #search the delimiters
        for k, row in enumerate(full_txt):
            if row[0] == delimiter:
                row_n_delim.append(k)
                print(row) if verbose else None
        #fraction the file
        file_divided = []
        for k in list(range(len(row_n_delim))):
            if k<len(row_n_delim)-1:
                #If data, cast to numpy array
                if row_n_delim[k+1]-row_n_delim[k]>1: # this is a number array
                    frac = np.array([[float(j)for j in row.split('\t')] for row in full_txt[row_n_delim[k]+1:row_n_delim[k+1]]])
                    file_divided.append(frac.T)
                else: # headers --> stay strings
                    file_divided.append(full_txt[row_n_delim[k]:row_n_delim[k+1]])
            else:
                frac = np.array([[float(j)for j in row.split('\t')] for row in full_txt[row_n_delim[k]+1:-1]])
                file_divided.append(frac.T)

        #cast the list to numpy arrays and flag what is data
        data_flag = [type(k)==np.ndarray for k in file_divided]

        return np.array(file_divided), data_flag

def find_single_param(header, param_name, verbose=True):
    '''
    Find a single parameter value in the header

    Inputs:
    - header:       the string containing the full header
    - param_name:   the name to search
    - verbose:      print info during processing

    Outputs:
    - value:        the parameter value

    Assumes that the data structure contains '   ... param_name=value ; ...'
    and that the header starts with '#Parameters ='

    Last modified: 09.04.2019 by Eugenio Senes
    '''
    header = header[0] # flatten
    if header[:13]=='#Parameters =':
        print(header) if verbose else None
        for par in header.split('; '):
            if par[0:len(param_name)] == param_name:
#                 return par
                return float(par[len(param_name)+1:]) #+1 to skip the =
    else:
        print('Parameter not found in the list')
        return None
