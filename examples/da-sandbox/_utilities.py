import numpy as np

LEFT_PAD = 3

def print_start_of(module_name):
    """
    Announce the start of a module

    Input:
        module_name (str): name of the module
    Output:
        None
    """
    print(" " * LEFT_PAD, end='')
    print(str(module_name) + ": started")

def print_end_of(module_name):
    """
    Announce the end of a module

    Input:
        module_name (str): name of the module
    Output:
        None
    """
    print(" " * LEFT_PAD, end='')
    print(str(module_name) + ": ended\n")

def print_info(
        module_name,
        *args,
        **kwargs):
    """
    Print info in a module

    Input:
        module_name (str): name of the module
        *args, **kwargs: to be passed to the 'print' function
    Output:
        None
    """
    print(" " * LEFT_PAD + str(module_name) + ": ", end='')
    print(*args, **kwargs)

def list_of_transition_rates_to_array(list_of_rates):
    """
    Convert a list of TransitionRates into np.array

    Input:
        list_of_rates (list): list of same-sized TransitionRates
    Output:
        rates_array (np.array): (n_list, n_parameters) array of rates
    """
    n_list = len(list_of_rates)
    n_parameters = list_of_rates[0].get_clinical_parameters_total_count()
    rates_array = np.empty( (n_list, n_parameters) )

    for i, rates in enumerate(list_of_rates):
        rates_array[i] = rates.get_clinical_parameters_as_array()

    return rates_array


