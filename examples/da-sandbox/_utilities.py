from math import isclose
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
    print(str(module_name) + ": started", flush=True)

def print_end_of(module_name):
    """
    Announce the end of a module

    Input:
        module_name (str): name of the module
    Output:
        None
    """
    print(" " * LEFT_PAD, end='')
    print(str(module_name) + ": ended\n", flush=True)

def print_info_module(
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
    print("*" + " " * (LEFT_PAD - 1) + str(module_name) + ": ", end='')
    print(*args, **kwargs, flush=True)

def print_warning_module(
        module_name,
        *args,
        **kwargs):
    """
    Print warning in a module

    Input:
        module_name (str): name of the module
        *args, **kwargs: to be passed to the 'print' function
    Output:
        None
    """
    print("!" + " " * (LEFT_PAD - 1) + str(module_name) + ": ", end='')
    print(*args, **kwargs, flush=True)

def print_info(
        *args,
        **kwargs):
    """
    Print info in the main program

    Input:
        *args, **kwargs: to be passed to the 'print' function
    Output:
        None
    """
    print("*" + " " * (LEFT_PAD - 1), end='')
    print(*args, **kwargs, flush=True)

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

def modulo_is_close_to_zero(
        time,
        modulo_interval,
        eps=0.125):
    """
    Determine if (time mod modulo_interval) ≈ 0

    Input:
        time (int),
             (float): time, real number (positive or negative)
        modulo_interval (int),
                        (float): modulo interval, positive real
        eps (float): (-eps/2, eps/2) is the approximate interval
    Output:
        is_close (bool): whether or not time is close to 0
    """
    return isclose( (time + eps/2) % modulo_interval, 0.0, abs_tol=eps )

def are_close(
        time1,
        time2,
        eps=0.125):
    """
    Determine if |time1 - time2| ≤ eps/2

    Input:
        time1 (int),
              (float): time, real number (positive or negative)
        time2 (int),
              (float): time, real number (positive or negative)
        eps (float): eps/2 is the maximum distance between time1 and time2
    Output:
        are_close (bool): whether or not time1 and time2 are close
    """
    return isclose(time1, time2, abs_tol=eps/2)


