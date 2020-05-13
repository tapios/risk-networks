# Ensemble Adjusted Kalman Filter (EAKF)

`eakf` implements a class called `EAKF` that represents
an Ensemble Adjusted Kalman Filter.

EAKF updates states and parameters at a particular time based on 
the difference between an observed and modeled state subject both to
observational uncertainty and an uncertainty estimated based on the spread
of states among ensemble members.

## Examples

1. `DA_forward_example.py`.

Key parameters to set on the line below `if __name__ == "__main__":` are

     * `T`: The duration of the model forward run, between observations.
     * `steps_DA`: The number of steps of duration `T`.
     * `n_samples`: The size of the ensemble used for data assimilation.

*Note: `n_samples` may need to be around `n_samples=10`. For example, `n_samples=2` is too small.*

This produces output in the directory `data/`:
    
    * `data/u.pkl`: parameters for every ensemble member at every DA step
    * `data/g.pkl`: states for every ensemble member at every DA step
    * `data/x.pkl`: states and parameters for every ensemble member at every `dt`
    * `data/error.pkl`: DA residual (misfit, or closeness between model states and observations)

2. `DA_backward_example.py`.
 
This script must be run after running `DA_forward_example.py`.
The format of `DA_backward_example.py` is similar to `DA_forward_example.py`.
Note that `n_samples` must be the same as in `DA_forward_example.py`.

3. `backwards_solve_test.py`

This script runs a model forwards, and then backwards.
It creates a plot using `DA_forward_plot.py`.

## New DA framework

Seeks to make the observations and and data more modular. Key files are

 `data.py`

Where one defines the data class, contains a method `makes_observation(time,state)`
of data at a time and which states to observe
               
 `observations`

contains 2 types of observation classes (state or time).

- State observations relate to which nodes/states are being observed, contains a
                      method `measurement(DA_window)` provides the observated states in a DA window

- Time observations relate to which times we make observations at, contains a
                      method `measurement(DA_window)` providing time in the DA_window where observations made

 `eakf.py`:

as previously, contains 2 DA methods, `obs(truth,cov)` to store an observation and  `update(state)`
to update a state based on these truth,cov values  

 `data_assimilation.py`

holds all the data,observations,DA method with an `update(DA_window)` that calls the
above methods fo perform the update step
     
## Example

 `forward_filter_example.py`

This script extends the DA framework in `DA_forward_example.py` and is written
in the new framework to take observations within a time window and possibly
limited nodes. For this it assimilates data at the time of observation and runs
the updated forward model to the end of the assimilation window.
Currently only allows one intermediate observation within an assimilation window.
