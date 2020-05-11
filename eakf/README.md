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

4. `forward_filter_example.py`

This script extends the DA framework in `DA_forward_example.py` to allow
ingestion of observations *within* assimilation windows, rather than only at
their edges as the current implementation in `DA_forward_example.py`.
For this it assimilates data at the time of observation and runs
the updated forward model to the end of the assimilation window.
It only allows one intermediate observation within an assimilation window.
