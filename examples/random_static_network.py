import os, sys; sys.path.append(os.getenv('EPIFORECAST', os.path.join("..")))

import epiforecast

model = epiforecast.StaticRiskNetworkModel(N_nodes=1000)
model.set_transition_rates(epiforecast.TransitionRates())
