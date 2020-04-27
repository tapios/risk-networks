# risk-networks
Code for risk networks: a blend of compartmental models, graphs, data assimilation and semi-supervised learning

- Dependencies:  
  - cycler==0.10.0
  - eon==1.1
  - kiwisolver==1.2.0
  - matplotlib==3.2.1
  - networkx==2.4
  - numpy==1.18.3
  - pyparsing==2.4.7
  - pytz==2019.3
  - scipy==1.4.1


- Added conda environment `yml` to have the bare minimum of `python` modules to
  work. To replicate the environment make sure you have anaconda preinstalled
  and use the following command from within the repo directory (or specify the
  full path to the yml file):
  <!--  -->
  ```{bash}
  conda env create -f risknet.yml
  ```
