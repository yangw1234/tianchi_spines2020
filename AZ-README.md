Run Steps:

1. please follow the README.md to generate data files

2. run `samples/generate_ndarray_data.py` to generate training and validation data as numpy.ndarray formats in `./data/ as `train.npy` and `val.npy`

3. edit the `ANALYTICS_ZOO_HOME` and `TF_LIBS` variable in `samples/tianchi/start_tfpark_local.sh` accordingly and run `samples/tianchi/start_tfpark_local.sh`