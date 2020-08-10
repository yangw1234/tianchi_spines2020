export KMP_BLOCKTIME=1
export KMP_AFFINITY=granularity=fine,compact,1,0
export OMP_NUM_THREADS=24
export MASTER=local[2]
export SPARK_DRIVER_MEMORY=100g

export ANALYTICS_ZOO_HOME=/home/cpx/yang/dist
TF_LIBS=$ANALYTICS_ZOO_HOME/lib/linux-x86_64
bash $ANALYTICS_ZOO_HOME/bin/spark-submit-python-with-zoo.sh --master $MASTER --driver-memory 80g --executor-memory 80g --driver-java-options "-Djava.library.path=${TF_LIBS}" train_spines2.py

