# sbt "run -m $master "
time spark-submit \
   --executor-memory 1G \
   --num-executors 12 \
   --class SparkLR target/scala-2.10/sparklr_2.10-0.1-SNAPSHOT.jar \
   $master
