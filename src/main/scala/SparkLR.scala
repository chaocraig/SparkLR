

import org.apache.spark.mllib.regression.LinearRegressionWithSGD
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors



import org.apache.spark._
import org.apache.spark.SparkContext._
import org.apache.spark.rdd.{RDD => SparkRDD}

import org.apache.spark.util.Utils
import org.apache.hadoop.fs.FileUtil

import org.apache.commons.io.FileUtils;
import org.apache.commons.io.filefilter.WildcardFileFilter;

import org.apache.commons.io._

import sys.process._

import java.io.File;


object SparkLR {
  
   //private val RUN_JAR="/home/cray/ScalaParseDate/target/scala-2.10/scalaparsedate_2.10-0.1-SNAPSHOT.jar"
   private val RUN_JAR="/home/cray/SparkLR/target/scala-2.10/sparkals_2.10-0.1-SNAPSHOT.jar"
   //private val Out_path = "/Users/cray/Documents/workspace-scala/SparkLR/data/output.data"
   private val Out_path = "/user/cray/SparkLR/output"
   private val Out_Hadoop_Path = "hdfs://hadoop-013:9000/user/cray/SparkLR/output"
   //private val In_path  = "hdfs://hadoop-013:9000/user/cray/SparkLR/test.data"
   private val In_path  = "/home/cray/SparkLR/data/lpsa.data"
   //private val In_path  = "/home/cray/SparkLR/data/tr.fm"
 
   
  def setSparkEnv(master:String) : SparkContext = {

    val conf = new SparkConf()
       //.setMaster("spark://craigmbp:7077")
       .setMaster(master)
       .setAppName("SparkLR")
       // runtime Spark Home, set by env SPARK_HOME or explicitly as below
       //.setSparkHome("/opt/spark")

       // be nice or nasty to others (per node)
       //.set("spark.executor.memory", "1g")
       //.set("spark.core.max", "2")

       // find a random port for driver application web-ui
       //.set("spark.ui.port", findAvailablePort.toString)
       //.setJars(findJars)
       //.setJars(Seq("/Users/cray/Documents/workspace-scala/ScalaParseDate/target/scala-2.10/scalaparsedate_2.10-1.0.jar"))
       //.setJars(Seq(RUN_JAR))
    
       // The coarse-grained mode will instead launch only one long-running Spark task on each Mesos machine,
       // and dynamically schedule its own “mini-tasks” within it. The benefit is much lower startup overhead,
       // but at the cost of reserving the Mesos resources for the complete duration of the application.
       // .set("spark.mesos.coarse", "true")

    // for debug purpose
    println("sparkconf: " + conf.toDebugString)

    val sc = new SparkContext(conf)
    sc
  }


  
  def ExecLR(sc:SparkContext) = {


    // Load and parse the data
    val data = sc.textFile(In_path)
    //test.data


    //val parsedData = data.map(_.concat(",n").split(',') match {
    val parsedData = data.map(_.split(',') match {

        case part: Array[String]  =>
           // println("=====>" + part.toList.mkString + "<--------" )
           LabeledPoint(part(0).toDouble, Vectors.dense(part(1).split(' ').map(_.toDouble)))
        case _ => {
            throw new Exception("-->Match error...")
        }
    })


    // Building the model
    val numIterations = 100
    val model = LinearRegressionWithSGD.train(parsedData, numIterations)

    // Evaluate model on training examples and compute training error
    val valuesAndPreds = parsedData.map { point =>
        val prediction = model.predict(point.features)
        (point.label, prediction)
    }

    val MSE = valuesAndPreds.map{case(v, p) => math.pow((v - p), 2)}.mean()
    println("\n\n\ntraining Mean Squared Error = " + MSE + "\n\n") 


  }
  
  
  def main(args: Array[String]) {

    val sc = setSparkEnv( args.toList(0) )

    ExecLR(sc)
    
    sc.stop()


  }
 

  
}
