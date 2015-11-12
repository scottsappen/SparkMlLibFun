package com.cloudera.csm.sparkmllib

/**
  * Created by scotts on 11/12/15.
  */

import org.apache.spark.mllib.classification.{SVMWithSGD, NaiveBayes}
import org.apache.spark.mllib.clustering.KMeans
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.{SparkContext, SparkConf}

object SparkMlLibFun {
  def main(args: Array[String]): Unit = {

    if (args.length == 0) {
      println("running local")

    }

    val runLocal = args.length == 0 || args(0).equals("L")

    val sc = if (runLocal) {
      println("Running Local")
      val sparkConfig = new SparkConf()
      sparkConfig.set("spark.broadcast.compress", "false")
      sparkConfig.set("spark.shuffle.compress", "false")
      sparkConfig.set("spark.shuffle.spill.compress", "false")
      sparkConfig.set("spark.io.compression.codec", "lzf")
      new SparkContext("local[4]", "SparkSQL on Kudu", sparkConfig)
    } else {
      println("Running Cluster")
      new SparkContext(new SparkConf())
    }

    //type of user, userID, current avg, 3 month avg, year avg
    val startingRDD = sc.parallelize(Array(
      "1,101,100,110,120,500,200",
      "2,102,100,60,140,500,200",
      "3,103,100,90,80,500,200",
      "4,104,100,110,70,500,200",
      "1,101,100,116,120,500,200",
      "2,102,100,60,140,500,200",
      "3,103,100,90,80,500,200",
      "4,104,100,110,70,500,200",
      "1,101,100,110,120,500,200",
      "2,102,100,60,140,500,200",
      "3,103,100,90,80,500,200",
      "4,104,100,110,70,500,200",
      "1,101,100,110,120,500,200",
      "2,102,100,60,140,500,200",
      "3,103,100,90,80,500,200",
      "4,104,100,110,70,500,200",
      "1,101,100,110,120,500,200",
      "2,102,100,60,140,500,200",
      "3,103,100,96,80,500,200",
      "4,104,100,110,70,500,200",
      "1,101,100,110,120,500,200",
      "2,102,100,60,140,500,200",
      "3,103,100,90,80,500,200",
      "4,104,100,110,70,500,200",
      "1,101,100,110,120,500,200",
      "2,102,100,60,140,500,200",
      "3,103,100,90,80,500,200",
      "4,104,100,110,70,500,200",
      "1,101,100,110,120,500,200",
      "2,102,100,60,140,500,200",
      "3,103,100,90,80,500,200",
      "4,104,100,110,70,500,200",
      "1,101,100,110,120,500,200",
      "2,102,100,60,140,500,200",
      "3,103,100,90,80,500,200",
      "4,104,100,115,70,500,200",
      "1,101,100,110,120,500,200",
      "2,102,100,60,140,500,200",
      "3,103,100,90,80,500,200",
      "4,104,100,110,70,500,200"
    ))

    //---- KMeans
    val vectorsRDD = startingRDD.map(r => {
      val cells = r.split(",")
      val array = Array(cells(2).toDouble, cells(3).toDouble, cells(4).toDouble, cells(5).toDouble, cells(6).toDouble)
      Vectors.dense(array)
    })

    val clusters = KMeans.train(vectorsRDD, 3, 5)
    clusters.clusterCenters.foreach(v => println(" Vector Center:" + v))

    //---- NaiveBayes
    val labeledPointRDD = startingRDD.map { r =>
      val cells = r.split(",")
      val array = Array(cells(2).toDouble, cells(3).toDouble, cells(4).toDouble, cells(5).toDouble, cells(6).toDouble)
      LabeledPoint(cells(0).toDouble, Vectors.dense(array))
    }

    val splits = labeledPointRDD.randomSplit(Array(0.6, 0.4), seed = 11L)
    val training = splits(0)
    val test = splits(1)

    val model = NaiveBayes.train(training, lambda = 1.0, modelType = "multinomial")

    val predictionAndLabel = test.map(p => {
      println("p:" + p + " (" + model.predict(p.features) + ")")

      (model.predict(p.features), p.label)

    })
    val accuracy = 1.0 * predictionAndLabel.filter(x => x._1 == x._2).count() / test.count()

    println("predictionAndLabel:")
    predictionAndLabel.foreach(r => println(" " + r))
    println("accuracy: " + accuracy)
    println("model: " + model.toString)

    //Random Forest
    val numClasses = 5
    val categoricalFeaturesInfo = Map[Int, Int]()
    val numTrees = 3 // Use more in practice.
    val featureSubsetStrategy = "auto" // Let the algorithm choose.
    val impurity = "gini"
    val maxDepth = 4
    val maxBins = 32

    val model2 = RandomForest.trainClassifier(training, numClasses, categoricalFeaturesInfo,
      numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins)

    val labelAndPreds2 = test.map { point =>
      val prediction = model.predict(point.features)
      (point.label, prediction)
    }

    val testErr = labelAndPreds2.filter(r => r._1 != r._2).count.toDouble / test.count()
    println("Test Error = " + testErr)
    println("Learned classification forest model:\n" + model2.toDebugString)

    //Principal component analysis

    val mat = new RowMatrix(vectorsRDD)

    val pc = mat.computePrincipalComponents(3)
    val projected = mat.multiply(pc)

    projected.rows.foreach( r => {
      println(r)
    })


    //Linear Support Vector Machines

    val binaryTraining = training.map(r => {
      val newLabel = (r.label.toInt % 2).toDouble
      LabeledPoint(newLabel, r.features)
    })

    val binaryTesting = test.map(r => {
      val newLabel = (r.label.toInt % 2).toDouble
      LabeledPoint(newLabel, r.features)
    })

    val numIterations = 100
    val model3 = SVMWithSGD.train(binaryTraining, numIterations)

    // Clear the default threshold.
    model3.clearThreshold()

    // Compute raw scores on the test set.
    val scoreAndLabels = binaryTesting.map { point =>
      val score = model.predict(point.features)
      (score, point.label)
    }

    // Get evaluation metrics.
    val metrics = new BinaryClassificationMetrics(scoreAndLabels)
    val auROC = metrics.areaUnderROC()

    println("Area under ROC = " + auROC)

  }

}
