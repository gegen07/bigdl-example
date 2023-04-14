package Main
import com.intel.analytics.bigdl.dllib.NNContext
import com.intel.analytics.bigdl.dllib.feature.dataset.DataSet
import com.intel.analytics.bigdl.dllib.feature.dataset.image.{BytesToGreyImg, GreyImgNormalizer, GreyImgToBatch}
import com.intel.analytics.bigdl.dllib.keras.{Model, Sequential}
import com.intel.analytics.bigdl.dllib.utils.Shape
import com.intel.analytics.bigdl.dllib.keras.layers._
import com.intel.analytics.bigdl.dllib.optim.SGD
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.dllib.nn.ClassNLLCriterion
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.dllib.optim._
import com.intel.analytics.bigdl.dllib.utils.Engine
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.{SQLContext, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.DoubleType

object POIIdentClassif {
  var sparkSession = SparkSession
    .builder()
    .master("local[*]")
    .appName("dl-spark")
    .getOrCreate()

  val spark = sparkSession.sqlContext
  val NNContext = NNContext(sparkSession)

  import spark.implicits._

  def buildModel(inputShape: Shape): Sequential[Float] = {
    import com.intel.analytics.bigdl.numeric.NumericFloat
    val model = Sequential()
    model.add(Conv2D(32, 3, 3, inputShape = inputShape))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(poolSize = (2, 2)))

    model.add(Conv2D(32, 3, 3))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(poolSize = (2, 2)))

    model.add(Conv2D(64, 3, 3))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(poolSize = (2, 2)))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation("relu"))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation("sigmoid"))

    model
  }

  def main(args: Array[String]): Unit = {
    val datapath = "/media/gegen07/Expansion/data/mnist"
    val trainData = datapath + "/train-images-idx3-ubyte"
    val trainLabel = datapath + "/train-labels-idx1-ubyte"
    val validationData = datapath + "/t10k-images-idx3-ubyte"
    val validationLabel = datapath + "/t10k-labels-idx1-ubyte"

    val model:Sequential[Float] = buildModel(Shape(28, 28, 1))

    val optimMethod = new SGD[Float](learningRate = 0.01, learningRateDecay = 0.0)

    val trainSet = DataSet.array(load(trainData, trainLabel), spark.sparkContext) ->
      BytesToGreyImg(28, 28) ->
      GreyImgNormalizer(trainMean, trainStd) ->
      GreyImgToBatch(1000)

    val validationSet = DataSet.array(load(validationData, validationLabel), sc) ->
      BytesToGreyImg(28, 28) -> GreyImgNormalizer(testMean, testStd) -> GreyImgToBatch(
      param.batchSize)

    model.

  }
}