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
import com.intel.analytics.bigdl.dllib.feature.dataset.ByteRecord
import java.nio.ByteBuffer
import java.nio.file.{Files, Path, Paths}
import com.intel.analytics.bigdl.dllib.utils.{File}
import com.intel.analytics.bigdl.dllib.utils.{OptimizerVersion}

object Main {
  var sparkSession = SparkSession
    .builder()
    .master("local[*]")
    .appName("dl-spark")
    .getOrCreate()

  val spark = sparkSession.sqlContext

  val trainMean = 0.13066047740239506
  val trainStd = 0.3081078

  val testMean = 0.13251460696903547
  val testStd = 0.31048024

  import spark.implicits._

  def load(featureFile: String, labelFile: String): Array[ByteRecord] = {
    val featureBuffer = ByteBuffer.wrap(Files.readAllBytes(Paths.get(featureFile)))
    val labelBuffer = ByteBuffer.wrap(Files.readAllBytes(Paths.get(labelFile)))
    
    val labelMagicNumber = labelBuffer.getInt()

    require(labelMagicNumber == 2049)
    val featureMagicNumber = featureBuffer.getInt()
    require(featureMagicNumber == 2051)

    val labelCount = labelBuffer.getInt()
    val featureCount = featureBuffer.getInt()
    require(labelCount == featureCount)

    val rowNum = featureBuffer.getInt()
    val colNum = featureBuffer.getInt()

    val result = new Array[ByteRecord](featureCount)
    var i = 0
    while (i < featureCount) {
      val img = new Array[Byte]((rowNum * colNum))
      var y = 0
      while (y < rowNum) {
        var x = 0
        while (x < colNum) {
          img(x + y * colNum) = featureBuffer.get()
          x += 1
        }
        y += 1
      }
      result(i) = ByteRecord(img, labelBuffer.get().toFloat + 1.0f)
      i += 1
    }

    result
  }

  def buildModel(inputShape: Shape): Sequential[Float] = {
    import com.intel.analytics.bigdl.numeric.NumericFloat
    val model = Sequential()
    model.add(Reshape(Array(1, 28, 28), inputShape = Shape(28, 28, 1)))
    model.add(Conv2D(32, 3, 3))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(poolSize = (2, 2)))

    model.add(Flatten())
    model.add(Dense(100))
    model.add(Activation("relu"))
    model.add(Dense(10))
    model.add(Activation("softmax"))

    model
  }

  // Top1 Accuracy: 0.9857
  // Top5 Accuracy: 1.0
  def main(args: Array[String]): Unit = {
    val sc = NNContext.initNNContext()

    val datapath = "/media/gegen07/Expansion/data/mnist"
    val trainData = datapath + "/train-images-idx3-ubyte"
    val trainLabel = datapath + "/train-labels-idx1-ubyte"
    val validationData = datapath + "/t10k-images-idx3-ubyte"
    val validationLabel = datapath + "/t10k-labels-idx1-ubyte"

    val model:Sequential[Float] = buildModel(Shape(28, 28, 1))

    val optimMethod = new SGD[Float](learningRate = 0.05, learningRateDecay = 0.0)

    val trainSet = DataSet.array(load(trainData, trainLabel), sc) ->
      BytesToGreyImg(28, 28) ->
      GreyImgNormalizer(trainMean, trainStd) ->
      GreyImgToBatch(12)

    val validationSet = DataSet.array(load(validationData, validationLabel), sc) ->
      BytesToGreyImg(28, 28) -> GreyImgNormalizer(testMean, testStd) -> GreyImgToBatch(
      128)

    model.compile(optimizer = optimMethod,
      loss = new ClassNLLCriterion[Float](),
      metrics = List(new Top1Accuracy[Float], new Top5Accuracy[Float], new Loss[Float]))

    model.fit(trainSet, nbEpoch = 15, validationData = validationSet)

    sc.stop()
  }
}