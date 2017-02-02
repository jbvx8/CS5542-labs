

import org.apache.hadoop.io.compress.GzipCodec
import org.apache.spark.{SparkConf, SparkContext}


object SparkLab2 {

  def main(args: Array[String]) {

    System.setProperty("hadoop.home.dir","C:\\Users\\Anonymous\\Documents\\CS5542\\hadoopforspark");

    val sparkConf = new SparkConf().setAppName("SparkLab2").setMaster("local[*]")

    val sc=new SparkContext(sparkConf)

    val input=sc.textFile("input")

    val letters = input.flatMap(line => {line.toLowerCase.toList})

    val noPunctuation = letters.filter(letter => {!("" + letter).replaceAll("\\p{P}", "").equals("")})

    val noPunctuationCharacters = noPunctuation.map(letter=>(letter,1)).cache()

    val output=noPunctuationCharacters.reduceByKey(_+_)

    output.saveAsTextFile("output")

    val o=output.collect()

    var s:String="Letters:Count \n"
    o.foreach{case(letter,count)=>{

      s+=letter+" : "+count+"\n"

    }}

  output.repartition(1).saveAsTextFile("combine");

  }

}
