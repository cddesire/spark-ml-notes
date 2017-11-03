/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.mllib.feature

import java.lang.{Iterable => JavaIterable}

import scala.collection.JavaConverters._
import scala.collection.mutable

import org.apache.spark.annotation.Since
import org.apache.spark.api.java.JavaRDD
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.rdd.RDD
import org.apache.spark.util.Utils

/**
 * Maps a sequence of terms to their term frequencies using the hashing trick.
 *
 * @param numFeatures number of features (default: 2^20^)
 */
@Since("1.1.0")
class HashingTF(val numFeatures: Int) extends Serializable {

  @Since("1.1.0")
  def this() = this(1 << 20)

  /**
   * Returns the index of the input term.
   * (term.## % numFeatures) + (term.## < 0 ? numFeatures : 0)
   */
  @Since("1.1.0")
  def indexOf(term: Any): Int = Utils.nonNegativeMod(term.##, numFeatures)

  /**
   * Transforms the input document into a sparse term frequency vector.
   */
  @Since("1.1.0")
  def transform(document: Iterable[_]): Vector = {
    // 词频向量 
    val termFrequencies = mutable.HashMap.empty[Int, Double]
    document.foreach { term =>
      // 通过term找到下标
      val i = indexOf(term)
      termFrequencies.put(i, termFrequencies.getOrElse(i, 0.0) + 1.0)
    }
    // 转化为一个稀疏的向量
    Vectors.sparse(numFeatures, termFrequencies.toSeq)
  }

  /**
   * Transforms the input document into a sparse term frequency vector (Java version).
   */
  @Since("1.1.0")
  def transform(document: JavaIterable[_]): Vector = {
    transform(document.asScala)
  }

  /**
   * Transforms the input document to term frequency vectors.
   *  val documents: RDD[Seq[String]] = sc.textFile("...").map(_.split(" ").toSeq)
   * 源数据为一篇文章一行，所有文章的词频向量，一篇文章一个向量
   */
  @Since("1.1.0")
  def transform[D <: Iterable[_]](dataset: RDD[D]): RDD[Vector] = {
    dataset.map(this.transform)
  }

  /**
   * Transforms the input document to Transforms vectors (Java version).
   */
  @Since("1.1.0")
  def transform[D <: JavaIterable[_]](dataset: JavaRDD[D]): JavaRDD[Vector] = {
    dataset.rdd.map(this.transform).toJavaRDD()
  }
}
