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

package org.apache.spark.mllib.tree.impurity

import org.apache.spark.annotation.{DeveloperApi, Experimental, Since}

/**
 * :: Experimental ::
 * Class for calculating the
 * [[http://en.wikipedia.org/wiki/Decision_tree_learning#Gini_impurity Gini impurity]]
 * during binary classification.
 */
@Since("1.0.0")
@Experimental
object Gini extends Impurity {

  /**
   * Gini(D)反映了从数据集D中随机取样两个样本，
   * 其类别标记不一致的概率。因此，Gini(D)越小，
   * 则数据集D的纯度越高。
   * Gini impurity can be computed by summing the 
   * probability f_{i} of an item with label i being 
   * chosen times the probability 1-f_{i} of a mistake 
   * in categorizing that item. It reaches its minimum 
   * (zero) when all cases in the node fall into a 
   * single target category.
   *
   * To compute Gini impurity for a set of items with J 
   * classes, suppose i\in \{1,2,...,J\}, and let f_{i} 
   * be the fraction of items labeled with class i in the set.

   * gini(D) = \sum_{i=1}^J f_i(1-f_i) 
   *         = 1 - \sum_{i=1}^J f_i^2
   * 
   * :: DeveloperApi ::
   * information calculation for multiclass classification
   * @param counts Array[Double] with counts for each label
   * @param totalCount sum of counts for all labels
   * @return information value, or 0 if totalCount = 0
   */
  @Since("1.1.0")
  @DeveloperApi
  override def calculate(counts: Array[Double], totalCount: Double): Double = {
    if (totalCount == 0) {
      return 0
    }
    val numClasses = counts.length
    var impurity = 1.0
    var classIndex = 0
    while (classIndex < numClasses) {
      // 第classIndex类的样本所占的比例
      val freq = counts(classIndex) / totalCount
      impurity -= freq * freq
      classIndex += 1
    }
    impurity
  }

  /**
   * :: DeveloperApi ::
   * variance calculation
   * @param count number of instances
   * @param sum sum of labels
   * @param sumSquares summation of squares of the labels
   * @return information value, or 0 if count = 0
   */
  @Since("1.0.0")
  @DeveloperApi
  override def calculate(count: Double, sum: Double, sumSquares: Double): Double =
    throw new UnsupportedOperationException("Gini.calculate")

  /**
   * Get this impurity instance.
   * This is useful for passing impurity parameters to a Strategy in Java.
   */
  @Since("1.1.0")
  def instance: this.type = this

}

/**
 * Class for updating views of a vector of sufficient statistics,
 * in order to compute impurity from a sample.
 * Note: Instances of this class do not hold the data; they operate on views of the data.
 * @param numClasses  Number of classes for label.
 */
private[tree] class GiniAggregator(numClasses: Int)
  extends ImpurityAggregator(numClasses) with Serializable {

  
  /**
   * Get an [[ImpurityCalculator]] for a (node, feature, bin).
   * @param allStats  Flat stats array, with stats for this (node, feature, bin) contiguous.
   * @param offset    Start index of stats for this (node, feature, bin).
   */
  def getCalculator(allStats: Array[Double], offset: Int): GiniCalculator = {
    new GiniCalculator(allStats.view(offset, offset + statsSize).toArray)
  }

}

/**
 * Stores statistics for one (node, feature, bin) for calculating impurity.
 * Unlike [[GiniAggregator]], this class stores its own data and is for a specific
 * (node, feature, bin).
 * @param stats  Array of sufficient statistics for a (node, feature, bin).
 */
private[spark] class GiniCalculator(stats: Array[Double]) extends ImpurityCalculator(stats) {

  /**
   * Make a deep copy of this [[ImpurityCalculator]].
   */
  def copy: GiniCalculator = new GiniCalculator(stats.clone())

  /**
   * Calculate the impurity from the stored sufficient statistics.
   */
  def calculate(): Double = Gini.calculate(stats, stats.sum)

  /**
   * Number of data points accounted for in the sufficient statistics.
   */
  def count: Long = stats.sum.toLong

  /**
   * Prediction which should be made based on the sufficient statistics.
   */
  def predict: Double = if (count == 0) {
    0
  } else {
    indexOfLargestArrayElement(stats)
  }

  /**
   * Probability of the label given by [[predict]].
   */
  override def prob(label: Double): Double = {
    val lbl = label.toInt
    require(lbl < stats.length,
      s"GiniCalculator.prob given invalid label: $lbl (should be < ${stats.length}")
    require(lbl >= 0, "GiniImpurity does not support negative labels")
    val cnt = count
    if (cnt == 0) {
      0
    } else {
      stats(lbl) / cnt
    }
  }

  override def toString: String = s"GiniCalculator(stats = [${stats.mkString(", ")}])"

}
