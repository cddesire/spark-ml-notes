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

package org.apache.spark.mllib.optimization

import org.apache.spark.annotation.DeveloperApi
import org.apache.spark.mllib.linalg.{DenseVector, Vector, Vectors}
import org.apache.spark.mllib.linalg.BLAS.{axpy, dot, scal}
import org.apache.spark.mllib.util.MLUtils

/**
 * :: DeveloperApi ::
 * Class used to compute the gradient for a loss function, given a single data point.
 */
@DeveloperApi
abstract class Gradient extends Serializable {
  /**
   * Compute the gradient and loss given the features of a single data point.
   *
   * @param data features for one data point
   * @param label label for this data point
   * @param weights weights/coefficients corresponding to features
   *
   * @return (gradient: Vector, loss: Double)
   */
  def compute(data: Vector, label: Double, weights: Vector): (Vector, Double) = {
    val gradient = Vectors.zeros(weights.size)
    val loss = compute(data, label, weights, gradient)
    (gradient, loss)
  }

  /**
   * Compute the gradient and loss given the features of a single data point,
   * add the gradient to a provided vector to avoid creating new objects, and return loss.
   *
   * @param data features for one data point
   * @param label label for this data point
   * @param weights weights/coefficients corresponding to features
   * @param cumGradient the computed gradient will be added to this vector
   *
   * @return loss
   */
  def compute(data: Vector, label: Double, weights: Vector, cumGradient: Vector): Double
}


@DeveloperApi
class LogisticGradient(numClasses: Int) extends Gradient {

  def this() = this(2)

  override def compute(data: Vector, label: Double, weights: Vector): (Vector, Double) = {
    val gradient = Vectors.zeros(weights.size)
    val loss = compute(data, label, weights, gradient)
    (gradient, loss)
  }

  override def compute(
                              data: Vector,
                              label: Double,
                              weights: Vector,
                              cumGradient: Vector): Double = {
    val dataSize = data.size

    // (weights.size / dataSize + 1) is number of classes
    require(weights.size % dataSize == 0 && numClasses == weights.size / dataSize + 1)
    numClasses match {
      case 2 =>

        /**
          * For Binary Logistic Regression.
          *
          * Although the loss and gradient calculation for multinomial one is more generalized,
          * and multinomial one can also be used in binary case, we still implement a specialized
          * binary version for performance reason.
          */
        val margin = -1.0 * dot(data, weights)
        val multiplier = (1.0 / (1.0 + math.exp(margin))) - label
        // cumGradient += multiplier * data
        axpy(multiplier, data, cumGradient)
        if (label > 0) {
          // The following is equivalent to log(1 + exp(margin)) but more numerically stable.
          MLUtils.log1pExp(margin)
        } else {
          MLUtils.log1pExp(margin) - margin
        }
      case _ =>

        /**
          * For Multinomial Logistic Regression.
          */
        val weightsArray = weights match {
          case dv: DenseVector => dv.values
          case _ =>
            throw new IllegalArgumentException(
              s"weights only supports dense vector but got type ${weights.getClass}.")
        }
        val cumGradientArray = cumGradient match {
          case dv: DenseVector => dv.values
          case _ =>
            throw new IllegalArgumentException(
              s"cumGradient only supports dense vector but got type ${cumGradient.getClass}.")
        }

        // marginY is margins(label - 1) in the formula.
        var marginY = 0.0
        var maxMargin = Double.NegativeInfinity
        var maxMarginIndex = 0

        val margins = Array.tabulate(numClasses - 1) { i =>
          var margin = 0.0
          data.foreachActive { (index, value) =>
            if (value != 0.0) margin += value * weightsArray((i * dataSize) + index)
          }
          if (i == label.toInt - 1) marginY = margin
          if (margin > maxMargin) {
            maxMargin = margin
            maxMarginIndex = i
          }
          margin
        }

        /**
          * When maxMargin > 0, the original formula will cause overflow as we discuss
          * in the previous comment.
          * We address this by subtracting maxMargin from all the margins, so it's guaranteed
          * that all of the new margins will be smaller than zero to prevent arithmetic overflow.
          */
        val sum = {
          var temp = 0.0
          if (maxMargin > 0) {
            for (i <- 0 until numClasses - 1) {
              margins(i) -= maxMargin
              if (i == maxMarginIndex) {
                temp += math.exp(-maxMargin)
              } else {
                temp += math.exp(margins(i))
              }
            }
          } else {
            for (i <- 0 until numClasses - 1) {
              temp += math.exp(margins(i))
            }
          }
          temp
        }

        for (i <- 0 until numClasses - 1) {
          val multiplier = math.exp(margins(i)) / (sum + 1.0) - {
            if (label != 0.0 && label == i + 1) 1.0 else 0.0
          }
          data.foreachActive { (index, value) =>
            if (value != 0.0) cumGradientArray(i * dataSize + index) += multiplier * value
          }
        }

        val loss = if (label > 0.0) math.log1p(sum) - marginY else math.log1p(sum)

        if (maxMargin > 0) {
          loss + maxMargin
        } else {
          loss
        }
    }
  }
}

/**
 * :: DeveloperApi ::
 * Compute gradient and loss for a Least-squared loss function, as used in linear regression.
 * This is correct for the averaged least squares loss function (mean squared error)
 *              L = 1/2n ||A weights-y||^2
 * See also the documentation for the precise formulation.
 */
@DeveloperApi
class LeastSquaresGradient extends Gradient {
  override def compute(data: Vector, label: Double, weights: Vector): (Vector, Double) = {
    val diff = dot(data, weights) - label
    val loss = diff * diff / 2.0
    val gradient = data.copy
    scal(diff, gradient)
    (gradient, loss)
  }

  override def compute(
      data: Vector,
      label: Double,
      weights: Vector,
      cumGradient: Vector): Double = {
    val diff = dot(data, weights) - label
    axpy(diff, data, cumGradient)
    diff * diff / 2.0
  }
}

/**
 * :: DeveloperApi ::
 * Compute gradient and loss for a Hinge loss function, as used in SVM binary classification.
 * See also the documentation for the precise formulation.
 * NOTE: This assumes that the labels are {0,1}
 */
@DeveloperApi
class HingeGradient extends Gradient {
  override def compute(data: Vector, label: Double, weights: Vector): (Vector, Double) = {
    val dotProduct = dot(data, weights)
    // Our loss function with {0, 1} labels is max(0, 1 - (2y - 1) (f_w(x)))
    // Therefore the gradient is -(2y - 1)*x
    val labelScaled = 2 * label - 1.0
    if (1.0 > labelScaled * dotProduct) {
      val gradient = data.copy
      scal(-labelScaled, gradient)
      (gradient, 1.0 - labelScaled * dotProduct)
    } else {
      (Vectors.sparse(weights.size, Array.empty, Array.empty), 0.0)
    }
  }

  override def compute(
      data: Vector,
      label: Double,
      weights: Vector,
      cumGradient: Vector): Double = {
    val dotProduct = dot(data, weights)
    // Our loss function with {0, 1} labels is max(0, 1 - (2y - 1) (f_w(x)))
    // Therefore the gradient is -(2y - 1)*x
    val labelScaled = 2 * label - 1.0
    if (1.0 > labelScaled * dotProduct) {
      axpy(-labelScaled, data, cumGradient)
      1.0 - labelScaled * dotProduct
    } else {
      0.0
    }
  }
}
