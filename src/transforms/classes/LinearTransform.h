/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2017-2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef OPENKALMAN_LINEARTRANSFORM_H
#define OPENKALMAN_LINEARTRANSFORM_H


namespace OpenKalman
{
  /**
   * @brief A linear transformation from one statistical distribution to another.
   */
  struct LinearTransform : internal::LinearTransformBase
  {
  protected:

    using Base = internal::LinearTransformBase;

    /// The underlying transform function model for LinearTransform.
    template<typename InputCoefficients, typename OutputCoefficients,
      typename TransformationMatrix, typename...PerturbationTransformationMatrices>
    struct TransformFunction
    {
    protected:
      using LinTrans = LinearTransformation<InputCoefficients, OutputCoefficients, TransformationMatrix,
        PerturbationTransformationMatrices...>;
      const LinTrans& transformation;

    public:
      TransformFunction(const LinTrans& t) : transformation(t) {}

      template<typename InputMean, typename ... NoiseMean>
      auto operator()(const InputMean& x, const NoiseMean& ... n) const
      {
        return std::tuple {transformation(x, n...), transformation.jacobian(x, n...)};
      }

      static constexpr bool correction = false;
    };

  public:
    /**
     * Linearly transform one statistical distribution to another.
     * @tparam InputCoefficients Coefficient types for the input.
     * @tparam OutputCoefficients Coefficient types for the output.
     * @tparam TransformationMatrix Transformation matrix. It is a native matrix type with rows corresponding to
     * OutputCoefficients and columns corresponding to InputCoefficients.
     * @tparam PerturbationTransformationMatrices Transformation matrices for each potential perturbation term.
     * if the parameter is not given, the transformation matrix is assumed to be identity (i.e., it is a translation).
     * It is a native matrix type with both rows and columns corresponding to OutputCoefficients.
     * @tparam InputDist Input distribution.
     * @tparam NoiseDist Noise distribution.
     **/
    template<
      typename InputCoefficients,
      typename OutputCoefficients,
      typename TransformationMatrix,
      typename...PerturbationTransformationMatrices,
      typename InputDist,
      typename ... NoiseDist,
      std::enable_if_t<std::conjunction_v<is_distribution<InputDist>, is_distribution<NoiseDist>...>, int> = 0>
    auto operator()(
      const LinearTransformation<InputCoefficients, OutputCoefficients, TransformationMatrix,
        PerturbationTransformationMatrices...>& transformation,
      const InputDist& in,
      const NoiseDist& ...n) const
    {
      return Base::transform(TransformFunction {transformation}, in, n...);
    }

  };


}


#endif //OPENKALMAN_LINEARTRANSFORM_H
