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

#include "transforms/transformations/LinearTransformation.h"
#include "distributions/GaussianDistribution.h"
#include "transforms/support/LinearTransformBase.h"

namespace OpenKalman
{
  /**
   * @brief A linear transformation from one statistical distribution to another.
   */
  template<
    /// The linear transformation on which the transform is based.
    typename LinearTransformationType>
  struct LinearTransform;


  namespace detail
  {
    template<typename LinearTransformation>
    struct LinearTransformFunction
    {
      const LinearTransformation transformation;

      explicit LinearTransformFunction(const LinearTransformation& trans) : transformation(trans) {}

      explicit LinearTransformFunction(LinearTransformation&& trans) noexcept : transformation(std::move(trans)) {}

      template<typename InputMean, typename ... NoiseMean>
      auto operator()(const InputMean& x, const NoiseMean& ... n) const
      {
        return std::tuple {transformation(x, n...), transformation.jacobian(x, n...)};
      }

      static constexpr bool correction = false;
    };
  }


  template<typename LinearTransformationType>
  struct LinearTransform
    : internal::LinearTransformBase<
      typename LinearTransformationType::InputCoefficients,
      typename LinearTransformationType::OutputCoefficients,
      detail::LinearTransformFunction<LinearTransformationType>>
  {
    using InputCoefficients = typename LinearTransformationType::InputCoefficients;
    using OutputCoefficients = typename LinearTransformationType::OutputCoefficients;
    using Function = detail::LinearTransformFunction<LinearTransformationType>;
    using Base = internal::LinearTransformBase<InputCoefficients, OutputCoefficients, Function>;

    explicit LinearTransform(const LinearTransformationType& transformation)
      : Base(Function(transformation)) {}

    explicit LinearTransform(LinearTransformationType&& transformation) noexcept
      : Base(Function(std::move(transformation))) {}

    template<typename T, typename ... Noise,
      std::enable_if_t<std::conjunction_v<is_typed_matrix<T>, is_typed_matrix<Noise>...>, int> = 0>
    explicit LinearTransform(T&& t, Noise&&...n)
      : Base(Function(LinearTransformationType(std::forward<T>(t), std::forward<Noise>(n)...))) {}

    template<typename T, typename ... Noise,
      std::enable_if_t<std::conjunction_v<is_typed_matrix_base<T>, is_typed_matrix_base<Noise>...>, int> = 0>
    explicit LinearTransform(T&& t, Noise&&...n)
      : Base(Function(LinearTransformationType(std::forward<T>(t), std::forward<Noise>(n)...))) {}

  };


  ////////////////////////
  //  Deduction guides  //
  ////////////////////////

  template<typename T, typename ... Noise,
    std::enable_if_t<std::conjunction_v<is_typed_matrix<T>, is_typed_matrix<Noise>...>, int> = 0>
  LinearTransform(T&&, Noise&& ...)
  -> LinearTransform<LinearTransformation<
    typename MatrixTraits<T>::RowCoefficients,
    typename MatrixTraits<T>::ColumnCoefficients,
    typename MatrixTraits<T>::BaseMatrix,
    typename MatrixTraits<Noise>::BaseMatrix...>>;

  template<typename T, typename ... Noise,
    std::enable_if_t<std::conjunction_v<is_typed_matrix_base<T>, is_typed_matrix_base<Noise>...>, int> = 0>
  LinearTransform(T&&, Noise&& ...)
  -> LinearTransform<LinearTransformation<
    Axes<MatrixTraits<T>::columns>,
    Axes<MatrixTraits<T>::dimension>,
    std::decay_t<T>,
    std::decay_t<Noise>...>>;

}


#endif //OPENKALMAN_LINEARTRANSFORM_H
