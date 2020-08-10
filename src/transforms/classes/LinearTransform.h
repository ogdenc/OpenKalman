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
  template<
    typename InputCoefficients,
    typename OutputCoefficients,
    typename TransformationMatrix,
    typename ... PerturbationTransformationMatrices>
  struct LinearTransform;


  namespace detail
  {
  template<typename LinearTransformationType>
    struct LinearTransformFunction
    {
      const LinearTransformationType transformation;

      //explicit LinearTransformFunction(const LinearTransformationType& trans) : transformation(trans) {}

      //explicit LinearTransformFunction(LinearTransformationType&& trans) noexcept : transformation(std::move(trans)) {}

      template<typename InputMean, typename ... NoiseMean>
      auto operator()(const InputMean& x, const NoiseMean& ... n) const
      {
        return std::tuple {transformation(x, n...), transformation.jacobian(x, n...)};
      }

      static constexpr bool correction = false;
    };
  }


  template<
    typename InputCoefficients,
    typename OutputCoefficients,
    typename TransformationMatrix,
    typename ... PerturbationTransformationMatrices>
  struct LinearTransform
    : internal::LinearTransformBase<
      InputCoefficients,
      OutputCoefficients,
      detail::LinearTransformFunction<LinearTransformation<InputCoefficients, OutputCoefficients,
        TransformationMatrix, PerturbationTransformationMatrices...>>>
  {
    using LinearTransformationType = LinearTransformation<InputCoefficients, OutputCoefficients,
      TransformationMatrix, PerturbationTransformationMatrices...>;
    using TransformFunction = detail::LinearTransformFunction<LinearTransformationType>;
    using Base = internal::LinearTransformBase<InputCoefficients, OutputCoefficients, TransformFunction>;

    static_assert(is_typed_matrix_base_v<TransformationMatrix>);
    static_assert(std::conjunction_v<is_typed_matrix_base<PerturbationTransformationMatrices>...>);
    static_assert(MatrixTraits<TransformationMatrix>::dimension == OutputCoefficients::size);
    static_assert(MatrixTraits<TransformationMatrix>::columns == InputCoefficients::size);
    static_assert(((MatrixTraits<PerturbationTransformationMatrices>::dimension == OutputCoefficients::size) and ...));
    static_assert(((MatrixTraits<PerturbationTransformationMatrices>::columns == OutputCoefficients::size) and ...));

    explicit LinearTransform(const LinearTransformationType& transformation)
      : Base(TransformFunction {transformation}) {}

    explicit LinearTransform(LinearTransformationType&& transformation) noexcept
      : Base(TransformFunction {std::move(transformation)}) {}

    template<typename T, typename ... Ps,
      std::enable_if_t<std::conjunction_v<std::disjunction<is_typed_matrix<T>, is_typed_matrix_base<T>>,
        std::disjunction<is_typed_matrix<Ps>, is_typed_matrix_base<Ps>>...>, int> = 0>
    LinearTransform(T&& t, Ps&&...n)
      : LinearTransform(LinearTransformationType(std::forward<T>(t), std::forward<Ps>(n)...))
    {
      static_assert(MatrixTraits<T>::dimension == OutputCoefficients::size);
      static_assert(MatrixTraits<T>::columns == InputCoefficients::size);
      static_assert(((MatrixTraits<Ps>::dimension == OutputCoefficients::size) and ...));
      static_assert(((MatrixTraits<Ps>::columns == OutputCoefficients::size) and ...));
    }

    LinearTransform(TransformationMatrix&& t, PerturbationTransformationMatrices&&...n)
      : LinearTransform(LinearTransformationType(std::forward<TransformationMatrix>(t),
        std::forward<PerturbationTransformationMatrices>(n)...)) {}

  };


  ////////////////////////
  //  Deduction guides  //
  ////////////////////////

  template<typename T, typename ... Ps,
    std::enable_if_t<std::conjunction_v<is_typed_matrix<T>, is_typed_matrix<Ps>...>, int> = 0>
  LinearTransform(T&&, Ps&& ...)
  -> LinearTransform<
    typename MatrixTraits<T>::RowCoefficients,
    typename MatrixTraits<T>::ColumnCoefficients,
    strict_t<typename MatrixTraits<T>::BaseMatrix>,
    strict_t<typename MatrixTraits<Ps>::BaseMatrix>...>;

  template<typename T, typename ... Ps,
    std::enable_if_t<std::conjunction_v<is_typed_matrix_base<T>, is_typed_matrix_base<Ps>...>, int> = 0>
  LinearTransform(T&&, Ps&& ...)
  -> LinearTransform<
    Axes<MatrixTraits<T>::columns>,
    Axes<MatrixTraits<T>::dimension>,
    strict_t<std::decay_t<T>>,
    strict_t<std::decay_t<Ps>>...>;

}


#endif //OPENKALMAN_LINEARTRANSFORM_H
