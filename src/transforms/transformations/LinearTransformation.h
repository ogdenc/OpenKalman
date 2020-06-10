/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2018-2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef OPENKALMAN_LINEARTRANSFORMATION_H
#define OPENKALMAN_LINEARTRANSFORMATION_H

#include "Transformation.h"
#include "variables/classes/tests-typedmatrix.h"
#include "variables/support/ArrayUtils.h"


namespace OpenKalman
{

  /**
   * @brief A linear transformation between two Mean vectors.
   *
   * The linear transformation of one Mean to another, by matrix multiplication of Mean vectors
   * by a TypedMatrix. This is generally only useful for Mean vectors in Euclidean space.
   *
   * @tparam InputCoefficients Coefficient types for the input.
   * @tparam OutputCoefficients Coefficient types for the output.
   * @tparam TransformationMatrix Transformation matrix. It has regular matrix type with rows corresponding to
   * OutputCoefficients and columns corresponding to InputCoefficients.
   * @tparam NoiseTransformationMatrices Transformation matrices for each potential noise term (usually none or just
   * one). It has regular matrix type with both rows and columns corresponding to OutputCoefficients.
   */
  template<
    typename InputCoefficients,
    typename OutputCoefficients,
    typename TransformationMatrix,
    typename ... NoiseTransformationMatrices>
  struct LinearTransformation;


  namespace internal
  {
    namespace detail
    {
      template<std::size_t begin, typename T, std::size_t... I>
      constexpr auto tuple_slice_impl(T&& t, std::index_sequence<I...>)
      {
        return std::forward_as_tuple(std::get<begin + I>(std::forward<T>(t))...);
      }
    }

    /// Return a subset of a tuple, given an index range.
    template<std::size_t index1, std::size_t index2, typename T>
    constexpr auto tuple_slice(T&& t)
    {
      static_assert(index1 <= index2, "Index range is invalid");
      static_assert(index2 <= std::tuple_size_v<std::decay_t<T>>, "Index is out of bounds");
      return detail::tuple_slice_impl<index1>(std::forward<T>(t), std::make_index_sequence<index2 - index1>());
    }


    /// Create a tuple that replicates a value.
    template<std::size_t N, typename T>
    constexpr auto tuple_replicate(const T& t)
    {
      if constexpr(N < 1)
      {
        return std::tuple {};
      }
      else
      {
        return std::tuple_cat(std::make_tuple(t), tuple_replicate<N - 1>(t));
      }
    }
  }


  template<
    typename InputCoeffs,
    typename OutputCoeffs,
    typename TransformationMatrix,
    typename ... NoiseTransformationMatrices>
  struct LinearTransformation
  {
    using InputCoefficients = InputCoeffs;
    using OutputCoefficients = OutputCoeffs;
    static_assert(MatrixTraits<TransformationMatrix>::dimension == OutputCoefficients::size);
    static_assert(MatrixTraits<TransformationMatrix>::columns == InputCoefficients::size);
    static_assert(((MatrixTraits<NoiseTransformationMatrices>::dimension == OutputCoefficients::size) and ...));
    static_assert(((MatrixTraits<NoiseTransformationMatrices>::columns == OutputCoefficients::size) and ...));

    LinearTransformation(const TransformationMatrix& mat, const NoiseTransformationMatrices& ... noise_mat)
      : transformation_matrices(mat, noise_mat...) {}

    template<typename T, typename ... Ns,
      std::enable_if_t<std::conjunction_v<is_typed_matrix<T>, is_typed_matrix<Ns>...>, int> = 0>
    LinearTransformation(const T& mat, const Ns& ... noise_mat)
      : transformation_matrices(mat, noise_mat...)
    {
      static_assert(OpenKalman::is_equivalent_v<typename MatrixTraits<T>::RowCoefficients, OutputCoefficients>);
      static_assert(OpenKalman::is_equivalent_v<typename MatrixTraits<T>::ColumnCoefficients, InputCoefficients>);
      static_assert(((OpenKalman::is_equivalent_v<typename MatrixTraits<Ns>::RowCoefficients, OutputCoefficients>) and ...));
      static_assert(((OpenKalman::is_equivalent_v<typename MatrixTraits<Ns>::ColumnCoefficients, OutputCoefficients>) and ...));
    }

    template<typename T, typename ... Ns,
      std::enable_if_t<std::conjunction_v<is_typed_matrix_base<T>, is_typed_matrix_base<Ns>...>, int> = 0>
    LinearTransformation(const T& mat, const Ns& ... noise_mat)
      : transformation_matrices(make_Matrix<OutputCoefficients, InputCoefficients>(mat),
        make_Matrix<OutputCoefficients, InputCoefficients>(noise_mat)...)
    {
      static_assert(MatrixTraits<T>::dimension == OutputCoefficients::size);
      static_assert(MatrixTraits<T>::columns == InputCoefficients::size);
      static_assert(((MatrixTraits<Ns>::dimension == OutputCoefficients::size) and ...));
      static_assert(((MatrixTraits<Ns>::columns == OutputCoefficients::size) and ...));
    }

  protected:
    const std::tuple<TypedMatrix<OutputCoefficients, InputCoefficients, TransformationMatrix>,
      TypedMatrix<OutputCoefficients, OutputCoefficients, NoiseTransformationMatrices>...> transformation_matrices;

  private:
    template<typename InputTuple, std::size_t...ints>
    constexpr auto sumprod(InputTuple&& inputs, std::index_sequence<ints...>) const
    {
      return strict(((std::get<ints>(transformation_matrices) * std::get<ints>(std::forward<InputTuple>(inputs))) + ...));
    }

  public:
    template<
      typename In, typename ... Noise,
      std::enable_if_t<std::conjunction_v<is_typed_matrix<In>, is_typed_matrix<Noise>...>, int> = 0>
    auto operator()(In&& in, Noise&& ... noise) const
    {
      static_assert(is_column_vector_v<In>);
      static_assert(sizeof...(Noise) <= sizeof...(NoiseTransformationMatrices));
      static_assert(OpenKalman::is_equivalent_v<typename MatrixTraits<In>::RowCoefficients, InputCoefficients>);
      static_assert(std::conjunction_v<std::is_same<typename NoiseTraits<Noise>::RowCoefficients, OutputCoefficients>...>);
      auto ret = sumprod(
        std::forward_as_tuple(std::forward<In>(in), get_noise(std::forward<Noise>(noise))...),
        std::make_index_sequence<std::min(sizeof...(NoiseTransformationMatrices), sizeof...(Noise)) + 1>{});
      static_assert(OpenKalman::is_equivalent_v<typename MatrixTraits<decltype(ret)>::RowCoefficients, OutputCoefficients>);
      return ret;
    }

    /// The Jacobians corresponding to the input and all noise matrices.
    /// Returns a tuple of the transformation matrices.
    template<typename In, typename ... Noise,
      std::enable_if_t<is_typed_matrix_v<In>, int> = 0, typename = std::void_t<typename NoiseTraits<Noise>::type...>>
    auto jacobian(In&&, Noise&&...) const
    {
      static_assert(is_column_vector_v<In>);
      static_assert(sizeof...(Noise) <= sizeof...(NoiseTransformationMatrices));
      static_assert(OpenKalman::is_equivalent_v<typename MatrixTraits<In>::RowCoefficients, InputCoefficients>);
      static_assert(std::conjunction_v<std::is_same<typename NoiseTraits<Noise>::RowCoefficients, OutputCoefficients>...>);
      return internal::tuple_slice<0, sizeof...(Noise) + 1>(transformation_matrices);
    }

    /// The Hessian matrices corresponding to the input and all noise matrices.
    /// Returns a tuple of arrays of Hessian matrices. In this case, they are all zero.
    template<typename In, typename ... Noise,
      std::enable_if_t<std::conjunction_v<is_typed_matrix<In>, is_typed_matrix<Noise>...>, int> = 0>
    auto hessian(In&&, Noise&&...) const
    {
      static_assert(is_column_vector_v<In>);
      static_assert(sizeof...(Noise) <= sizeof...(NoiseTransformationMatrices));
      static_assert(OpenKalman::is_equivalent_v<typename MatrixTraits<In>::RowCoefficients, InputCoefficients>);
      static_assert(std::conjunction_v<std::is_same<typename NoiseTraits<Noise>::RowCoefficients, OutputCoefficients>...>);
      constexpr std::size_t input_size = MatrixTraits<In>::columns;
      constexpr std::size_t output_size = MatrixTraits<In>::dimension;
      using HessianMatrixIn = decltype(make_Matrix<InputCoefficients, InputCoefficients, typename MatrixTraits<In>::BaseMatrix>());
      using HessianArrayIn = std::array<HessianMatrixIn, output_size>;
      HessianArrayIn a;
      a.fill(HessianMatrixIn::zero());
      if constexpr (sizeof...(Noise) >= 1)
      {
        using HessianMatrixNoise = decltype(make_Matrix<OutputCoefficients, InputCoefficients, typename MatrixTraits<In>::BaseMatrix>());
        using HessianArrayNoise = std::array<HessianMatrixNoise, output_size>;
        HessianArrayNoise an;
        an.fill(HessianMatrixNoise::zero());
        return std::tuple_cat(std::tuple(std::move(a)), internal::tuple_replicate<sizeof...(Noise)>(std::move(an)));
      }
      else
      {
        return std::tuple(std::move(a));
      }
    }
  };


  /**
   * Deduction guides
   */

  template<typename T, typename ... Noise,
    std::enable_if_t<std::conjunction_v<is_typed_matrix<T>, is_typed_matrix<Noise>...>, int> = 0>
  LinearTransformation(T&&, Noise&& ...)
  -> LinearTransformation<
    typename MatrixTraits<T>::RowCoefficients,
    typename MatrixTraits<T>::ColumnCoefficients,
    typename MatrixTraits<T>::BaseMatrix,
    typename MatrixTraits<Noise>::BaseMatrix...>;

  template<typename T, typename ... Noise,
    std::enable_if_t<std::conjunction_v<is_typed_matrix_base<T>, is_typed_matrix_base<Noise>...>, int> = 0>
  LinearTransformation(T&&, Noise&& ...)
  -> LinearTransformation<
    Axes<MatrixTraits<T>::columns>,
    Axes<MatrixTraits<T>::dimension>,
    std::decay_t<T>,
    std::decay_t<Noise>...>;

}


#endif //OPENKALMAN_LINEARTRANSFORMATION_H
