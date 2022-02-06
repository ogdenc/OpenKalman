/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2021 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef OPENKALMAN_TYPED_MATRIX_ARITHMETIC_HPP
#define OPENKALMAN_TYPED_MATRIX_ARITHMETIC_HPP


namespace OpenKalman
{
  /// Add two typed matrices. If the operands are of different types, the result will be a regular typed matrix.
#ifdef __cpp_concepts
  template<typed_matrix V1, typed_matrix V2>
#else
  template<typename V1, typename V2, std::enable_if_t<typed_matrix<V1> and typed_matrix<V2>, int> = 0>
#endif
  inline auto operator+(V1&& v1, V2&& v2)
  {
    using RC1 = typename MatrixTraits<V1>::RowCoefficients;
    using CC1 = typename MatrixTraits<V1>::ColumnCoefficients;
    static_assert(equivalent_to<typename MatrixTraits<V2>::RowCoefficients, RC1>);
    static_assert(equivalent_to<typename MatrixTraits<V2>::ColumnCoefficients, CC1>);
    static_assert(euclidean_transformed<V1> == euclidean_transformed<V2>);
    using CommonV = std::decay_t<std::conditional_t<
      (euclidean_mean<V1> and euclidean_mean<V2>) or (mean<V1> and mean<V2>), V1, decltype(Matrix {v1})>>;
    auto b = nested_matrix(std::forward<V1>(v1)) + nested_matrix(std::forward<V2>(v2));
    return MatrixTraits<CommonV>::make(make_self_contained<V1, V2>(std::move(b)));
  }


  /// Subtract two typed matrices. The result is a regular typed matrix unless both operands are EuclideanMean.
#ifdef __cpp_concepts
  template<typed_matrix V1, typed_matrix V2>
#else
  template<typename V1, typename V2, std::enable_if_t<typed_matrix<V1> and typed_matrix<V2>, int> = 0>
#endif
  inline auto operator-(V1&& v1, V2&& v2)
  {
    using RC1 = typename MatrixTraits<V1>::RowCoefficients;
    using CC1 = typename MatrixTraits<V1>::ColumnCoefficients;
    static_assert(equivalent_to<typename MatrixTraits<V2>::RowCoefficients, RC1>);
    static_assert(equivalent_to<typename MatrixTraits<V2>::ColumnCoefficients, CC1>);
    static_assert(euclidean_transformed<V1> == euclidean_transformed<V2>);
    using CommonV = std::decay_t<std::conditional_t<
      (euclidean_mean<V1> and euclidean_mean<V2>), V1, decltype(Matrix {v1})>>;
    auto b = nested_matrix(std::forward<V1>(v1)) - nested_matrix(std::forward<V2>(v2));
    if constexpr (mean<V1> and mean<V2>)
    {
      // WC is the difference type for the coefficients. However, the result should retain coefficient types RC1.
      using WC = typename RC1::difference_type;
      return MatrixTraits<CommonV>::make(make_self_contained<V1, V2>(wrap_angles<WC>(std::move(b))));
    }
    else
    {
      return MatrixTraits<CommonV>::make(make_self_contained<V1, V2>(std::move(b)));
    }
  }


  /// Multiply a typed matrix by a scalar. The result type is the same as the operand type, so angles in the result may be wrapped.
#ifdef __cpp_concepts
  template<typed_matrix V, std::convertible_to<const scalar_type_of_t<V>> S>
#else
  template<typename V, typename S, std::enable_if_t<
    typed_matrix<V> and std::is_convertible_v<S, const typename scalar_type_of<V>::type>, int> = 0>
#endif
  inline auto operator*(V&& v, S scale)
  {
    using Sc = scalar_type_of_t<V>;
    auto b = nested_matrix(std::forward<V>(v)) * static_cast<Sc>(scale);
    return MatrixTraits<V>::make(make_self_contained<V>(std::move(b)));
  }


  /// Multiply a scalar by a typed matrix. The result type is the same as the operand type, so angles in the result may be wrapped.
#ifdef __cpp_concepts
  template<typed_matrix V, std::convertible_to<const scalar_type_of_t<V>> S>
#else
  template<typename V, typename S, std::enable_if_t<typed_matrix<V> and
    std::is_convertible_v<S, const typename scalar_type_of<V>::type>, int> = 0>
#endif
  inline auto operator*(S scale, V&& v)
  {
    using Sc = const scalar_type_of_t<V>;
    auto b = static_cast<Sc>(scale) * nested_matrix(std::forward<V>(v));
    return MatrixTraits<V>::make(make_self_contained<V>(std::move(b)));
  }


  /// Divide a typed matrix by a scalar. The result type is the same as the operand type, so angles in the result may be wrapped.
#ifdef __cpp_concepts
  template<typed_matrix V, std::convertible_to<const scalar_type_of_t<V>> S>
#else
  template<typename V, typename S, std::enable_if_t<typed_matrix<V> and
    std::is_convertible_v<S, const typename scalar_type_of<V>::type>, int> = 0>
#endif
  inline auto operator/(V&& v, S scale)
  {
    using Sc = scalar_type_of_t<V>;
    auto b = nested_matrix(std::forward<V>(v)) / static_cast<Sc>(scale);
    return MatrixTraits<V>::make(make_self_contained<V>(std::move(b)));
  }


  /// Multiply a typed matrix by another typed matrix. The result is a regular typed matrix unless the first operand is EuclideanMean.
#ifdef __cpp_concepts
  template<typed_matrix V1, typed_matrix V2>
#else
  template<typename V1, typename V2, std::enable_if_t<typed_matrix<V1> and typed_matrix<V2>, int> = 0>
#endif
  inline auto operator*(V1&& v1, V2&& v2)
  {
    static_assert(equivalent_to<typename MatrixTraits<V1>::ColumnCoefficients, typename MatrixTraits<V2>::RowCoefficients>);
    static_assert(column_extent_of_v<V1> == row_extent_of_v<V2>);
    using RC = typename MatrixTraits<V1>::RowCoefficients;
    using CC = typename MatrixTraits<V2>::ColumnCoefficients;
    auto b = nested_matrix(std::forward<V1>(v1)) * nested_matrix(std::forward<V2>(v2));
    using CommonV = std::decay_t<std::conditional_t<euclidean_mean<V1>, V1, decltype(Matrix {v1})>>;
    return MatrixTraits<CommonV>::template make<RC, CC>(make_self_contained<V1, V2>(std::move(b)));
  }


  /// Negate a vector object. The result is a regular typed matrix unless the operand is EuclideanMean.
#ifdef __cpp_concepts
  template<typed_matrix V>
#else
  template<typename V, std::enable_if_t<typed_matrix<V>, int> = 0>
#endif
  inline auto operator-(V&& v)
  {
    using Res = std::decay_t<std::conditional_t<euclidean_mean<V>, V, decltype(Matrix {v})>>;
    auto b = -nested_matrix(std::forward<V>(v));
    return MatrixTraits<Res>::make(make_self_contained<V>(std::move(b)));
  }


  /// typed_matrix == typed_matrix.
#ifdef __cpp_concepts
  template<typed_matrix V1, typed_matrix V2>
#else
  template<typename V1, typename V2, std::enable_if_t<typed_matrix<V1> and typed_matrix<V2>, int> = 0>
#endif
  constexpr bool operator==(V1&& v1, V2&& v2)
  {
    if constexpr(
      equivalent_to<typename MatrixTraits<V1>::RowCoefficients, typename MatrixTraits<V2>::RowCoefficients> and
      equivalent_to<typename MatrixTraits<V1>::ColumnCoefficients, typename MatrixTraits<V2>::ColumnCoefficients>)
    {
      return make_native_matrix(std::forward<V1>(v1)) == make_native_matrix(std::forward<V2>(v2));
    }
    else
    {
      return false;
    }
  }


#ifndef __cpp_impl_three_way_comparison
  /// typed_matrix != typed_matrix.
#ifdef __cpp_concepts
  template<typed_matrix V1, typed_matrix V2>
#else
  template<typename V1, typename V2, std::enable_if_t<typed_matrix<V1> and typed_matrix<V2>, int> = 0>
#endif
  constexpr bool operator!=(V1&& v1, V2&& v2)
  {
    return not (std::forward<V1>(v1) == std::forward<V2>(v2));
  }
#endif


}

#endif //OPENKALMAN_TYPED_MATRIX_ARITHMETIC_H