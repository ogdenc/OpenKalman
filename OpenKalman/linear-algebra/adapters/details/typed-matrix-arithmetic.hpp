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
    using RC1 = vector_space_descriptor_of_t<V1, 0>;
    using CC1 = vector_space_descriptor_of_t<V1, 1>;
    static_assert(equivalent_to<vector_space_descriptor_of_t<V2, 0>, RC1>);
    static_assert(equivalent_to<vector_space_descriptor_of_t<V2, 1>, CC1>);
    static_assert(euclidean_transformed<V1> == euclidean_transformed<V2>);

    auto b = make_self_contained<V1, V2>(nested_object(std::forward<V1>(v1)) + nested_object(std::forward<V2>(v2)));

    if constexpr (euclidean_mean<V1> and euclidean_mean<V2>) return make_euclidean_mean<RC1>(std::move(b));
    else if constexpr (mean<V1> and mean<V2>) return make_mean<RC1>(std::move(b));
    else return make_vector_space_adapter(std::move(b), RC1{}, CC1{});
  }


  /// Subtract two typed matrices. The result is a regular typed matrix unless both operands are EuclideanMean.
#ifdef __cpp_concepts
  template<typed_matrix V1, typed_matrix V2>
#else
  template<typename V1, typename V2, std::enable_if_t<typed_matrix<V1> and typed_matrix<V2>, int> = 0>
#endif
  inline auto operator-(V1&& v1, V2&& v2)
  {
    using RC1 = vector_space_descriptor_of_t<V1, 0>;
    using CC1 = vector_space_descriptor_of_t<V1, 1>;
    static_assert(equivalent_to<vector_space_descriptor_of_t<V2, 0>, RC1>);
    static_assert(equivalent_to<vector_space_descriptor_of_t<V2, 1>, CC1>);
    static_assert(euclidean_transformed<V1> == euclidean_transformed<V2>);

    auto b = make_self_contained<V1, V2>(nested_object(std::forward<V1>(v1)) - nested_object(std::forward<V2>(v2)));

    if constexpr (euclidean_mean<V1> and euclidean_mean<V2>) return make_euclidean_mean<RC1>(std::move(b));
    else if constexpr (mean<V1> and mean<V2>) return make_vector_space_adapter(std::move(b), typename RC1::difference_type{}, CC1{});
    else return make_vector_space_adapter(std::move(b), RC1{}, CC1{});
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
    using RC = vector_space_descriptor_of_t<V, 0>;
    using CC = vector_space_descriptor_of_t<V, 1>;
    using Sc = scalar_type_of_t<V>;

    auto b = make_self_contained<V>(nested_object(std::forward<V>(v)) * static_cast<Sc>(scale));

    if constexpr (euclidean_mean<V>) return make_euclidean_mean<RC>(std::move(b));
    else if constexpr (mean<V>) return make_mean<RC>(std::move(b));
    else return make_vector_space_adapter(std::move(b), RC{}, CC{});
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
    using RC = vector_space_descriptor_of_t<V, 0>;
    using CC = vector_space_descriptor_of_t<V, 1>;
    using Sc = const scalar_type_of_t<V>;

    auto b = make_self_contained<V>(static_cast<Sc>(scale) * nested_object(std::forward<V>(v)));

    if constexpr (euclidean_mean<V>) return make_euclidean_mean<RC>(std::move(b));
    else if constexpr (mean<V>) return make_mean<RC>(std::move(b));
    else return make_vector_space_adapter(std::move(b), RC{}, CC{});
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
    using RC = vector_space_descriptor_of_t<V, 0>;
    using CC = vector_space_descriptor_of_t<V, 1>;
    using Sc = scalar_type_of_t<V>;

    auto b = make_self_contained<V>(nested_object(std::forward<V>(v)) / static_cast<Sc>(scale));

    if constexpr (euclidean_mean<V>) return make_euclidean_mean<RC>(std::move(b));
    else if constexpr (mean<V>) return make_mean<RC>(std::move(b));
    else return make_vector_space_adapter(std::move(b), RC{}, CC{});
  }


  /// Multiply a typed matrix by another typed matrix. The result is a regular typed matrix unless the first operand is EuclideanMean.
#ifdef __cpp_concepts
  template<typed_matrix V1, typed_matrix V2>
#else
  template<typename V1, typename V2, std::enable_if_t<typed_matrix<V1> and typed_matrix<V2>, int> = 0>
#endif
  inline auto operator*(V1&& v1, V2&& v2)
  {
    static_assert(equivalent_to<vector_space_descriptor_of_t<V1, 1>, vector_space_descriptor_of_t<V2, 0>>);
    using RC = vector_space_descriptor_of_t<V1, 0>;
    using CC = vector_space_descriptor_of_t<V2, 1>;

    auto b = make_self_contained<V1, V2>(nested_object(std::forward<V1>(v1)) * nested_object(std::forward<V2>(v2)));

    if constexpr (euclidean_mean<V1> and euclidean_mean<V2>) return make_euclidean_mean<RC>(std::move(b));
    else return make_vector_space_adapter(std::move(b), RC{}, CC{});
  }


  /// Negate a vector object. The result is a regular typed matrix unless the operand is EuclideanMean.
#ifdef __cpp_concepts
  template<typed_matrix V>
#else
  template<typename V, std::enable_if_t<typed_matrix<V>, int> = 0>
#endif
  inline auto operator-(V&& v)
  {
    using RC = vector_space_descriptor_of_t<V, 0>;
    using CC = vector_space_descriptor_of_t<V, 1>;

    auto b = make_self_contained<V>(-nested_object(std::forward<V>(v)));

    if constexpr (euclidean_mean<V>) return make_euclidean_mean<RC>(std::move(b));
    else return make_vector_space_adapter(std::move(b), RC{}, CC{});
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
      equivalent_to<vector_space_descriptor_of_t<V1, 0>, vector_space_descriptor_of_t<V2, 0>> and
      equivalent_to<vector_space_descriptor_of_t<V1, 1>, vector_space_descriptor_of_t<V2, 1>>)
    {
      return to_dense_object(std::forward<V1>(v1)) == to_dense_object(std::forward<V2>(v2));
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