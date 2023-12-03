/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2022 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definitions for \ref make_zero_matrix_like.
 */

#ifndef OPENKALMAN_MAKE_ZERO_MATRIX_LIKE_HPP
#define OPENKALMAN_MAKE_ZERO_MATRIX_LIKE_HPP

namespace OpenKalman
{
  /**
   * \brief Make a \ref zero_matrix associated with a particular library.
   * \tparam T A matrix or other tensor within a particular library. Its details are not important.
   * \tparam Scalar An optional scalar type for the new zero matrix. By default, T's scalar type is used.
   * \param Ds A set of \ref vector_space_descriptor defining the dimensions of each index.
   * If none are provided and T has no dynamic dimensions, the function takes \ref vector_space_descriptor from T.
   */
#ifdef __cpp_concepts
  template<indexible T, scalar_type Scalar = scalar_type_of_t<T>, vector_space_descriptor...Ds> requires
    (sizeof...(Ds) == index_count_v<T>) or (sizeof...(Ds) == 0 and not has_dynamic_dimensions<T>)
  constexpr zero_matrix auto
#else
  template<typename T, typename Scalar = scalar_type_of_t<T>, typename...Ds, std::enable_if_t<indexible<T> and
    scalar_type<Scalar> and (vector_space_descriptor<Ds> and ...) and
    (sizeof...(Ds) == index_count_v<T>) or (sizeof...(Ds) == 0 and not has_dynamic_dimensions<T>), int> = 0>
  constexpr auto
#endif
  make_zero_matrix_like(Ds&&...ds)
  {
    return make_constant_matrix_like<T, Scalar, 0>(std::forward<Ds>(ds)...);
  }


  /**
   * \overload
   * \brief Make a \ref zero_matrix based on an argument.
   * \tparam T The matrix or array on which the new zero matrix is patterned.
   * \tparam Scalar A scalar type for the new matrix.
   */
#ifdef __cpp_concepts
  template<scalar_type Scalar, indexible T>
  constexpr zero_matrix auto
#else
  template<typename Scalar, typename T, std::enable_if_t<scalar_type<Scalar> and indexible<T>, int> = 0>
  constexpr auto
#endif
  make_zero_matrix_like(const T& t)
  {
    return make_constant_matrix_like<Scalar, 0>(t);
  }


  /**
   * \overload
   * \brief Make a zero matrix based on T.
   * \details The new scalar type is also derived from T.
   */
#ifdef __cpp_concepts
  template<indexible T>
  constexpr zero_matrix auto
#else
  template<typename T, std::enable_if_t<indexible<T>, int> = 0>
  constexpr auto
#endif
  make_zero_matrix_like(const T& t)
  {
    return make_constant_matrix_like<scalar_type_of_t<T>, 0>(t);
  }


} // namespace OpenKalman

#endif //OPENKALMAN_MAKE_ZERO_MATRIX_LIKE_HPP
