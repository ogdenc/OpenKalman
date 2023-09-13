/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2022-2023 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Overloaded general functions for making a default dense writable object.
 */

#ifndef OPENKALMAN_MAKE_DEFAULT_DENSE_WRITABLE_MATRIX_LIKE_HPP
#define OPENKALMAN_MAKE_DEFAULT_DENSE_WRITABLE_MATRIX_LIKE_HPP

namespace OpenKalman
{
  // ----------------------------------------- //
  //  make_default_dense_writable_matrix_like  //
  // ----------------------------------------- //

  /**
   * \brief Make a default, dense, writable matrix based on a list of Dimensions describing the sizes of each index.
   * \description If
   * \tparam T A dummy matrix or array from the relevant library (size and shape does not matter)
   * \param d a tuple of Dimensions describing the sizes of each index. This can be omitted if T is of fixed size.
   * In that case, the index descriptors will be derived from T.
   */
#ifdef __cpp_concepts
  template<indexible T, scalar_type Scalar = scalar_type_of_t<T>, index_descriptor...D> requires
    (sizeof...(D) > 0 or not has_dynamic_dimensions<T>)
  constexpr writable auto
#else
  template<typename T, typename Scalar = scalar_type_of_t<T>, typename...D, std::enable_if_t<
    indexible<T> and scalar_type<Scalar> and (index_descriptor<D> and ...) and
    (sizeof...(D) > 0 or not has_dynamic_dimensions<T>), int> = 0>
  constexpr auto
#endif
  make_default_dense_writable_matrix_like(D&&...d)
  {
    if constexpr (sizeof...(D) == 0)
      return std::apply(
        [](auto&&...d) { return make_default_dense_writable_matrix_like<T, Scalar>(std::forward<decltype(d)>(d)...); },
        get_all_dimensions_of<T>());
    else
      return interface::LibraryRoutines<std::decay_t<T>>::template make_default<Scalar>(std::forward<D>(d)...);
  }


  /**
   * \overload
   * \brief Make a default, dense, writable matrix based on an existing object.
   * \param t The existing object
   * \tparam Scalar A scalar type for the new matrix.
   */
#ifdef __cpp_concepts
  template<scalar_type Scalar, indexible T>
  constexpr writable auto
#else
  template<typename Scalar, typename T, std::enable_if_t<scalar_type<Scalar> and indexible<T>, int> = 0>
  constexpr auto
#endif
  make_default_dense_writable_matrix_like(const T& t)
  {
    using Traits = interface::LibraryRoutines<T>;
    return std::apply(
      [](auto&&...d) { return Traits::template make_default<Scalar>(std::forward<decltype(d)>(d)...); },
      get_all_dimensions_of(t));
  }


  /**
   * \overload
   * \brief Make a default, dense, writable matrix based on an existing object, with the same scalar type.
   * \param t The existing object
   */
#ifdef __cpp_concepts
  template<indexible T>
  constexpr writable auto
#else
  template<typename T, std::enable_if_t<indexible<T>, int> = 0>
  constexpr auto
#endif
  make_default_dense_writable_matrix_like(const T& t)
  {
    return make_default_dense_writable_matrix_like<scalar_type_of_t<T>>(t);
  }


} // namespace OpenKalman

#endif //OPENKALMAN_MAKE_DEFAULT_DENSE_WRITABLE_MATRIX_LIKE_HPP
