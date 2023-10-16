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
   * \tparam T A dummy matrix or array from the relevant library (size, shape, and layout are ignored)
   * \tparam layout The \ref Layout of the resulting object. If this is Layout::none, it will be the default layout for the library of T.
   * \tparam Scalar The scalar type of the resulting object (by default, it is the same scalar type as T).
   * \param d a tuple of \ref vector_space_descriptor describing dimensions of each index.
   * These can be omitted, in which case the \ref vector_space_descriptor will be derived from T.
   */
#ifdef __cpp_concepts
  template<indexible T, Layout layout = Layout::none, scalar_type Scalar = scalar_type_of_t<T>>
  constexpr writable auto
  make_default_dense_writable_matrix_like(vector_space_descriptor auto&&...d)
    requires (sizeof...(d) > 0 or not has_dynamic_dimensions<T>) and (layout != Layout::stride)
#else
  template<typename T, Layout layout = Layout::none, typename Scalar = scalar_type_of_t<T>, typename...D, std::enable_if_t<
    indexible<T> and scalar_type<Scalar> and (vector_space_descriptor<D> and ...) and
    (sizeof...(D) > 0 or not has_dynamic_dimensions<T>) and (layout != Layout::stride), int> = 0>
  constexpr auto
  make_default_dense_writable_matrix_like(D&&...d)
#endif
  {
    if constexpr (sizeof...(d) == 0)
    {
      return std::apply(
        [](const auto&...d) {
          return make_default_dense_writable_matrix_like<T, layout, Scalar>(d...);
        }, get_all_dimensions_of<T>());
    }
    else
    {
      using Traits = interface::library_interface<std::decay_t<T>>;
      return Traits::template make_default<layout, Scalar>(std::forward<decltype(d)>(d)...);
    }
  }


  /**
   * \overload
   * \brief Make a default, dense, writable matrix based on an existing object.
   * \tparam layout The \ref Layout of the resulting object.
   * \tparam Scalar The scalar type of the resulting object.
   * \param t The existing object
   * \tparam Scalar A scalar type for the new matrix.
   */
#ifdef __cpp_concepts
  template<Layout layout, scalar_type Scalar> requires (layout != Layout::stride)
  constexpr writable auto
  make_default_dense_writable_matrix_like(const indexible auto& t)
#else
  template<Layout layout, typename Scalar, typename T, std::enable_if_t<scalar_type<Scalar> and indexible<T> and
    (layout != Layout::stride), int> = 0>
  constexpr auto
  make_default_dense_writable_matrix_like(const T& t)
#endif
  {
    return std::apply(
      [](auto&&...d) {
        return make_default_dense_writable_matrix_like<decltype(t), layout, Scalar>(std::forward<decltype(d)>(d)...);
      }, get_all_dimensions_of(t));
  }


  /**
   * \overload
   * \brief Make a default, dense, writable matrix based on an existing object, with the same scalar type.
   * \tparam layout The \ref Layout of the resulting object. If this is \ref Layout::none, it will be
   * the default layout for the library (<em>not</em> the layout of t!).
   * \param t The existing object
   */
#ifdef __cpp_concepts
  template<Layout layout = Layout::none> requires (layout != Layout::stride)
  constexpr writable auto
  make_default_dense_writable_matrix_like(const indexible auto& t)
#else
  template<Layout layout = Layout::none, typename T, std::enable_if_t<indexible<T> and (layout != Layout::stride), int> = 0>
  constexpr auto
  make_default_dense_writable_matrix_like(const T& t)
#endif
  {
    return make_default_dense_writable_matrix_like<layout, scalar_type_of_t<decltype(t)>>(t);
  }


} // namespace OpenKalman

#endif //OPENKALMAN_MAKE_DEFAULT_DENSE_WRITABLE_MATRIX_LIKE_HPP
