/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2022-2024 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definitions for \ref make_zero.
 */

#ifndef OPENKALMAN_MAKE_ZERO_HPP
#define OPENKALMAN_MAKE_ZERO_HPP

#include "make_constant.hpp"

namespace OpenKalman
{
  /**
   * \brief Make an \ref indexible object in which every element is 0.
   * \returns An mdspan returning the constant value at every set of indices.
   * \param extents An std::extents object defining the extents.
   */
#ifdef __cpp_concepts
  template<values::value C = double, values::integral IndexType, std::size_t...Extents>
  constexpr constant_object auto
#else
  template<typename C = double, typename IndexType, std::size_t...Extents, std::enable_if_t<values::value<C>, int> = 0>
  constexpr auto
#endif
  make_zero(stdex::extents<IndexType, Extents...> extents)
  {
    return make_constant(values::fixed_value<C, 0>{}, std::move(extents));
  }


  /**
   * \brief Make a \ref zero associated with a particular library.
   * \tparam T An \indexible object (matrix or tensor) from a particular library. Its shape and contents are irrelevant.
   * \tparam Scalar An optional scalar type for the new zero matrix. By default, T's scalar type is used.
   * \param Descriptors A \ref pattern_collection defining the dimensions of each index.
   * If none are provided and T has no dynamic dimensions, the function takes \ref coordinates::pattern from T.
   */
/*#ifdef __cpp_concepts
  template<indexible T, values::number Scalar = scalar_type_of_t<T>, pattern_collection Descriptors>
  constexpr zero auto
#else
  template<typename T, typename Scalar = scalar_type_of_t<T>, typename...Ds, std::enable_if_t<indexible<T> and
    values::number<Scalar> and pattern_collection<Descriptors>, int> = 0>
  constexpr auto
#endif
  make_zero(Descriptors&& descriptors)
  {
    return make_constant<T, Scalar, 0>(std::forward<Descriptors>(descriptors));
  }*/


  /**
   * \overload
   * \brief Specify \ref coordinates::pattern objects as arguments.
   */
/*#ifdef __cpp_concepts
  template<indexible T, values::number Scalar = scalar_type_of_t<T>, coordinates::pattern...Ds>
  constexpr zero auto
#else
  template<typename T, typename Scalar = scalar_type_of_t<T>, typename...Ds, std::enable_if_t<indexible<T> and
    values::number<Scalar> and (... and coordinates::pattern<Ds>), int> = 0>
  constexpr auto
#endif
  make_zero(Ds&&...ds)
  {
    return make_zero<T, Scalar>(std::tuple {std::forward<Ds>(ds)...});
  }*/


  /**
   * \overload
   * \brief Make a \ref zero based on an argument.
   * \tparam T The matrix or array on which the new zero matrix is patterned.
   * \tparam Scalar A scalar type for the new matrix.
   */
/*#ifdef __cpp_concepts
  template<values::number Scalar, indexible T>
  constexpr zero auto
#else
  template<typename Scalar, typename T, std::enable_if_t<values::number<Scalar> and indexible<T>, int> = 0>
  constexpr auto
#endif
  make_zero(const T& t)
  {
    return make_constant<Scalar, 0>(t);
  }*/



  /**
   * \overload
   * \brief Make a zero matrix based on T.
   * \details The new scalar type is also derived from T.
   */
/*#ifdef __cpp_concepts
  template<indexible T>
  constexpr zero auto
#else
  template<typename T, std::enable_if_t<indexible<T>, int> = 0>
  constexpr auto
#endif
  make_zero(const T& t)
  {
    return make_constant<scalar_type_of_t<T>, 0>(t);
  }*/


}

#endif
