/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2025 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definition for \ref triangle_type_of.
 */

#ifndef OPENKALMAN_TRIANGLE_TYPE_OF_HPP
#define OPENKALMAN_TRIANGLE_TYPE_OF_HPP

#include "linear-algebra/enumerations.hpp"
#include "linear-algebra/concepts/one_dimensional.hpp"
#include "linear-algebra/interfaces/object_traits.hpp"
#include "linear-algebra/concepts/one_dimensional.hpp"
#include "linear-algebra/concepts/zero.hpp"

namespace OpenKalman
{
  namespace detail
  {
#ifdef __cpp_concepts
    template<typename T>
#else
    template<typename T, typename = void>
#endif
    struct triangle_type_of_impl
      : std::integral_constant<triangle_type, zero<T> or one_dimensional<T, 2> ? triangle_type::diagonal : triangle_type::none> {};


#ifdef __cpp_concepts
    template<indexible T> requires
      requires { interface::object_traits<std::remove_cvref_t<T>>::triangle_type_value; }
    struct triangle_type_of_impl<T>
#else
    template<typename T>
    struct triangle_type_of_impl<T, std::void_t<decltype(interface::object_traits<stdex::remove_cvref_t<T>>::triangle_type_value)>>
#endif
      : std::integral_constant<
        triangle_type, zero<T> or one_dimensional<T, 2> ?
        triangle_type::diagonal :
        interface::object_traits<stdex::remove_cvref_t<T>>::triangle_type_value>
    {
      static_assert(interface::object_traits<stdex::remove_cvref_t<T>>::triangle_type_value != triangle_type::any,
        "triangle_type_value interface member cannot be triangle_type::any");
    };
  }


  /**
   * \brief The \ref triangle_type associated with an \ref indexible object.
   * \details If the argument is not triangular, the result will be \ref triangle_type::none.
   * A triangular matrix need not be \ref square_shaped, but it must be zero either above or below the diagonal (or both).
   * For rank >2 tensors, this must be applicable on every rank-2 slice comprising dimensions 0 and 1.
   * \note One-dimensional matrices or vectors are considered to be triangular, and a vector is triangular if
   * every component other than its first component is zero.
   */
  template<typename T>
  struct triangle_type_of : detail::triangle_type_of_impl<T> {};


  /**
   * \brief Helper template for \ref triangle_type_of.
   */
  template<typename T>
  constexpr auto triangle_type_of_v = triangle_type_of<T>::value;

}

#endif
