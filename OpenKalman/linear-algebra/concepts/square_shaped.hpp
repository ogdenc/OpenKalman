/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2024 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definition for \ref square_shaped.
 */

#ifndef OPENKALMAN_SQUARE_SHAPED_HPP
#define OPENKALMAN_SQUARE_SHAPED_HPP

#include "linear-algebra/coordinates/concepts/compares_with.hpp"

namespace OpenKalman
{
  namespace detail
  {
    template<typename T, std::size_t I, std::size_t...Is>
    constexpr bool maybe_square_shaped(std::index_sequence<I, Is...>)
    {
      return (... and compares_with<vector_space_descriptor_of_t<T, I>, vector_space_descriptor_of_t<T, Is>, equal_to<>, Applicability::permitted>);
    }


#ifndef __cpp_concepts
    template<typename T, Applicability b, typename = void>
    struct square_shaped_impl : std::false_type {};

    template<typename T, Applicability b>
    struct square_shaped_impl<T, b, std::enable_if_t<indexible<T> and (index_count<T>::value != dynamic_size) and (index_count<T>::value > 1)>>
      : std::bool_constant<(b != Applicability::guaranteed or not has_dynamic_dimensions<T>) and
        (index_count_v<T> != 1 or dimension_size_of_index_is<T, 0, 1, Applicability::permitted>) and
        (index_count_v<T> < 2 or maybe_square_shaped<T>(std::make_index_sequence<index_count_v<T>>{}))> {};
#endif

  } // namespace detail


#ifndef __cpp_concepts
  namespace internal
  {
    template<typename T, Applicability b, typename = void>
    struct is_explicitly_square : std::false_type {};

    template<typename T, Applicability b>
    struct is_explicitly_square<T, b, std::enable_if_t<interface::indexible_object_traits<std::decay_t<T>>::template is_square<b>>>
      : std::true_type {};


    template<typename T, TriangleType t, typename = void>
    struct is_explicitly_triangular : std::false_type {};

    template<typename T, TriangleType t>
    struct is_explicitly_triangular<T, t, std::enable_if_t<interface::indexible_object_traits<std::decay_t<T>>::template is_triangular<t>>>
      : std::true_type {};
  }
#endif


  /**
   * \brief Specifies that an object is square (i.e., has equivalent \ref coordinates::pattern along each dimension).
   * \details Any trailing 1D Euclidean descriptors are disregarded. A vector must be one-dimensional.
   * An empty (0-by-0) matrix or tensor is considered to be square.
   * \tparam b Defines what happens when one or more of the indices has dynamic dimension:
   * - if <code>b == Applicability::guaranteed</code>: T is known at compile time to be square;
   * - if <code>b == Applicability::permitted</code>: It is known at compile time that T <em>may</em> be square.
   */
  template<typename T, Applicability b = Applicability::guaranteed>
#ifdef __cpp_concepts
  concept square_shaped = one_dimensional<T, b> or (indexible<T> and
    (not interface::is_square_defined_for<T, b> or interface::indexible_object_traits<std::decay_t<T>>::template is_square<b>) and
    (interface::is_square_defined_for<T, b> or ((b != Applicability::guaranteed or not has_dynamic_dimensions<T>) and
        (index_count_v<T> != 1 or dimension_size_of_index_is<T, 0, 1, Applicability::permitted>) and
        (index_count_v<T> < 2 or detail::maybe_square_shaped<T>(std::make_index_sequence<index_count_v<T>>{}))) or
      (b == Applicability::guaranteed and interface::indexible_object_traits<std::decay_t<T>>::template is_triangular<TriangleType::any, b>)));
#else
  constexpr bool square_shaped = one_dimensional<T, b> or (indexible<T> and
    (not interface::is_square_defined_for<T, b> or internal::is_explicitly_square<T, b>::value) and
    (interface::is_square_defined_for<T, b> or detail::square_shaped_impl<T, b>::value or
      (b == Applicability::guaranteed and internal::is_explicitly_triangular<T, TriangleType::any>::value)));
#endif


} // namespace OpenKalman

#endif //OPENKALMAN_SQUARE_SHAPED_HPP
