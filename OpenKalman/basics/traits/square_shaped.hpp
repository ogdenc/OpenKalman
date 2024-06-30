/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2023 Christopher Lee Ogden <ogden@gatech.edu>
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


namespace OpenKalman
{
  namespace detail
  {
    template<typename T, std::size_t...Is>
    constexpr bool maybe_square_shaped(std::index_sequence<Is...>)
    {
      return maybe_equivalent_to<vector_space_descriptor_of_t<T, Is>...>;
    }


#ifndef __cpp_concepts
    template<typename T, Qualification b, typename = void>
    struct square_shaped_impl : std::false_type {};

    template<typename T, Qualification b>
    struct square_shaped_impl<T, b, std::enable_if_t<indexible<T> and (index_count<T>::value != dynamic_size)>>
      : std::bool_constant<(b != Qualification::unqualified or not has_dynamic_dimensions<T>) and
        (index_count_v<T> != 1 or dimension_size_of_index_is<T, 0, 1, Qualification::depends_on_dynamic_shape>) and
        (index_count_v<T> < 2 or maybe_square_shaped<T>(std::make_index_sequence<index_count_v<T>>{}))> {};
#endif

  } // namespace detail


#ifndef __cpp_concepts
  namespace internal
  {
    template<typename T, Qualification b, typename = void>
    struct is_explicitly_square : std::false_type {};

    template<typename T, Qualification b>
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
   * \brief Specifies that an object is square (i.e., has equivalent \ref vector_space_descriptor along each dimension).
   * \details An object is square iff it meets the following requirements:
   * - each index (if any) has the name number of dimensions,
   * - every \ref vector_space_descriptor (omitting any trailing 1D Euclidean descriptors) is equivalent.
   * \note An empty (0-by-0) matrix or tensor is considered to be square.
   * \tparam b Defines what happens when one or more of the indices has dynamic dimension:
   * - if <code>b == Qualification::unqualified</code>: T is known at compile time to be square;
   * - if <code>b == Qualification::depends_on_dynamic_shape</code>: It is known at compile time that T <em>may</em> be square.
   */
  template<typename T, Qualification b = Qualification::unqualified>
#ifdef __cpp_concepts
  concept square_shaped = one_dimensional<T, b> or (indexible<T> and
    (not interface::is_square_defined_for<T, b> or interface::indexible_object_traits<std::decay_t<T>>::template is_square<b>) and
    (interface::is_square_defined_for<T, b> or ((b != Qualification::unqualified or not has_dynamic_dimensions<T>) and
        (index_count_v<T> != 1 or dimension_size_of_index_is<T, 0, 1, Qualification::depends_on_dynamic_shape>) and
        (index_count_v<T> < 2 or detail::maybe_square_shaped<T>(std::make_index_sequence<index_count_v<T>>{}))) or
      (b == Qualification::unqualified and interface::indexible_object_traits<std::decay_t<T>>::template is_triangular<TriangleType::any, b>)));
#else
  constexpr bool square_shaped = one_dimensional<T, b> or (indexible<T> and
    (not interface::is_square_defined_for<T, b> or internal::is_explicitly_square<T, b>::value) and
    (interface::is_square_defined_for<T, b> or detail::square_shaped_impl<T, b>::value or
      (b == Qualification::unqualified and internal::is_explicitly_triangular<T, TriangleType::any>::value)));
#endif


} // namespace OpenKalman

#endif //OPENKALMAN_SQUARE_SHAPED_HPP
