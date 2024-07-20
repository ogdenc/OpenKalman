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
 * \brief Definition for \ref one_dimensional.
 */

#ifndef OPENKALMAN_ONE_DIMENSIONAL_HPP
#define OPENKALMAN_ONE_DIMENSIONAL_HPP


namespace OpenKalman
{
  namespace detail
  {
    template<typename T, Qualification b, std::size_t...Is>
    constexpr bool has_1_by_1_dims(std::index_sequence<Is...>)
    {
      return maybe_equivalent_to<vector_space_descriptor_of_t<T, Is>...> and (dimension_size_of_index_is<T, Is, 1, b> and ...);
    }


#ifdef __cpp_concepts
    template<typename T, Qualification b>
#else
    template<typename T, Qualification b, typename = void>
#endif
    struct one_dimensional_impl : std::false_type {};


#ifdef __cpp_concepts
    template<typename T, Qualification b> requires (index_count_v<T> == dynamic_size)
    struct one_dimensional_impl<T, b>
#else
    template<typename T, Qualification b>
    struct one_dimensional_impl<T, b, std::enable_if_t<index_count<T>::value == dynamic_size>>
#endif
      : std::bool_constant<b == Qualification::depends_on_dynamic_shape and detail::has_1_by_1_dims<T, b>(std::make_index_sequence<2>{})> {};


#ifdef __cpp_concepts
    template<typename T, Qualification b> requires (index_count_v<T> != dynamic_size)
    struct one_dimensional_impl<T, b>
#else
    template<typename T, Qualification b>
    struct one_dimensional_impl<T, b, std::enable_if_t<index_count<T>::value != dynamic_size>>
#endif
      : std::bool_constant<detail::has_1_by_1_dims<T, b>(std::make_index_sequence<index_count_v<T>>{})> {};
  } // namespace detail


  /**
   * \brief Specifies that a type is one-dimensional in every index.
   * \details Each index also must have an equivalent \ref vector_space_descriptor object.
   */
  template<typename T, Qualification b = Qualification::unqualified>
#ifdef __cpp_concepts
  concept one_dimensional = indexible<T> and
    (not interface::one_dimensional_defined_for<T, b> or interface::indexible_object_traits<std::decay_t<T>>::template one_dimensional<b>) and
    (interface::one_dimensional_defined_for<T, b> or detail::one_dimensional_impl<T, b>::value);
#else
  constexpr bool one_dimensional = indexible<T> and
    (interface::one_dimensional_defined_for<T, b> ? interface::is_explicitly_one_dimensional<T, b>::value : detail::one_dimensional_impl<T, b>::value);
#endif


} // namespace OpenKalman

#endif //OPENKALMAN_ONE_DIMENSIONAL_HPP
