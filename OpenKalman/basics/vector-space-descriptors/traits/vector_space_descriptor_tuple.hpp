/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2024 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definition for \ref vector_space_descriptor_tuple.
 */

#ifndef OPENKALMAN_VECTOR_SPACE_DESCRIPTOR_TUPLE_HPP
#define OPENKALMAN_VECTOR_SPACE_DESCRIPTOR_TUPLE_HPP


namespace OpenKalman
{
  namespace detail
  {
    template<typename T, std::size_t...Ix>
    constexpr bool is_descriptor_tuple_impl(std::index_sequence<Ix...>)
    {
      return (... and vector_space_descriptor<std::tuple_element_t<Ix, T>>); 
    }
    
    
#ifdef __cpp_concepts
    template<typename T>
#else
    template<typename T, typename = void>
#endif
    struct is_descriptor_tuple : std::false_type {};

 
#ifdef __cpp_concepts
    template<tuple_like T>
    struct is_descriptor_tuple<T> 
#else
    template<typename T>
    struct is_descriptor_tuple<T, std::enable_if_t<tuple_like<T>>> 
#endif
      : std::bool_constant<is_descriptor_tuple_impl<T>(std::make_index_sequence<std::tuple_size_v<T>>{})> {}; 
 
  } // namespace detail
	
	
  /**
   * \brief An object describing a tuple-like collection of /ref vector_space_descriptor objects.
   */
  template<typename T>
#ifdef __cpp_lib_ranges
  concept vector_space_descriptor_tuple =
#else
  constexpr bool vector_space_descriptor_tuple =
#endif
    detail::is_descriptor_tuple<T>::value;


} // namespace OpenKalman

#endif //OPENKALMAN_VECTOR_SPACE_DESCRIPTOR_TUPLE_HPP
