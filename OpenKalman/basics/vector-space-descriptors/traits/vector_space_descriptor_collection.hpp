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
 * \brief Definition for \ref vector_space_descriptor_collection.
 */

#ifndef OPENKALMAN_VECTOR_SPACE_DESCRIPTOR_COLLECTION_HPP
#define OPENKALMAN_VECTOR_SPACE_DESCRIPTOR_COLLECTION_HPP


namespace OpenKalman
{
#ifndef __cpp_lib_ranges
  namespace detail
  {
    template<typename T, typename = void>
    struct is_descriptor_range : std::false_type {};
 
    template<typename T>
    struct is_descriptor_range<T, std::enable_if_t<vector_space_descriptor<decltype(*std::declval<T>().begin())>>> 
      : std::true_type {}; 
  } // namespace detail
#endif 
	
	
  /**
   * \brief An object describing a collection of /ref vector_space_descriptor objects.
   * \details This will be a \ref vector_space_descriptor_tuple or a dynamic range over a collection such as std::vector.
   */
  template<typename T>
#ifdef __cpp_lib_ranges
  concept vector_space_descriptor_collection = vector_space_descriptor_tuple<T> or 
      (std::ranges::input_range<T> and vector_space_descriptor<std::ranges::range_value_t<T>>);
#else
  constexpr bool vector_space_descriptor_collection = 
    vector_space_descriptor_tuple<T> or detail::is_descriptor_range<T>::value;
#endif


} // namespace OpenKalman

#endif //OPENKALMAN_VECTOR_SPACE_DESCRIPTOR_COLLECTION_HPP
