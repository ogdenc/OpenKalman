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
 * \brief Definition for \ref static_vector_space_descriptor_collection.
 */

#ifndef OPENKALMAN_STATIC_VECTOR_SPACE_DESCRIPTOR_COLLECTION_HPP
#define OPENKALMAN_STATIC_VECTOR_SPACE_DESCRIPTOR_COLLECTION_HPP

#ifdef __cpp_lib_ranges
#include <ranges>
#endif
#include "basics/values/internal/collection.hpp"
#include "static_vector_space_descriptor.hpp"
#include "static_vector_space_descriptor_tuple.hpp"


namespace OpenKalman
{
#ifndef __cpp_lib_ranges
  namespace detail
  {
    template<typename T, typename = void>
    struct is_static_descriptor_range : std::false_type {};
 
    template<typename T>
    struct is_static_descriptor_range<T, std::enable_if_t<static_vector_space_descriptor<decltype(*std::declval<T>().begin())>>>
      : std::true_type {}; 
  } // namespace detail
#endif 
	
	
  /**
   * \brief An object describing a collection of /ref vector_space_descriptor objects.
   * \details This will be a \ref vector_space_descriptor_tuple or a dynamic range over a collection such as std::vector.
   */
  template<typename T>
#if defined(__cpp_lib_ranges) and defined(__cpp_lib_remove_cvref)
  concept static_vector_space_descriptor_collection = internal::collection<T> and
    (static_vector_space_descriptor_tuple<T> or static_vector_space_descriptor<std::ranges::range_value_t<std::decay_t<T>>>);
#else
  constexpr bool static_vector_space_descriptor_collection = internal::collection<T> and
    (static_vector_space_descriptor_tuple<T> or detail::is_static_descriptor_range<std::decay_t<T>>::value);
#endif


} // namespace OpenKalman

#endif //OPENKALMAN_STATIC_VECTOR_SPACE_DESCRIPTOR_COLLECTION_HPP
