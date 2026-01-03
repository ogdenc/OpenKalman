/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2025-2026 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definition for \ref collections::pattern_collection_element.
 */

#ifndef OPENKALMAN_PATTERNS_PATTERN_COLLECTION_ELEMENT_HPP
#define OPENKALMAN_PATTERNS_PATTERN_COLLECTION_ELEMENT_HPP

#include "collections/collections.hpp"
#include "patterns/functions/get_pattern.hpp"

namespace OpenKalman::patterns
{
  /**
   * \brief The type of the element at a given index, if it can be determined at compile time.
   * \details This is a generalized version of std::tuple_element.
   * If i exceeds the size of T (if that size is fixed), the result will be Dimensions<1>.
   */
#ifdef __cpp_concepts
  template<std::size_t i, typename T>
#else
  template<std::size_t i, typename T, typename = void>
#endif
  struct pattern_collection_element {};


  /**
   * \overload
   * \details Does not do any bounds checking if the size is dynamic.
   */
#ifdef __cpp_concepts
  template<std::size_t i, pattern_collection T>
  struct pattern_collection_element<i, T>
#else
  template<std::size_t i, typename T>
  struct pattern_collection_element<i, T, std::enable_if_t<pattern_collection<T>>>
#endif
  {
    using type = OpenKalman::internal::remove_rvalue_reference_t<
      decltype(patterns::get_pattern(std::declval<T>(), std::integral_constant<std::size_t, i>{}))>;
  };


  /**
   * \brief Helper template for \ref collection_element.
   */
  template<std::size_t i, typename T>
  using pattern_collection_element_t = typename pattern_collection_element<i, T>::type;


}

#endif
