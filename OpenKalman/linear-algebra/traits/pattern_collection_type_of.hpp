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
 * \brief Definition for \ref pattern_collection_type_of.
 */

#ifndef OPENKALMAN_PATTERN_COLLECTION_TYPE_OF_HPP
#define OPENKALMAN_PATTERN_COLLECTION_TYPE_OF_HPP


#include "linear-algebra/traits/get_mdspan.hpp"

namespace OpenKalman
{
  /**
   * \brief The \ref patterns::pattern_collection "pattern_collection" type of an \ref indexible object.
   * \tparam T A tensor or other array.
   */
#ifdef __cpp_concepts
  template<typename T>
#else
  template<typename T, typename = void>
#endif
  struct pattern_collection_type_of {};


  /**
   * \overload
   */
#ifdef __cpp_concepts
  template<indexible T>
  struct pattern_collection_type_of<T>
#else
  template<typename T>
  struct pattern_collection_type_of<T, std::enable_if_t<indexible<T>>>
#endif
  {
    using type = typename std::decay_t<decltype(get_pattern_collection(std::declval<T&>()))>;
  };


  /**
   * \brief helper template for \ref pattern_collection_type_of.
   */
  template<typename T>
  using pattern_collection_type_of_t = typename pattern_collection_type_of<T>::type;


}

#endif
