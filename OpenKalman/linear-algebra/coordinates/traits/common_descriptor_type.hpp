/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2025 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definition for \ref collections::common_descriptor_type.
 */

#ifndef OPENKALMAN_COLLECTIONS_COMMON_DESCRIPTOR_TYPE_HPP
#define OPENKALMAN_COLLECTIONS_COMMON_DESCRIPTOR_TYPE_HPP

#include "collections/collections.hpp"
#include "linear-algebra/coordinates/concepts/descriptor_collection.hpp"

namespace OpenKalman::coordinates
{
  /**
   * \brief The common type within a \ref descriptor_collection, if it exists.
   * \details If T is \ref euclidean_pattern, the common type is Dimensions<1>.
   */
#ifdef __cpp_concepts
  template<typename T>
#else
  template<typename T, typename = void, typename = void>
#endif
  struct common_descriptor_type {};


#ifndef __cpp_concepts
  namespace internal
  {
    template<typename T, typename = void>
    struct has_common_collection_type : std::false_type {};

    template<typename T>
    struct has_common_collection_type<T, std::void_t<typename collections::common_collection_type<T>::type>> : std::true_type {};
  }
#endif


  /// \overload
#ifdef __cpp_concepts
  template<descriptor_collection T>
  struct common_descriptor_type<T>
#else
  template<typename T>
  struct common_descriptor_type<T, std::enable_if_t<descriptor_collection<T>>>
#endif
    : std::conditional<
      euclidean_pattern<T>,
      Dimensions<1>,
      typename std::conditional_t<
#ifdef __cpp_concepts
        requires { typename collections::common_collection_type<T>::type; },
#else
        internal::has_common_collection_type<T>::value,
#endif
        collections::common_collection_type<T>,
        stdcompat::type_identity<Any<>>
      >::type> {};


  /**
   * \brief Helper template for \ref common_descriptor_type.
   */
  template<typename T>
  using common_descriptor_type_t = typename common_descriptor_type<T>::type;

}

#endif
