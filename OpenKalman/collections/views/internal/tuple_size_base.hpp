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
 * \internal
 * \brief Definition for \ref collections::internal::tuple_size_base.
 */

#ifndef OPENKALMAN_COLLECTIONS_TUPLE_SIZE_BASE_HPP
#define OPENKALMAN_COLLECTIONS_TUPLE_SIZE_BASE_HPP

#include <tuple>
#include "collections/concepts/tuple_like.hpp"

namespace OpenKalman::collections::internal
{
  /**
   * \internal
   * \brief Base class for std::tuple_size specializations, potentially inheriting from std::tuple_size<T>.
   */
#ifdef __cpp_concepts
  template<typename T>
#else
  template<typename T, typename = void>
#endif
  struct tuple_size_base {};


#ifdef __cpp_concepts
  template<typename T> requires requires { std::tuple_size<T>::value; }
  struct tuple_size_base<T>
#else
  template<typename T>
  struct tuple_size_base<T, std::void_t<decltype(std::tuple_size<T>::value)>>
#endif
    : std::tuple_size<T> {};

}

#endif //OPENKALMAN_COLLECTIONS_TUPLE_SIZE_BASE_HPP
