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
 * \brief Definition for \ref has_nested_vector.
 */

#ifndef OPENKALMAN_HAS_NESTED_VECTOR_HPP
#define OPENKALMAN_HAS_NESTED_VECTOR_HPP

#include "traits/nested_object_of.hpp"

namespace OpenKalman::internal
{
#ifndef __cpp_concepts
  namespace detail
  {
    template<typename T, std::size_t N, typename = void>
    struct has_nested_vector_impl : std::false_type {};

    template<typename T, std::size_t N>
    struct has_nested_vector_impl<T, N, std::enable_if_t<has_nested_object<T>>>
      : std::bool_constant<vector<nested_object_of_t<T>, N>> {};
  }
#endif


  /**
   * \brief Specifies that a type is a wrapper containing a nested vector.
   * \tparam T A matrix or tensor.
   * \tparam N An index designating the "large" index of the nested vector (0 for a column vector, 1 for a row vector)
   */
  template<typename T, std::size_t N = 0>
#ifdef __cpp_concepts
  concept has_nested_vector = vector<nested_object_of_t<T>, N>;
#else
  constexpr bool has_nested_vector = detail::has_nested_vector_impl<T, N>::value;
#endif


}

#endif
