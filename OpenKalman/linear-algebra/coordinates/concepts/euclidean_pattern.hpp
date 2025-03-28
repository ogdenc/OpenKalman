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
 * \brief Definition for \ref coordinate::euclidean_pattern.
 */

#ifndef OPENKALMAN_EUCLIDEAN_VECTOR_SPACE_DESCRIPTOR_HPP
#define OPENKALMAN_EUCLIDEAN_VECTOR_SPACE_DESCRIPTOR_HPP

#include <type_traits>
#include "values/concepts/fixed.hpp"
#include "values/traits/fixed_number_of.hpp"
#include "linear-algebra/coordinates/concepts/pattern.hpp"
#include "linear-algebra/coordinates/functions/get_is_euclidean.hpp"

namespace OpenKalman::coordinate
{
#ifndef __cpp_concepts
  namespace detail
  {
    template<typename T, typename = void>
    struct euclidean_pattern_impl : std::false_type {};

    template<typename T>
    struct euclidean_pattern_impl<T, std::enable_if_t<value::fixed<decltype(coordinate::get_is_euclidean(std::declval<T>()))>>>
      : std::bool_constant<value::fixed_number_of_v<decltype(coordinate::get_is_euclidean(std::declval<T>()))>> {};
  }
#endif


  /**
   * \brief A \ref coordinate::pattern for a normal Euclidean vector.
   */
#ifdef __cpp_concepts
  template<typename T>
  concept euclidean_pattern = pattern<T> and
    (value::fixed_number_of<decltype(coordinate::get_is_euclidean(std::declval<T>()))>::value);
#else
  template<typename T>
  constexpr bool euclidean_pattern = detail::euclidean_pattern_impl<T>::value;
#endif

} // namespace OpenKalman::coordinate

#endif //OPENKALMAN_EUCLIDEAN_VECTOR_SPACE_DESCRIPTOR_HPP
