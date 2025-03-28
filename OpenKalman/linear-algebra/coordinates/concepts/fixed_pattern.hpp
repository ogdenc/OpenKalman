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
 * \brief Definition for \ref coordinate::fixed_pattern.
 */

#ifndef OPENKALMAN_COORDINATE_FIXED_PATTERN_HPP
#define OPENKALMAN_COORDINATE_FIXED_PATTERN_HPP

#include <type_traits>
#include "values/concepts/fixed.hpp"
#include "pattern.hpp"
#include "linear-algebra/coordinates/traits/size_of.hpp"

namespace OpenKalman::coordinate
{
#ifndef __cpp_concepts
  namespace detail
  {
    template<typename T, typename = void>
    struct fixed_pattern_impl : std::false_type {};

    template<typename T>
    struct fixed_pattern_impl<T, std::enable_if_t<size_of<T>::value != dynamic_size>> : std::true_type {};
  }
#endif


  /**
   * \brief A \ref coordinate::pattern for which the \ref coordinate::size_of "size" is fixed at compile time.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept fixed_pattern = std::default_initializable<std::decay_t<T>> and
    pattern<T> and (size_of_v<T> != dynamic_size);
#else
  constexpr bool fixed_pattern = std::is_default_constructible<std::decay_t<T>>::value and
    detail::fixed_pattern_impl<T>::value;
#endif


} // namespace OpenKalman::coordinate

#endif //OPENKALMAN_COORDINATE_FIXED_PATTERN_HPP
