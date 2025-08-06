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
 * \brief Definition for \ref uniform_pattern.
 */

#ifndef OPENKALMAN_UNIFORM_PATTERN_HPP
#define OPENKALMAN_UNIFORM_PATTERN_HPP

#include "linear-algebra/coordinates/traits/uniform_pattern_type.hpp"

namespace OpenKalman::coordinates
{
#ifndef __cpp_concepts
  namespace detail
  {
    template<typename T, typename = void>
    struct uniform_pattern_impl : std::false_type {};

    template<typename T>
    struct uniform_pattern_impl<T, std::void_t<typename uniform_pattern_type<T>::type>> : std::true_type {};
  }
#endif


  /**
   * \brief T is a \ref coordinates::pattern that is either empty or can be decomposed into a uniform set of 1D \ref coordinates::pattern.
   * \details If T is a uniform pattern, \ref uniform_pattern_type<T>::type will exist and will be one-dimensional.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept uniform_pattern = requires { typename uniform_pattern_type<T>::type; };
#else
  constexpr bool uniform_pattern = detail::uniform_pattern_impl<T>::value;
#endif


}

#endif
