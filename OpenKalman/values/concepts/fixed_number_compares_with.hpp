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
 * \brief Definition for \values::fixed_number_compares_with.
 */

#ifndef OPENKALMAN_VALUE_FIXED_NUMBER_COMPARES_WITH_HPP
#define OPENKALMAN_VALUE_FIXED_NUMBER_COMPARES_WITH_HPP

#include "basics/basics.hpp"
#include "fixed.hpp"
#include "values/traits/fixed_number_of.hpp"

namespace OpenKalman::values
{
#ifndef __cpp_concepts
  namespace detail
  {
    template<typename T, auto N, typename Comp, typename = void>
    struct fixed_number_is_impl : std::false_type {};

    template<typename T, auto N, typename Comp>
    struct fixed_number_is_impl<T, N, Comp, std::enable_if_t<(stdcompat::invoke(Comp{}, fixed_number_of<T>::value, N))>>
      : std::true_type {};
  }
#endif


  /**
   * \brief T has a fixed value that compares with N according to Comp.
   */
  template<typename T, auto N, typename Comp = std::equal_to<>>
#ifdef __cpp_concepts
  concept fixed_number_compares_with = fixed<T> and stdcompat::invoke(Comp{}, fixed_number_of_v<T>, N);
#else
  constexpr bool fixed_number_compares_with = detail::fixed_number_is_impl<T, N, Comp>::value;
#endif

}

#endif
