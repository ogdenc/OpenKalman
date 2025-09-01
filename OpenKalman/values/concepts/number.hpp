/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2022-2025 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definition for \ref values::number.
 */

#ifndef OPENKALMAN_VALUES_NUMBER_HPP
#define OPENKALMAN_VALUES_NUMBER_HPP

#include "basics/basics.hpp"
#include "values/interface/number_traits.hpp"

namespace OpenKalman::values
{
  /**
   * \brief T is a numerical field type.
   * \details T can be any arithmetic, complex, or custom number type in which certain traits in
   * interface::number_traits are defined and typical math operations (+, -, *, /, and ==) are also defined.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept number =
    interface::number_traits<std::decay_t<T>>::is_specialized and
    std::regular<T> and
    requires(const std::remove_reference_t<T>& t) {
        { t + t } -> std::convertible_to<const std::decay_t<T>&>;
        { t - t } -> std::convertible_to<const std::decay_t<T>&>;
        { t * t } -> std::convertible_to<const std::decay_t<T>&>;
        { t / t } -> std::convertible_to<const std::decay_t<T>&>;
    };
#else
  constexpr bool number =
    interface::number_traits<std::decay_t<T>>::is_specialized;
#endif


}


#endif
