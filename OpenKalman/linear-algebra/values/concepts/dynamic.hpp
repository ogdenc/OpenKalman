/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2024 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definition for \value::dynamic.
 */

#ifndef OPENKALMAN_VALUE_DYNAMIC_HPP
#define OPENKALMAN_VALUE_DYNAMIC_HPP

#include <type_traits>
#include "number.hpp"
#include "fixed.hpp"

namespace OpenKalman::value
{
#ifndef __cpp_concepts
  namespace internal
  {
    // This functions is also used in value::to_number

    template<typename T, typename = void>
    struct is_dynamic : std::false_type {};

    template<typename T>
    struct is_dynamic<T, std::enable_if_t<value::number<typename std::invoke_result<T>::type>>>
      : std::true_type {};

  } // namespace internal
#endif


  /**
   * \brief T is a value::value that is not determinable at compile time.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept dynamic = (not value::fixed<T>) and (value::number<T> or requires(std::decay_t<T> t){ {t()} -> value::number; });
#else
  constexpr bool dynamic =
    (not value::fixed<T>) and (value::number<T> or internal::is_dynamic<std::decay_t<T>>::value);
#endif


} // namespace OpenKalman::value

#endif //OPENKALMAN_VALUE_DYNAMIC_HPP
