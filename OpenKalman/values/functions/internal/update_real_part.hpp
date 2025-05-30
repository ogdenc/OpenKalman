/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2024 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \internal
 * \file
 * \brief Definition for \ref update_real_part function.
 */

#ifndef OPENKALMAN_UPDATE_REAL_PART_HPP
#define OPENKALMAN_UPDATE_REAL_PART_HPP

#include <type_traits>
#include "values/concepts/complex.hpp"
#include "values/concepts/number.hpp"
#include "make_complex_number.hpp"


namespace OpenKalman::values::internal
{
  /**
   * \internal
   * \brief Update only the real part of a (potentially) complex number, leaving the imaginary part unchanged.
   * \param arg A potentially complex number to update.
   * \param re A real value.
   */
#ifdef __cpp_concepts
  constexpr values::number decltype(auto)
  update_real_part(values::number auto&& arg, values::number auto&& re) requires (not values::complex<decltype(re)>)
#else
  template<typename T, typename Re, std::enable_if_t<values::number<T> and values::number<Re> and not values::complex<Re>, int> = 0>
  constexpr decltype(auto) update_real_part(T&& arg, Re&& re)
#endif
  {
    using Arg = std::decay_t<decltype(arg)>;
    if constexpr (values::complex<Arg>)
    {
      auto im = values::imag(std::forward<decltype(arg)>(arg));
      using R = std::decay_t<decltype(im)>;
      return values::internal::make_complex_number<Arg>(static_cast<R>(std::forward<decltype(re)>(re)), std::move(im));
    }
    else return std::forward<decltype(re)>(re);
  }


} // namespace OpenKalman::values::internal

#endif //OPENKALMAN_UPDATE_REAL_PART_HPP
