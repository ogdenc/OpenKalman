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
#include "linear-algebra/values/concepts/complex.hpp"
#include "linear-algebra/values/concepts/number.hpp"
#include "make_complex_number.hpp"


namespace OpenKalman::value::internal
{
  /**
   * \internal
   * \brief Update only the real part of a (potentially) complex number, leaving the imaginary part unchanged.
   * \param arg A potentially complex number to update.
   * \param re A real value.
   */
#ifdef __cpp_concepts
  constexpr value::number decltype(auto)
  update_real_part(value::number auto&& arg, value::number auto&& re) requires (not value::complex<decltype(re)>)
#else
  template<typename T, typename Re, std::enable_if_t<value::number<T> and value::number<Re> and not value::complex<Re>, int> = 0>
  constexpr decltype(auto) update_real_part(T&& arg, Re&& re)
#endif
  {
    using Arg = std::decay_t<decltype(arg)>;
    if constexpr (value::complex<Arg>)
    {
      auto im = value::imag(std::forward<decltype(arg)>(arg));
      using R = std::decay_t<decltype(im)>;
      return value::internal::make_complex_number<Arg>(static_cast<R>(std::forward<decltype(re)>(re)), std::move(im));
    }
    else return std::forward<decltype(re)>(re);
  }


} // namespace OpenKalman::value::internal

#endif //OPENKALMAN_UPDATE_REAL_PART_HPP
