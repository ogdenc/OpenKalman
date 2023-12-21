/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2023 Christopher Lee Ogden <ogden@gatech.edu>
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

namespace OpenKalman::internal
{
  /**
   * \internal
   * \brief Update only the real part of a (potentially) complex number, leaving the imaginary part unchanged.
   * \param arg A potentially complex number to update.
   * \param re A real value.
   */
#ifdef __cpp_concepts
  constexpr scalar_type decltype(auto)
  update_real_part(scalar_type auto&& arg, scalar_type auto&& re) requires (not complex_number<decltype(re)>)
#else
  template<typename T, typename Re, std::enable_if_t<scalar_type<T> and scalar_type<Re> and not complex_number<Re>, int> = 0>
  constexpr decltype(auto) update_real_part(T&& arg, Re&& re)
#endif
  {
    using Arg = std::decay_t<decltype(arg)>;
    if constexpr (complex_number<Arg>)
    {
      auto im = constexpr_imag(std::forward<decltype(arg)>(arg));
      using R = std::decay_t<decltype(im)>;
      return make_complex_number<Arg>(static_cast<R>(std::forward<decltype(re)>(re)), std::move(im));
    }
    else return std::forward<decltype(re)>(re);
  }


} // namespace OpenKalman::internal

#endif //OPENKALMAN_UPDATE_REAL_PART_HPP
