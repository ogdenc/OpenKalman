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
 * \internal
 * \file
 * \brief Definition for \ref update_real_part function.
 */

#ifndef OPENKALMAN_UPDATE_REAL_PART_HPP
#define OPENKALMAN_UPDATE_REAL_PART_HPP

#include "values/concepts/value.hpp"
#include "values/concepts/complex.hpp"
#include "values/functions/internal/make_complex_number.hpp"

namespace OpenKalman::values::internal
{
  /**
   * \internal
   * \brief Update only the real part of a (potentially) complex \ref values::number, leaving the imaginary part unchanged.
   * \param t A potentially complex \ref values::value to update.
   * \param re A real \ref values::value.
   * \returns A new \ref values::value, which will be \ref complex only if t is complex.
   */
#ifdef __cpp_concepts
  template<value T, value Re> requires (not complex<Re>) and std::common_with<real_type_of_t<T>, number_type_of_t<Re>>
  constexpr values::value decltype(auto)
#else
  template<typename T, typename Re, std::enable_if_t<value<T> and value<Re> and not complex<Re>, int> = 0>
  constexpr decltype(auto)
#endif
  update_real_part(T t, Re&& re)
  {
    if constexpr (complex<T>)
    {
      using U = std::common_type_t<real_type_of_t<T>, Re>;
      return values::internal::make_complex_number<U>(std::forward<Re>(re), values::imag(std::move(t)));
    }
    else
    {
      return std::forward<Re>(re);
    }
  }


} // namespace OpenKalman::values::internal

#endif //OPENKALMAN_UPDATE_REAL_PART_HPP
