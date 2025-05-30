/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2022-2023 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definition of conjugate function.
 */

#ifndef OPENKALMAN_CONJUGATE_HPP
#define OPENKALMAN_CONJUGATE_HPP

#include<complex>


namespace OpenKalman
{
  /**
   * \brief Take the conjugate of a matrix
   * \tparam Arg The matrix
   */
#ifdef __cpp_concepts
  template<indexible Arg>
#else
  template<typename Arg, std::enable_if_t<indexible<Arg>, int> = 0>
#endif
  constexpr decltype(auto) conjugate(Arg&& arg)
  {
    if constexpr (not values::complex<scalar_type_of_t<Arg>> or zero<Arg> or identity_matrix<Arg>)
    {
      return std::forward<Arg>(arg);
    }
    else if constexpr (constant_matrix<Arg>)
    {
      if constexpr (values::not_complex<constant_coefficient<Arg>>)
        return std::forward<Arg>(arg);
      else
        return make_constant(values::conj(constant_coefficient{arg}), std::forward<Arg>(arg));
    }
    else if constexpr (constant_diagonal_matrix<Arg>)
    {
      if constexpr (values::not_complex<constant_diagonal_coefficient<Arg>>)
        return std::forward<Arg>(arg);
      else
        return to_diagonal(make_constant(values::conj(constant_diagonal_coefficient{arg}),
          diagonal_of(std::forward<Arg>(arg))));
    }
    else if constexpr (diagonal_matrix<Arg>)
    {
      return to_diagonal(conjugate(diagonal_of(std::forward<Arg>(arg))));
    }
    else
    {
      return interface::library_interface<std::decay_t<Arg>>::conjugate(std::forward<Arg>(arg));
    }
  }

} // namespace OpenKalman

#endif //OPENKALMAN_CONJUGATE_HPP
