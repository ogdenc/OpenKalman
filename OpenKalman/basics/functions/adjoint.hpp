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
 * \brief Definition of adjoint function.
 */

#ifndef OPENKALMAN_ADJOINT_HPP
#define OPENKALMAN_ADJOINT_HPP

#include<complex>


namespace OpenKalman
{
  /**
   * \brief Take the adjoint of a matrix
   * \tparam Arg The matrix
   */
#ifdef __cpp_concepts
  template<indexible Arg> requires (max_tensor_order_v<Arg> <= 2)
#else
  template<typename Arg, std::enable_if_t<indexible<Arg> and (max_tensor_order_v<Arg> <= 2), int> = 0>
#endif
  constexpr decltype(auto) adjoint(Arg&& arg) noexcept
  {
    if constexpr (hermitian_matrix<Arg>)
    {
      return std::forward<Arg>(arg);
    }
    else if constexpr (diagonal_matrix<Arg> and square_shaped<Arg>)
    {
      return conjugate(std::forward<Arg>(arg));
    }
    else if constexpr (zero<Arg>)
    {
      return transpose(std::forward<Arg>(arg));
    }
    else if constexpr (constant_matrix<Arg>)
    {
      if constexpr (real_axis_number<constant_coefficient<Arg>>)
        return transpose(std::forward<Arg>(arg));
      else if constexpr (not has_dynamic_dimensions<Arg> and index_dimension_of_v<Arg, 0> == index_dimension_of_v<Arg, 1>)
        return conjugate(std::forward<Arg>(arg));
      else
      {
        constexpr std::make_index_sequence<std::max({index_count_v<Arg>, 2_uz}) - 2_uz> seq;
        return internal::transpose_constant(internal::constexpr_conj(constant_coefficient{arg}), std::forward<Arg>(arg), seq);
      }
    }
    else if constexpr (interface::adjoint_defined_for<Arg, Arg&&>)
    {
      return interface::library_interface<std::decay_t<Arg>>::adjoint(std::forward<Arg>(arg));
    }
    else
    {
      return transpose(conjugate(std::forward<Arg>(arg)));
    }
  }

} // namespace OpenKalman

#endif //OPENKALMAN_ADJOINT_HPP
