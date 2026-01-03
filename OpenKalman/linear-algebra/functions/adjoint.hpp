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
 * \brief Definition of adjoint function.
 */

#ifndef OPENKALMAN_ADJOINT_HPP
#define OPENKALMAN_ADJOINT_HPP

#include "conjugate.hpp"
#include "transpose.hpp"

namespace OpenKalman
{
  /**
   * \brief Take the conjugate-transpose of an \ref indexible_object
   * \details By default, the first two indices are transposed.
   */
#ifdef __cpp_concepts
  template<std::size_t indexa = 0, std::size_t indexb = 1, indexible Arg> requires (indexa < indexb)
#else
  template<std::size_t indexa = 0, std::size_t indexb = 1, typename Arg, std::enable_if_t<
    indexible<Arg> and (indexa < indexb), int> = 0>
#endif
  constexpr decltype(auto)
  adjoint(Arg&& arg)
  {
    if constexpr (hermitian_matrix<Arg>)
    {
      return std::forward<Arg>(arg);
    }
    else if constexpr ((diagonal_matrix<Arg> or constant_object<Arg>) and
      values::size_compares_with<index_dimension_of<Arg, 0>, index_dimension_of<Arg, 1>>)
    {
      return conjugate(std::forward<Arg>(arg));
    }
    else if constexpr (not values::complex<element_type_of_t<Arg>> or values::not_complex<constant_value_of<Arg>>)
    {
      return transpose(std::forward<Arg>(arg));
    }
    else if constexpr (indexb == 1 and interface::matrix_adjoint_defined_for<Arg&&>)
    {
      return interface::library_interface<stdex::remove_cvref_t<Arg>>::adjoint(std::forward<Arg>(arg));
    }
    else if constexpr (interface::adjoint_defined_for<Arg&&, indexa, indexb>)
    {
      return interface::library_interface<stdex::remove_cvref_t<Arg>>::template adjoint<indexa, indexb>(std::forward<Arg>(arg));
    }
    else
    {
      return conjugate(transpose(std::forward<Arg>(arg)));
    }
  }

}

#endif
