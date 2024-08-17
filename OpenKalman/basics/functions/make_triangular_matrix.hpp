/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2022 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definitions for \ref make_triangular_matrix.
 */

#ifndef OPENKALMAN_MAKE_TRIANGULAR_MATRIX_HPP
#define OPENKALMAN_MAKE_TRIANGULAR_MATRIX_HPP

namespace OpenKalman
{
  /**
   * \brief Create a \ref triangular_matrix from a general matrix.
   * \tparam t The intended \ref TriangleType of the result.
   * \tparam Arg A general matrix to be made triangular.
   */
#ifdef __cpp_concepts
  template<TriangleType t = TriangleType::lower, indexible Arg> requires
    (t == TriangleType::lower or t == TriangleType::upper or t == TriangleType::diagonal)
  constexpr triangular_matrix<t> decltype(auto)
#else
  template<TriangleType t = TriangleType::lower, typename Arg, std::enable_if_t<indexible<Arg> and
    (t == TriangleType::lower or t == TriangleType::upper or t == TriangleType::diagonal), int> = 0>
  constexpr decltype(auto)
#endif
  make_triangular_matrix(Arg&& arg)
  {
    if constexpr (triangular_matrix<Arg, t>)
    {
      // If Arg is already triangular of type t, pass through to the result
      return std::forward<Arg>(arg);
    }
    else if constexpr (t == TriangleType::diagonal)
    {
      return to_diagonal(diagonal_of(std::forward<Arg>(arg)));
    }
    else if constexpr (triangular_matrix<Arg, TriangleType::any> and not triangular_matrix<Arg, t>)
    {
      // Arg is the opposite triangle of t.
      return make_triangular_matrix<TriangleType::diagonal>(std::forward<Arg>(arg));
    }
    else if constexpr (hermitian_adapter<Arg>)
    {
      if constexpr (hermitian_adapter<Arg, static_cast<HermitianAdapterType>(t)>)
        return make_triangular_matrix<t>(nested_object(std::forward<Arg>(arg)));
      else
        return make_triangular_matrix<t>(adjoint(nested_object(std::forward<Arg>(arg))));
    }
    else if constexpr (interface::make_triangular_matrix_defined_for<Arg, t, Arg&&>)
    {
      using Traits = interface::library_interface<std::decay_t<Arg>>;
      return Traits::template make_triangular_matrix<t>(std::forward<Arg>(arg));
    }
    else // Default behavior if interface function not defined:
    {
      return TriangularMatrix<Arg, t> {std::forward<Arg>(arg)};
    }
  }

} // namespace OpenKalman

#endif //OPENKALMAN_MAKE_TRIANGULAR_MATRIX_HPP
