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
  template<TriangleType t = TriangleType::lower, square_matrix<Likelihood::maybe> Arg> requires
    (t == TriangleType::lower or t == TriangleType::upper or t == TriangleType::diagonal)
  constexpr triangular_matrix<t> decltype(auto)
#else
  template<TriangleType t = TriangleType::lower, typename Arg, std::enable_if_t<square_matrix<Arg, Likelihood::maybe> and
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
    else if constexpr (triangular_matrix<Arg, TriangleType::any, Likelihood::maybe> and not triangular_matrix<Arg, t, Likelihood::maybe>)
    {
      // Arg is the opposite triangle of t.
      return make_triangular_matrix<TriangleType::diagonal>(std::forward<Arg>(arg));
    }
    else if constexpr (triangular_adapter<Arg>)
    {
      // If Arg is a triangular adapter but was not known to be square at compile time, return a result guaranteed to be square triangular.
      return make_triangular_matrix<t>(nested_matrix(std::forward<Arg>(arg)));
    }
    else if constexpr (hermitian_adapter<Arg>)
    {
      if constexpr (hermitian_adapter<Arg, static_cast<HermitianAdapterType>(t)>)
        return make_triangular_matrix<t>(nested_matrix(std::forward<Arg>(arg)));
      else
        return make_triangular_matrix<t>(adjoint(nested_matrix(std::forward<Arg>(arg))));
    }
    else if constexpr (interface::make_triangular_matrix_defined_for<std::decay_t<Arg>, t, Arg&&>)
    {
      using Traits = interface::library_interface<std::decay_t<Arg>>;
      auto new_t {Traits::template make_triangular_matrix<t>(std::forward<Arg>(arg))};
      static_assert(triangular_matrix<decltype(new_t), t>, "make_triangular_matrix interface must return a triangular matrix");
      return new_t;
    }
    else // Default behavior if interface function not defined:
    {
      using pArg = std::conditional_t<std::is_lvalue_reference_v<Arg>, Arg, std::remove_reference_t<decltype(make_self_contained(arg))>>;
      return TriangularMatrix<pArg, t> {std::forward<Arg>(arg)};
    }
  }

} // namespace OpenKalman

#endif //OPENKALMAN_MAKE_TRIANGULAR_MATRIX_HPP
