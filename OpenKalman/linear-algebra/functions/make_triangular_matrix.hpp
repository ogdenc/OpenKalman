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
   * \tparam t The intended \ref triangle_type of the result.
   * \tparam Arg A general matrix to be made triangular.
   */
#ifdef __cpp_concepts
  template<triangle_type t = triangle_type::lower, indexible Arg> requires
    (t == triangle_type::lower or t == triangle_type::upper or t == triangle_type::diagonal)
  constexpr triangular_matrix<t> decltype(auto)
#else
  template<triangle_type t = triangle_type::lower, typename Arg, std::enable_if_t<indexible<Arg> and
    (t == triangle_type::lower or t == triangle_type::upper or t == triangle_type::diagonal), int> = 0>
  constexpr decltype(auto)
#endif
  make_triangular_matrix(Arg&& arg)
  {
    if constexpr (triangular_matrix<Arg, t>)
    {
      // If Arg is already triangular of type t, pass through to the result
      return std::forward<Arg>(arg);
    }
    else if constexpr (t == triangle_type::diagonal)
    {
      return to_diagonal(diagonal_of(std::forward<Arg>(arg)));
    }
    else if constexpr (triangular_matrix<Arg, triangle_type::any> and not triangular_matrix<Arg, t>)
    {
      // Arg is the opposite triangle of t.
      return make_triangular_matrix<triangle_type::diagonal>(std::forward<Arg>(arg));
    }
    else if constexpr (hermitian_adapter<Arg>)
    {
      if constexpr (hermitian_adapter<Arg, static_cast<HermitianAdapterType>(t)>)
        return make_triangular_matrix<t>(nested_object(std::forward<Arg>(arg)));
      else
        return make_triangular_matrix<t>(conjugate_transpose(nested_object(std::forward<Arg>(arg))));
    }
    else if constexpr (interface::make_triangular_matrix_defined_for<Arg, t, Arg&&>)
    {
      using Traits = interface::library_interface<stdex::remove_cvref_t<Arg>>;
      return Traits::template make_triangular_matrix<t>(std::forward<Arg>(arg));
    }
    else // Default behavior if interface function not defined:
    {
      return TriangularAdapter<Arg, t> {std::forward<Arg>(arg)};
    }
  }

}

#endif
