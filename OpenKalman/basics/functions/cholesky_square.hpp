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
 * \file
 * \brief Definition of cholesky_square.
 */

#ifndef OPENKALMAN_CHOLESKY_SQUARE_HPP
#define OPENKALMAN_CHOLESKY_SQUARE_HPP

namespace OpenKalman
{
  /**
   * \brief Take the Cholesky square of a \ref triangular_matrix.
   * \tparam A A square matrix.
   * \return AA<sup>*</sup> (if A is lower \ref triangular_matrix) or otherwise A<sup>*</sup>A.
   */
#ifdef __cpp_concepts
  template<triangular_matrix A> requires square_shaped<A, Qualification::depends_on_dynamic_shape>
  constexpr hermitian_matrix decltype(auto)
#else
  template<typename A, std::enable_if_t<triangular_matrix<A> and square_shaped<A, Qualification::depends_on_dynamic_shape>, int> = 0>
  constexpr decltype(auto)
#endif
  cholesky_square(A&& a) noexcept
  {
    if constexpr (not square_shaped<A>)
      if (not is_square_shaped(a)) throw std::invalid_argument {"Argument to cholesky_square must be a square matrix"};

    if constexpr (zero<A> or identity_matrix<A>)
    {
      return std::forward<A>(a);
    }
    else if constexpr (diagonal_matrix<A>)
    {
      return to_diagonal(n_ary_operation([](const auto x){ return x * internal::constexpr_conj(x); }, diagonal_of(std::forward<A>(a))));
    }
    else if constexpr (triangular_matrix<A, TriangleType::upper>)
    {
      return make_hermitian_matrix<HermitianAdapterType::upper>(contract(adjoint(a), a));
    }
    else
    {
      return make_hermitian_matrix<HermitianAdapterType::lower>(contract(a, adjoint(a)));
    }
  }


} // namespace OpenKalman


#endif //OPENKALMAN_CHOLESKY_SQUARE_HPP
