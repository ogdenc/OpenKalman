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
   * \return AA<sup>T</sup> (if A is lower \ref triangular_matrix) or otherwise A<sup>T</sup>A.
   */
#ifdef __cpp_concepts
  template<triangular_matrix A>
  constexpr hermitian_matrix decltype(auto)
#else
  template<typename A, std::enable_if_t<triangular_matrix<A>, int> = 0>
  constexpr decltype(auto)
#endif
  cholesky_square(A&& a) noexcept
  {
    if constexpr (zero<A> or identity_matrix<A>)
    {
      return std::forward<A>(a);
    }
    else if constexpr (diagonal_matrix<A>)
    {
      return to_diagonal(n_ary_operation([](const auto x){
        if constexpr (complex_number<decltype(x)>) { using std::conj; return x * conj(x); }
        else return x * x;
      }, diagonal_of(std::forward<A>(a))));
    }
    else
    {
      constexpr auto triangle_type = triangle_type_of_v<A>;
      auto prod {make_dense_object(adjoint(a))};
      constexpr bool on_the_right = triangular_matrix<A, TriangleType::upper>;
      interface::library_interface<std::decay_t<A>>::template contract_in_place<on_the_right>(prod, std::forward<A>(a));
      return SelfAdjointMatrix<decltype(prod), triangle_type> {std::move(prod)};
    }
  }


} // namespace OpenKalman


#endif //OPENKALMAN_CHOLESKY_SQUARE_HPP
