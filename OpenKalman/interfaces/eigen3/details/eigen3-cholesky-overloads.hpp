/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2022 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definitions of Cholesky_square and Cholesky_factor for Eigen3 matrices
 */

#ifndef OPENKALMAN_EIGEN3_CHOLESKY_HPP
#define OPENKALMAN_EIGEN3_CHOLESKY_HPP


namespace OpenKalman
{
  /**
   * \brief Take the Cholesky square of a diagonal native Eigen matrix.
   * \tparam D A native Eigen diagonal matrix.
   * \return dd<sup>T</sup>
   */
#ifdef __cpp_concepts
  template<native_eigen_matrix D> requires has_dynamic_dimensions<D> or diagonal_matrix<D>
#else
  template<typename D, std::enable_if_t<
    native_eigen_matrix<D> and (has_dynamic_dimensions<D> or diagonal_matrix<D>), int> = 0>
#endif
  constexpr decltype(auto)
  Cholesky_square(D&& d) noexcept
  {
    if constexpr ((has_dynamic_dimensions<D> and not diagonal_matrix<D>) or one_by_one_matrix<D>)
    {
      if constexpr (has_dynamic_dimensions<D> and not diagonal_matrix<D>)
        assert(get_dimensions_of<0>(d) == 1 and get_dimensions_of<1>(d) == 1);

      return std::forward<D>(d).array().square().matrix();
    }
    else if constexpr (identity_matrix<D> or zero_matrix<D>)
    {
      return std::forward<D>(d);
    }
    else
    {
      auto n = std::forward<D>(d).diagonal().array().square().matrix();
      return DiagonalMatrix<decltype(n)> {std::move(n)};
    }
  }


  /**
   * \brief Take the Cholesky factor of a diagonal native Eigen matrix.
   * \tparam D A native Eigen diagonal matrix.
   * \return e, where ee<sup>T</sup> = d.
   */
#ifdef __cpp_concepts
  template<TriangleType = TriangleType::diagonal, native_eigen_matrix D>
  requires has_dynamic_dimensions<D> or diagonal_matrix<D>
#else
  template<TriangleType = TriangleType::diagonal,
    typename D, std::enable_if_t<native_eigen_matrix<D> and (has_dynamic_dimensions<D> or diagonal_matrix<D>), int> = 0>
#endif
  constexpr decltype(auto)
  Cholesky_factor(D&& d) noexcept
  {
    if constexpr((has_dynamic_dimensions<D> and not diagonal_matrix<D>) or one_by_one_matrix<D>)
    {
      if constexpr (has_dynamic_dimensions<D> and not diagonal_matrix<D>)
        assert(get_dimensions_of<0>(d) == 1 and get_dimensions_of<1>(d) == 1);

      return std::forward<D>(d).cwiseSqrt();
    }
    else if constexpr(identity_matrix<D> or zero_matrix<D>)
    {
      return std::forward<D>(d);
    }
    else
    {
      auto n = std::forward<D>(d).diagonal().cwiseSqrt();
      return DiagonalMatrix<decltype(n)> {std::move(n)};
    }
  }


}


#endif //OPENKALMAN_EIGEN3_CHOLESKY_HPP
