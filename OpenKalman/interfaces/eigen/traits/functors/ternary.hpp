/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2024 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Trait details for Eigen ternary functors.
 */

#ifndef OPENKALMAN_EIGEN_TRAITS_FUNCTORS_TERNARY_HPP
#define OPENKALMAN_EIGEN_TRAITS_FUNCTORS_TERNARY_HPP

#include <type_traits>

namespace OpenKalman::Eigen3
{

  // Default ternary functor traits
  template<typename Operation, typename Arg1, typename Arg2, typename Arg3>
  struct TernaryFunctorTraits
  {
    /**
     * \brief Return a scalar constant or std::monostate
     * \tparam is_diag True if \ref constant_diagonal_coefficient, false if \ref constant_coefficient.
     * \return \ref scalar_constant
     */
    template<bool is_diag, typename Arg>
    static constexpr auto get_constant(const Arg& arg)
    {
      return std::monostate {};
    }

    template<TriangleType t>
    static constexpr bool is_triangular = false;

    static constexpr bool is_hermitian = false;
  };


} // namespace OpenKalman::Eigen3

#endif //OPENKALMAN_EIGEN_TRAITS_FUNCTORS_TERNARY_HPP
