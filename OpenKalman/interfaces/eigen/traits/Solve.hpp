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
 * \brief Type traits as applied to Eigen::Solve.
 */

#ifndef OPENKALMAN_EIGEN_TRAITS_SOLVE_HPP
#define OPENKALMAN_EIGEN_TRAITS_SOLVE_HPP

#include <type_traits>


namespace OpenKalman::interface
{
  template<typename Decomposition, typename RhsType>
  struct indexible_object_traits<Eigen::Solve<Decomposition, RhsType>>
    : Eigen3::indexible_object_traits_base<Eigen::Solve<Decomposition, RhsType>>
  {
  private:

    using Base = Eigen3::indexible_object_traits_base<Eigen::Solve<Decomposition, RhsType>>;

  public:

    using dependents = std::tuple<const Decomposition&, const RhsType&>;

    static constexpr bool has_runtime_parameters = false;

    template<std::size_t i, typename Arg>
    static decltype(auto) get_nested_matrix(Arg&& arg)
    {
      static_assert(i < 2);
      if constexpr (i == 0)
        return std::forward<Arg>(arg).dec();
      else
        return std::forward<Arg>(arg).rhs();
    }

    // Eigen::Solve can never be self-contained.

    // get_constant() not defined

    // get_constant_diagonal() not defined
  };


} // namespace OpenKalman::interface

#endif //OPENKALMAN_EIGEN_TRAITS_SOLVE_HPP
