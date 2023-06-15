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

#ifndef OPENKALMAN_EIGEN3_TRAITS_SOLVE_HPP
#define OPENKALMAN_EIGEN3_TRAITS_SOLVE_HPP

#include <type_traits>


namespace OpenKalman::interface
{
#ifndef __cpp_concepts
  template<typename Decomposition, typename RhsType>
  struct IndexTraits<Eigen::Solve<Decomposition, RhsType>>
    : detail::IndexTraits_Eigen_default<Eigen::Ref<Eigen::Solve<Decomposition, RhsType>>> {};
#endif


  template<typename Decomposition, typename RhsType>
  struct Dependencies<Eigen::Solve<Decomposition, RhsType>>
  {
    static constexpr bool has_runtime_parameters = false;
    using type = std::tuple<const Decomposition&, const RhsType&>;

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

  };


} // namespace OpenKalman::interface

#endif //OPENKALMAN_EIGEN3_TRAITS_SOLVE_HPP
