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
 * \brief Type traits as applied to Eigen::PermutationMatrix.
 */

#ifndef OPENKALMAN_EIGEN_TRAITS_PERMUTATIONMATRIX_HPP
#define OPENKALMAN_EIGEN_TRAITS_PERMUTATIONMATRIX_HPP

#include <type_traits>


namespace OpenKalman::interface
{
  template<int SizeAtCompileTime, int MaxSizeAtCompileTime, typename StorageIndex>
  struct IndexibleObjectTraits<Eigen::PermutationMatrix<SizeAtCompileTime, MaxSizeAtCompileTime, StorageIndex>>
    : Eigen3::IndexibleObjectTraitsBase<Eigen::PermutationMatrix<SizeAtCompileTime, MaxSizeAtCompileTime, StorageIndex>>
  {
    static constexpr bool has_runtime_parameters = false;

    using type = std::tuple<
      typename Eigen::PermutationMatrix<SizeAtCompileTime, MaxSizeAtCompileTime, StorageIndex>::IndicesType>;

    template<std::size_t i, typename Arg>
    static decltype(auto) get_nested_matrix(Arg&& arg)
    {
      static_assert(i == 0);
      return std::forward<Arg>(arg).indices();
    }

    // PermutationMatrix is always self-contained.

    // get_constant() not defined

    // get_constant_diagonal() not defined
  };

} // namespace OpenKalman::interface

#endif //OPENKALMAN_EIGEN_TRAITS_PERMUTATIONMATRIX_HPP
