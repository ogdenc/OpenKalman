/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2023 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Type traits as applied to Eigen::TensorFixedSize.
 */

#ifndef OPENKALMAN_EIGEN_TRAITS_TENSORFIXEDSIZE_HPP
#define OPENKALMAN_EIGEN_TRAITS_TENSORFIXEDSIZE_HPP

#include <type_traits>


namespace OpenKalman::interface
{
  template<typename S, typename Dims, int options, typename IndexType>
  struct IndexibleObjectTraits<Eigen::TensorFixedSize<S, Dims, options, IndexType>>
    : Eigen3::IndexibleObjectTraitsBase<Eigen::TensorFixedSize<S, Dims, options, IndexType>>
  {
    static constexpr std::size_t max_indices = Dims::count;

    template<std::size_t N, typename Arg>
    static constexpr auto get_index_descriptor(const Arg& arg)
    {
      return std::integral_constant<std::size_t, Eigen::internal::get<N, typename Dims::Base>::value>{};
    }

    static constexpr bool has_runtime_parameters = true;

    using type = std::tuple<>;

    // get_nested_matrix() not defined

    // convert_to_self_contained() not defined

    // get_constant() not defined

    // get_constant_diagonal() not defined
  };

} // namespace OpenKalman::interface

#endif //OPENKALMAN_EIGEN_TRAITS_TENSORFIXEDSIZE_HPP
