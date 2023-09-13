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
 * \brief Type traits as applied to Eigen::VectorBlock.
 */

#ifndef OPENKALMAN_EIGEN_TRAITS_VECTORBLOCK_HPP
#define OPENKALMAN_EIGEN_TRAITS_VECTORBLOCK_HPP

#include <type_traits>


namespace OpenKalman::interface
{
  template<typename VectorType, int Size>
  struct IndexibleObjectTraits<Eigen::VectorBlock<VectorType, Size>>
    : Eigen3::IndexibleObjectTraitsBase<Eigen::VectorBlock<VectorType, Size>>
  {
    static constexpr bool has_runtime_parameters = true;

    using type = std::tuple<typename Eigen::internal::ref_selector<VectorType>::non_const_type>;

    template<std::size_t i, typename Arg>
    static decltype(auto) get_nested_matrix(Arg&& arg)
    {
      static_assert(i == 0);
      return std::forward<Arg>(arg).nestedExpression();
    }

    // Eigen::VectorBlock should always be converted to Matrix

    template<typename Arg>
    static constexpr auto get_constant(const Arg& arg)
    {
      return constant_coefficient {arg.nestedExpression()};
    }
  };


} // namespace OpenKalman::interface

#endif //OPENKALMAN_EIGEN_TRAITS_VECTORBLOCK_HPP
