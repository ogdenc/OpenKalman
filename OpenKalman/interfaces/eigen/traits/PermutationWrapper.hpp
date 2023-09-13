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
 * \brief Type traits as applied to Eigen::PermutationWrapper.
 */

#ifndef OPENKALMAN_EIGEN_TRAITS_PERMUTATIONWRAPPER_HPP
#define OPENKALMAN_EIGEN_TRAITS_PERMUTATIONWRAPPER_HPP

#include <type_traits>


namespace OpenKalman::interface
{
  template<typename IndicesType>
  struct IndexibleObjectTraits<Eigen::PermutationWrapper<IndicesType>>
    : Eigen3::IndexibleObjectTraitsBase<Eigen::PermutationWrapper<IndicesType>>
  {
    static constexpr bool has_runtime_parameters = false;

    using type = std::tuple<typename IndicesType::Nested>;

    template<std::size_t i, typename Arg>
    static decltype(auto) get_nested_matrix(Arg&& arg)
    {
      static_assert(i == 0);
      return std::forward<Arg>(arg).indices();
    }

    template<typename Arg>
    static auto convert_to_self_contained(Arg&& arg)
    {
      using NewIndicesType = equivalent_self_contained_t<IndicesType>;
      if constexpr (not std::is_lvalue_reference_v<typename NewIndicesType::Nested>)
        return Eigen::PermutationWrapper<NewIndicesType> {make_self_contained(arg.nestedExpression()), arg.functor()};
      else
        return make_dense_writable_matrix_from(std::forward<Arg>(arg));
    }

    // get_constant() not defined

    // get_constant_diagonal() not defined
  };

} // namespace OpenKalman::interface

#endif //OPENKALMAN_EIGEN_TRAITS_PERMUTATIONWRAPPER_HPP
