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
  struct indexible_object_traits<Eigen::PermutationWrapper<IndicesType>>
    : Eigen3::indexible_object_traits_base<Eigen::PermutationWrapper<IndicesType>>
  {
  private:

    using Base = Eigen3::indexible_object_traits_base<Eigen::PermutationWrapper<IndicesType>>;

  public:

    using dependents = std::tuple<typename IndicesType::Nested>;

    static constexpr bool has_runtime_parameters = false;


    template<typename Arg>
    static decltype(auto) nested_object(Arg&& arg)
    {
      return std::forward<Arg>(arg).indices();
    }

    // get_constant() not defined

    // get_constant_diagonal() not defined
  };

} // namespace OpenKalman::interface

#endif //OPENKALMAN_EIGEN_TRAITS_PERMUTATIONWRAPPER_HPP
