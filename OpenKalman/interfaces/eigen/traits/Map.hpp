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
 * \brief Type traits as applied to Eigen::Map.
 */

#ifndef OPENKALMAN_EIGEN_TRAITS_MAP_HPP
#define OPENKALMAN_EIGEN_TRAITS_MAP_HPP

#include <type_traits>


namespace OpenKalman::interface
{
  template<typename PlainObjectType, int MapOptions, typename StrideType>
  struct indexible_object_traits<Eigen::Map<PlainObjectType, MapOptions, StrideType>>
    : Eigen3::indexible_object_traits_base<Eigen::Map<PlainObjectType, MapOptions, StrideType>>
  {
  private:

    using Xpr = Eigen::Map<PlainObjectType, MapOptions, StrideType>;
    using Base = Eigen3::indexible_object_traits_base<Xpr>;

  public:

    template<typename Arg>
    static decltype(auto) nested_object(Arg&& arg)
    {
      return *std::forward<Arg>(arg).data();
    }


    // get_constant() not defined

    // get_constant_diagonal() not defined


    static constexpr Layout layout = std::is_same_v<StrideType, Eigen::Stride<0, 0>> ? layout_of_v<PlainObjectType> : Layout::stride;

  };

} // namespace OpenKalman::interface

#endif //OPENKALMAN_EIGEN_TRAITS_MAP_HPP
