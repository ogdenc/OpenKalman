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
 * \brief Type traits as applied to Eigen::Ref.
 */

#ifndef OPENKALMAN_EIGEN_TRAITS_REF_HPP
#define OPENKALMAN_EIGEN_TRAITS_REF_HPP

#include <type_traits>


namespace OpenKalman::interface
{
  template<typename PlainObjectType, int Options, typename StrideType>
  struct indexible_object_traits<Eigen::Ref<PlainObjectType, Options, StrideType>>
    : Eigen3::indexible_object_traits_base<Eigen::Ref<PlainObjectType, Options, StrideType>>
  {
    template<typename Arg>
    static constexpr auto get_constant(const Arg& arg)
    {
      if constexpr (constant_matrix<PlainObjectType, ConstantType::static_constant>)
        return constant_coefficient<PlainObjectType> {};
      else
        return std::monostate {};
    }

    template<typename Arg>
    static constexpr auto get_constant_diagonal(const Arg& arg)
    {
      if constexpr (constant_diagonal_matrix<PlainObjectType, ConstantType::static_constant>)
        return constant_diagonal_coefficient<PlainObjectType> {};
      else
        return std::monostate {};
    }


    static constexpr Layout layout = std::is_same_v<StrideType, Eigen::Stride<0, 0>> ? layout_of_v<PlainObjectType> : Layout::stride;

  };


} // namespace OpenKalman::interface

#endif //OPENKALMAN_EIGEN_TRAITS_REF_HPP
