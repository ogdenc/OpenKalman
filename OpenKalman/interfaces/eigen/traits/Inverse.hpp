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
 * \brief Type traits as applied to Eigen::Inverse.
 */

#ifndef OPENKALMAN_EIGEN_TRAITS_INVERSE_HPP
#define OPENKALMAN_EIGEN_TRAITS_INVERSE_HPP

#include <type_traits>


namespace OpenKalman::interface
{
  template<typename XprType>
  struct indexible_object_traits<Eigen::Inverse<XprType>>
    : Eigen3::indexible_object_traits_base<Eigen::Inverse<XprType>>
  {
  private:

    using Base = Eigen3::indexible_object_traits_base<Eigen::Inverse<XprType>>;

  public:

    using dependents = std::tuple<typename Eigen::Inverse<XprType>::XprTypeNested>;

    static constexpr bool has_runtime_parameters = false;


    template<typename Arg>
    static decltype(auto) nested_object(Arg&& arg)
    {
      return std::forward<Arg>(arg).nestedExpression();
    }


    template<typename Arg>
    static auto convert_to_self_contained(Arg&& arg)
    {
      using N = Eigen::Inverse<equivalent_self_contained_t<XprType>>;
      if constexpr (not std::is_lvalue_reference_v<typename N::XprTypeNested>)
        return N {make_self_contained(arg.nestedExpression())};
      else
        return make_dense_object(std::forward<Arg>(arg));
    }

    // get_constant() not defined

    // get_constant_diagonal() not defined
  };

} // namespace OpenKalman::interface

#endif //OPENKALMAN_EIGEN_TRAITS_INVERSE_HPP
