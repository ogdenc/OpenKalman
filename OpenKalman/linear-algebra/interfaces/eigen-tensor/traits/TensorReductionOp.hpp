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
 * \brief Type traits as applied to Eigen::TensorReductionOp.
 */

#ifndef OPENKALMAN_EIGEN_TRAITS_TENSORREDUCTIONOP_HPP
#define OPENKALMAN_EIGEN_TRAITS_TENSORREDUCTIONOP_HPP


namespace OpenKalman::interface
{
  template<typename Op, typename Dims, typename XprType, template<typename> typename MakePointer>
  struct indexible_object_traits<Eigen::TensorReductionOp<Op, Dims, XprType, MakePointer>>
    : Eigen3::indexible_object_traits_tensor_base<Eigen::TensorReductionOp<Op, Dims, XprType, MakePointer>>
  {
    template<typename Arg, typename N>
    static constexpr auto get_index_descriptor(const Arg& arg, N n)
    {
      if constexpr (values::fixed<N>)
        return std::integral_constant<std::size_t, Eigen::internal::get<n, typename Dims::Base>::value>{};
      else
        return static_cast<std::size_t>(arg.dimension[n]);
    }

    template<std::size_t i, typename Arg>
    static decltype(auto) get_nested_object(Arg&& arg)
    {
      if constexpr (i == 0)
        return std::forward<Arg>(arg).expression();
      else
        return std::forward<Arg>(arg).reducer();
      static_assert(i <= 1);
    }


    template<typename Arg>
    static constexpr auto get_constant(const Arg& arg)
    {
      return constant_coefficient {arg}; /// \todo fix this
    }

    // get_constant_diagonal() not defined

    static constexpr bool is_writable = false;

    // data() not defined

    // layout not defined

  };

}

#endif
