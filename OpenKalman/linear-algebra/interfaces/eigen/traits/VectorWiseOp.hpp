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
 * \brief Type traits as applied to Eigen::VectorwiseOp.
 */

#ifndef OPENKALMAN_EIGEN_TRAITS_VECTORWISEOP_HPP
#define OPENKALMAN_EIGEN_TRAITS_VECTORWISEOP_HPP

#include <type_traits>


namespace OpenKalman::interface
{
  template<typename ExpressionType, int Direction>
  struct object_traits<Eigen::VectorwiseOp<ExpressionType, Direction>>
  {
    using scalar_type = scalar_type_of_t<ExpressionType>;

    template<typename Arg>
    static constexpr auto count_indices(const Arg& arg) { return std::integral_constant<std::size_t, 2>{}; }

    template<typename Arg, typename N>
    static constexpr auto get_pattern_collection(const Arg& arg, N n)
    {
      return OpenKalman::get_pattern_collection(arg._expression(), n);
    }

    template<typename Arg>
    static decltype(auto) nested_object(Arg&& arg)
    {
      return std::forward<Arg>(arg)._expression();
    }

    template<typename Arg>
    static constexpr auto get_constant(const Arg& arg)
    {
      return constant_value {arg._expression()};
    }

    template<applicability b>
    static constexpr bool one_dimensional = OpenKalman::one_dimensional<ExpressionType, b>;

    template<applicability b>
    static constexpr bool is_square = square_shaped<ExpressionType, b>;

    // No get or set defined.
  };

}

#endif
