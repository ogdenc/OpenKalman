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
 * \brief Traits for Eigen::CwiseTernaryOp.
 */

#ifndef OPENKALMAN_EIGEN_TRAITS_CWISETERNARYOP_HPP
#define OPENKALMAN_EIGEN_TRAITS_CWISETERNARYOP_HPP

#include <type_traits>


namespace OpenKalman::interface
{
  template<typename TernaryOp, typename Arg1, typename Arg2, typename Arg3>
  struct object_traits<Eigen::CwiseTernaryOp<TernaryOp, Arg1, Arg2, Arg3>>
    : Eigen3::object_traits_base<Eigen::CwiseTernaryOp<TernaryOp, Arg1, Arg2, Arg3>>
  {
  private:

    using Xpr = Eigen::CwiseTernaryOp<TernaryOp, Arg1, Arg2, Arg3>;
    using Base = Eigen3::object_traits_base<Xpr>;

  public:

    template<typename Arg, typename N>
    static constexpr auto
    get_pattern_collection(const Arg& arg, N n)
    {
      if constexpr (square_shaped<Arg1> or square_shaped<Arg2> or square_shaped<Arg3>)
        return internal::most_fixed_pattern(
          OpenKalman::get_pattern_collection<0>(arg.arg1()),
          OpenKalman::get_pattern_collection<0>(arg.arg2()),
          OpenKalman::get_pattern_collection<0>(arg.arg3()),
          OpenKalman::get_pattern_collection<1>(arg.arg1()),
          OpenKalman::get_pattern_collection<1>(arg.arg2()),
          OpenKalman::get_pattern_collection<1>(arg.arg3()));
      else
        return internal::most_fixed_pattern(
          OpenKalman::get_pattern_collection(arg.arg1(), n),
          OpenKalman::get_pattern_collection(arg.arg2(), n),
          OpenKalman::get_pattern_collection(arg.arg3(), n));
    }


    // nested_object not defined


    template<typename Arg>
    static constexpr auto get_constant(const Arg& arg)
    {
      using Traits = Eigen3::TernaryFunctorTraits<TernaryOp, Arg1, Arg2, Arg3>;
      return Traits::template get_constant<false>(arg);
    }

    template<typename Arg>
    static constexpr auto get_constant_diagonal(const Arg& arg)
    {
      using Traits = Eigen3::TernaryFunctorTraits<TernaryOp, Arg1, Arg2, Arg3>;
      return Traits::template get_constant<true>(arg);
    }

    template<applicability b>
    static constexpr bool one_dimensional =
      OpenKalman::one_dimensional<Arg1, values::unbounded_size, applicability::permitted> and
      OpenKalman::one_dimensional<Arg2, values::unbounded_size, applicability::permitted> and
      OpenKalman::one_dimensional<Arg3, values::unbounded_size, applicability::permitted> and
      (b != applicability::guaranteed or
        not has_dynamic_dimensions<Xpr> or
        OpenKalman::one_dimensional<Arg1, b> or
        OpenKalman::one_dimensional<Arg2, b> or
        OpenKalman::one_dimensional<Arg3, b>);


    template<applicability b>
    static constexpr bool is_square =
      square_shaped<Arg1, 2, applicability::permitted> and
      square_shaped<Arg2, 2, applicability::permitted> and
      square_shaped<Arg3, 2, applicability::permitted> and
      (b != applicability::guaranteed or
        not has_dynamic_dimensions<Xpr> or
        square_shaped<Arg1, 2, b> or
        square_shaped<Arg2, 2, b> or
        square_shaped<Arg3, 2, b>);


    template<triangle_type t>
    static constexpr bool triangle_type_value = Eigen3::TernaryFunctorTraits<TernaryOp, Arg1, Arg2, Arg3>::template triangle_type_value<t>;

    static constexpr bool is_triangular_adapter = false;

    static constexpr bool is_hermitian = Eigen3::TernaryFunctorTraits<TernaryOp, Arg1, Arg2, Arg3>::is_hermitian;
  };

}

#endif
