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
  struct indexible_object_traits<Eigen::CwiseTernaryOp<TernaryOp, Arg1, Arg2, Arg3>>
    : Eigen3::indexible_object_traits_base<Eigen::CwiseTernaryOp<TernaryOp, Arg1, Arg2, Arg3>>
  {
  private:

    using T = Eigen::CwiseTernaryOp<TernaryOp, Arg1, Arg2, Arg3>;
    using Base = Eigen3::indexible_object_traits_base<T>;

  public:

    template<typename Arg, typename N>
    static constexpr auto
    get_vector_space_descriptor(const Arg& arg, N n)
    {
      if constexpr (static_index_value<N>)
      {
        if constexpr (not dynamic_dimension<Arg1, n>)
          return OpenKalman::get_vector_space_descriptor(arg.arg1(), n);
        else if constexpr (not dynamic_dimension<Arg2, n>)
          return OpenKalman::get_vector_space_descriptor(arg.arg2(), n);
        else
          return OpenKalman::get_vector_space_descriptor(arg.arg3(), n);
      }
      else return OpenKalman::get_vector_space_descriptor(arg.arg1(), n);
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

    template<Qualification b>
    static constexpr bool one_dimensional =
      OpenKalman::one_dimensional<Arg1, Qualification::depends_on_dynamic_shape> and
      OpenKalman::one_dimensional<Arg2, Qualification::depends_on_dynamic_shape> and
      OpenKalman::one_dimensional<Arg3, Qualification::depends_on_dynamic_shape> and
      (b != Qualification::unqualified or
        OpenKalman::one_dimensional<Arg1, b> or
        OpenKalman::one_dimensional<Arg2, b> or
        OpenKalman::one_dimensional<Arg3, b>);

    template<TriangleType t>
    static constexpr bool is_triangular = Eigen3::TernaryFunctorTraits<TernaryOp, Arg1, Arg2, Arg3>::template is_triangular<t>;

    static constexpr bool is_triangular_adapter = false;

    static constexpr bool is_hermitian = Eigen3::TernaryFunctorTraits<TernaryOp, Arg1, Arg2, Arg3>::is_hermitian;
  };

} // namespace OpenKalman::interface

#endif //OPENKALMAN_EIGEN_TRAITS_CWISETERNARYOP_HPP
