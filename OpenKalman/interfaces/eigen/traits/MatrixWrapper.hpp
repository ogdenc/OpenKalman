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
 * \brief Type traits as applied to Eigen::MatrixWrapper.
 */

#ifndef OPENKALMAN_EIGEN_TRAITS_MATRIXWRAPPER_HPP
#define OPENKALMAN_EIGEN_TRAITS_MATRIXWRAPPER_HPP

#include <type_traits>


namespace OpenKalman::interface
{
  template<typename XprType>
  struct indexible_object_traits<Eigen::MatrixWrapper<XprType>>
    : Eigen3::indexible_object_traits_base<Eigen::MatrixWrapper<XprType>>
  {
  private:

    using NestedXpr = typename Eigen::MatrixWrapper<XprType>::NestedExpressionType;
    using Base = Eigen3::indexible_object_traits_base<Eigen::MatrixWrapper<XprType>>;

  public:

    template<typename Arg, typename N>
    static constexpr auto get_vector_space_descriptor(const Arg& arg, N n)
    {
      return OpenKalman::get_vector_space_descriptor(arg.nestedExpression(), n);
    }

    using dependents = std::tuple<NestedXpr>;

    static constexpr bool has_runtime_parameters = false;


    template<typename Arg>
    static NestedXpr nested_object(Arg&& arg)
    {
      if constexpr (std::is_lvalue_reference_v<NestedXpr>)
        return const_cast<NestedXpr>(std::forward<Arg>(arg).nestedExpression());
      else
        return static_cast<NestedXpr>(std::forward<Arg>(arg).nestedExpression());
    }


    template<typename Arg>
    static auto convert_to_self_contained(Arg&& arg)
    {
      using N = Eigen::MatrixWrapper<equivalent_self_contained_t<XprType>>;
      if constexpr (not std::is_lvalue_reference_v<typename N::NestedExpressionType>)
        return make_self_contained(arg.nestedExpression()).matrix();
      else
        return make_dense_object(std::forward<Arg>(arg));
    }

    template<typename Arg>
    static constexpr auto get_constant(const Arg& arg)
    {
      return constant_coefficient{arg.nestedExpression()};
    }

    template<typename Arg>
    static constexpr auto get_constant_diagonal(const Arg& arg)
    {
      return constant_diagonal_coefficient {arg.nestedExpression()};
    }

    template<Qualification b>
    static constexpr bool one_dimensional = OpenKalman::one_dimensional<XprType, b>;

    template<Qualification b>
    static constexpr bool is_square = square_shaped<XprType, b>;

    template<TriangleType t, Qualification b>
    static constexpr bool is_triangular = triangular_matrix<XprType, t, b>;

    static constexpr bool is_triangular_adapter = false;

    static constexpr bool is_hermitian = hermitian_matrix<XprType, Qualification::depends_on_dynamic_shape>;

    // make_hermitian_adapter(Arg&& arg) not defined

    static constexpr Layout layout = layout_of_v<XprType>;
  };


} // namespace OpenKalman::interface

#endif //OPENKALMAN_EIGEN_TRAITS_MATRIXWRAPPER_HPP
