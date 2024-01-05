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
 * \brief Type traits as applied to Eigen::Reshaped (Eigen version 3.4).
 */

#ifndef OPENKALMAN_EIGEN_TRAITS_RESHAPED_HPP
#define OPENKALMAN_EIGEN_TRAITS_RESHAPED_HPP

#include <type_traits>


namespace OpenKalman::interface
{
  namespace detail
  {
    template<typename XprType, int Rows, int Cols, int Order, bool HasDirectAccess>
    struct ReshapedNested { using type = typename Eigen::Reshaped<XprType, Rows, Cols, Order>::MatrixTypeNested; };

    template<typename XprType, int Rows, int Cols, int Order>
    struct ReshapedNested<XprType, Rows, Cols, Order, true>
    {
      using type = typename Eigen::internal::ref_selector<XprType>::non_const_type;
    };
  }


  template<typename XprType, int Rows, int Cols, int Order>
  struct indexible_object_traits<Eigen::Reshaped<XprType, Rows, Cols, Order>>
    : Eigen3::indexible_object_traits_base<Eigen::Reshaped<XprType, Rows, Cols, Order>>
  {
  private:

    using Base = Eigen3::indexible_object_traits_base<Eigen::Reshaped<XprType, Rows, Cols, Order>>;

    static constexpr std::size_t xprtypeprod = has_dynamic_dimensions<XprType> ? dynamic_size :
      index_dimension_of_v<XprType, 0> * index_dimension_of_v<XprType, 1>;

    static constexpr std::size_t xprtypemax = std::max(
      dynamic_dimension<XprType, 0> ? 0 : index_dimension_of_v<XprType, 0>,
      dynamic_dimension<XprType, 1> ? 0 : index_dimension_of_v<XprType, 1>);

    static constexpr bool HasDirectAccess = Eigen::internal::traits<Eigen::Reshaped<XprType, Rows, Cols, Order>>::HasDirectAccess;

    using Nested_t = typename detail::ReshapedNested<XprType, Rows, Cols, Order, HasDirectAccess>::type;

  public:

    template<typename Arg, typename N>
    static constexpr auto get_vector_space_descriptor(const Arg& arg, N n)
    {
      if constexpr (static_index_value<N>)
      {
        constexpr auto dim = n == 0_uz ? Rows : Cols;
        constexpr auto other_dim = n == 0_uz ? Cols : Rows;
        constexpr std::size_t dimension =
          dim != Eigen::Dynamic ? dim :
          other_dim == Eigen::Dynamic or other_dim == 0 ? dynamic_size :
          other_dim == index_dimension_of_v<XprType, 0> ? index_dimension_of_v<XprType, 1> :
          other_dim == index_dimension_of_v<XprType, 1> ? index_dimension_of_v<XprType, 0> :
          xprtypeprod != dynamic_size and xprtypeprod % other_dim == 0 ? xprtypeprod / other_dim :
          dynamic_size;

        if constexpr (dimension == dynamic_size)
        {
          if constexpr (n == 0_uz) return static_cast<std::size_t>(arg.rows());
          else return static_cast<std::size_t>(arg.cols());
        }
        else return Dimensions<dimension>{};
      }
      else
      {
        if (n == 0_uz) return static_cast<std::size_t>(arg.rows());
        else return static_cast<std::size_t>(arg.cols());
      }
    }

    using dependents = std::tuple<Nested_t>;

    static constexpr bool has_runtime_parameters = HasDirectAccess ? Rows == Eigen::Dynamic or Cols == Eigen::Dynamic : false;


    template<typename Arg>
    static decltype(auto) nested_object(Arg&& arg)
    {
      return std::forward<Arg>(arg).nestedExpression();
    }


    template<typename Arg>
    static auto convert_to_self_contained(Arg&& arg)
    {
      return make_dense_object(std::forward<Arg>(arg));
    }

    template<typename Arg>
    static constexpr auto get_constant(const Arg& arg)
    {
      return constant_coefficient {arg.nestedExpression()};
    }

    template<typename Arg>
    static constexpr auto get_constant_diagonal(const Arg& arg)
    {
      if constexpr (
        (Rows != Eigen::Dynamic and (Rows == XprType::RowsAtCompileTime or Rows == XprType::ColsAtCompileTime or Rows == Cols)) or
        (Cols != Eigen::Dynamic and (Cols == XprType::RowsAtCompileTime or Cols == XprType::ColsAtCompileTime)))
      {
        return constant_diagonal_coefficient {arg.nestedExpression()};
      }
      else if constexpr (((Rows == Eigen::Dynamic and Cols == Eigen::Dynamic) or
        (XprType::RowsAtCompileTime == Eigen::Dynamic and XprType::ColsAtCompileTime == Eigen::Dynamic)) and
        constant_diagonal_matrix<XprType>)
      {
        constant_diagonal_coefficient cd {arg.nestedExpression()};
        return internal::ScalarConstant<Qualification::depends_on_dynamic_shape, std::decay_t<decltype(cd)>> {cd};
      }
      else return std::monostate{};
    }

    template<Qualification b>
    static constexpr bool one_dimensional =
      (Rows == 1 and Cols == 1 and OpenKalman::one_dimensional<XprType, Qualification::depends_on_dynamic_shape>) or
      ((Rows == 1 or Rows == Eigen::Dynamic) and (Cols == 1 or Cols == Eigen::Dynamic) and OpenKalman::one_dimensional<XprType, b>);

    template<Qualification b>
    static constexpr bool is_square =
      (b != Qualification::unqualified or (Rows != Eigen::Dynamic and Cols != Eigen::Dynamic) or
        ((Rows != Eigen::Dynamic or Cols != Eigen::Dynamic) and dynamic_index_count_v<XprType> <= 1)) and
      (Rows == Eigen::Dynamic or Cols == Eigen::Dynamic or Rows == Cols) and
      (xprtypeprod == dynamic_size or (
        internal::are_within_tolerance(xprtypeprod, internal::constexpr_sqrt(xprtypeprod) * internal::constexpr_sqrt(xprtypeprod)) and
        (Rows == Eigen::Dynamic or Rows * Rows == xprtypeprod) and
        (Cols == Eigen::Dynamic or Cols * Cols == xprtypeprod))) and
      (Rows == Eigen::Dynamic or xprtypemax == 0 or (Rows * Rows) % xprtypemax == 0) and
      (Cols == Eigen::Dynamic or xprtypemax == 0 or (Cols * Cols) % xprtypemax == 0);

    template<TriangleType t>
    static constexpr bool is_triangular = triangular_matrix<XprType, t> and
      (Rows == index_dimension_of_v<XprType, 0> or Rows == index_dimension_of_v<XprType, 1> or
        Cols == index_dimension_of_v<XprType, 1> or Cols == index_dimension_of_v<XprType, 0> or
        (Rows != Eigen::Dynamic and Rows == Cols));

    static constexpr bool is_triangular_adapter = false;

    static constexpr bool is_hermitian = hermitian_matrix<XprType, Qualification::depends_on_dynamic_shape> and
      (Rows == index_dimension_of_v<XprType, 0> or Rows == index_dimension_of_v<XprType, 1> or
        Cols == index_dimension_of_v<XprType, 1> or Cols == index_dimension_of_v<XprType, 0> or
        (Rows != Eigen::Dynamic and Rows == Cols));
  };

} // namespace OpenKalman::interface

#endif //OPENKALMAN_EIGEN_TRAITS_RESHAPED_HPP
