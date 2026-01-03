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
  struct object_traits<Eigen::Reshaped<XprType, Rows, Cols, Order>>
    : Eigen3::object_traits_base<Eigen::Reshaped<XprType, Rows, Cols, Order>>
  {
  private:

    using Base = Eigen3::object_traits_base<Eigen::Reshaped<XprType, Rows, Cols, Order>>;

    static constexpr std::size_t nested_components = has_dynamic_dimensions<XprType> ? stdex::dynamic_extent :
      index_dimension_of_v<XprType, 0> * index_dimension_of_v<XprType, 1>;

    static constexpr std::size_t xprtypemax = dynamic_index_count_v<XprType> < 2 ? std::max(
      dynamic_dimension<XprType, 0> ? 0 : index_dimension_of_v<XprType, 0>,
      dynamic_dimension<XprType, 1> ? 0 : index_dimension_of_v<XprType, 1>) : stdex::dynamic_extent;

    static constexpr bool HasDirectAccess = Eigen::internal::traits<Eigen::Reshaped<XprType, Rows, Cols, Order>>::HasDirectAccess;

    using Nested_t = typename detail::ReshapedNested<XprType, Rows, Cols, Order, HasDirectAccess>::type;

  public:

    template<typename Arg, typename N>
    static constexpr auto get_pattern_collection(const Arg& arg, N n)
    {
      if constexpr (values::fixed<N>)
      {
        constexpr auto dim = n == 0_uz ? Rows : Cols;
        constexpr auto other_dim = n == 0_uz ? Cols : Rows;
        constexpr std::size_t dimension =
          dim != Eigen::Dynamic ? dim :
          other_dim == Eigen::Dynamic or other_dim == 0 ? stdex::dynamic_extent :
          other_dim == index_dimension_of_v<XprType, 0> ? index_dimension_of_v<XprType, 1> :
          other_dim == index_dimension_of_v<XprType, 1> ? index_dimension_of_v<XprType, 0> :
            nested_components != stdex::dynamic_extent and nested_components % other_dim == 0 ? nested_components / other_dim :
          stdex::dynamic_extent;

        if constexpr (dimension == stdex::dynamic_extent)
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


    template<typename Arg>
    static decltype(auto) nested_object(Arg&& arg)
    {
      return std::forward<Arg>(arg).nestedExpression();
    }


    template<typename Arg>
    static constexpr auto get_constant(const Arg& arg)
    {
      return constant_value {arg.nestedExpression()};
    }


    template<typename Arg>
    static constexpr auto get_constant_diagonal(const Arg& arg)
    {
      if constexpr ((Rows != Eigen::Dynamic and Rows == XprType::RowsAtCompileTime) or (Cols != Eigen::Dynamic and Cols == XprType::ColsAtCompileTime))
        return constant_diagonal_value {arg.nestedExpression()};
      else
        return std::monostate{};
    }


    template<applicability b>
    static constexpr bool one_dimensional =
      (Rows == 1 and Cols == 1 and OpenKalman::one_dimensional<XprType, values::unbounded_size, applicability::permitted>) or
      ((Rows == 1 or Rows == Eigen::Dynamic) and (Cols == 1 or Cols == Eigen::Dynamic) and OpenKalman::one_dimensional<XprType, b>);


    template<applicability b>
    static constexpr bool is_square =
      (b != applicability::guaranteed or (Rows != Eigen::Dynamic and Cols != Eigen::Dynamic) or
        ((Rows != Eigen::Dynamic or Cols != Eigen::Dynamic) and not has_dynamic_dimensions<XprType>)) and
      (Rows == Eigen::Dynamic or Cols == Eigen::Dynamic or Rows == Cols) and
      (nested_components == stdex::dynamic_extent or (
        values::internal::near(nested_components, values::sqrt(nested_components) * values::sqrt(nested_components)) and
        (Rows == Eigen::Dynamic or Rows * Rows == nested_components))) and
      (Rows == Eigen::Dynamic or xprtypemax == stdex::dynamic_extent or (Rows * Rows) % xprtypemax == 0) and
      (Cols == Eigen::Dynamic or xprtypemax == stdex::dynamic_extent or (Cols * Cols) % xprtypemax == 0);


    template<triangle_type t>
    static constexpr bool triangle_type_value = triangular_matrix<XprType, t> and
      ((Rows != Eigen::Dynamic and Rows == XprType::RowsAtCompileTime) or (Cols != Eigen::Dynamic and Cols == XprType::ColsAtCompileTime));


    static constexpr bool is_triangular_adapter = false;


    static constexpr bool is_hermitian = hermitian_matrix<XprType, applicability::permitted> and
      ((XprType::RowsAtCompileTime == Eigen::Dynamic and XprType::ColsAtCompileTime == Eigen::Dynamic) or
        (Rows == Eigen::Dynamic or Rows == XprType::RowsAtCompileTime or Rows == XprType::ColsAtCompileTime) and
        (Cols == Eigen::Dynamic or Cols == XprType::ColsAtCompileTime or Cols == XprType::RowsAtCompileTime));
  };

}

#endif
