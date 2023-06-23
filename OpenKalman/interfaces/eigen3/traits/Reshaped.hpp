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

#ifndef OPENKALMAN_EIGEN3_TRAITS_RESHAPED_HPP
#define OPENKALMAN_EIGEN3_TRAITS_RESHAPED_HPP

#include <type_traits>


namespace OpenKalman::interface
{
  template<typename XprType, int Rows, int Cols, int Order>
  struct IndexTraits<Eigen::Reshaped<XprType, Rows, Cols, Order>>
  {
  private:

    static constexpr std::size_t xprtypeprod = has_dynamic_dimensions<XprType> ? dynamic_size :
      index_dimension_of_v<XprType, 0> * index_dimension_of_v<XprType, 1>;

    static constexpr std::size_t xprtypemax = std::max(
      dynamic_dimension<XprType, 0> ? 0 : index_dimension_of_v<XprType, 0>,
      dynamic_dimension<XprType, 1> ? 0 : index_dimension_of_v<XprType, 1>);

  public:

    static constexpr std::size_t max_indices = max_indices_of_v<XprType>;

    template<std::size_t N, typename Arg>
    static constexpr auto get_index_descriptor(const Arg& arg)
    {
      constexpr auto dim = N == 0 ? Rows : Cols;
      constexpr auto other_dim = N == 0 ? Cols : Rows;
      constexpr std::size_t dimension =
        dim != Eigen::Dynamic ? dim :
        other_dim == Eigen::Dynamic or other_dim == 0 ? dynamic_size :
        other_dim == index_dimension_of_v<XprType, 0> ? index_dimension_of_v<XprType, 1> :
        other_dim == index_dimension_of_v<XprType, 1> ? index_dimension_of_v<XprType, 0> :
        xprtypeprod != dynamic_size and xprtypeprod % other_dim == 0 ? xprtypeprod / other_dim :
        dynamic_size;

      if constexpr (dimension == dynamic_size)
      {
        if constexpr (N == 0) return static_cast<std::size_t>(arg.rows());
        else return static_cast<std::size_t>(arg.cols());
      }
      else return Dimensions<dimension>{};
    }

    template<Likelihood b>
    static constexpr bool is_one_by_one =
      (Rows == 1 and Cols == 1 and one_by_one_matrix<XprType, Likelihood::maybe>) or
      ((Rows == 1 or Rows == Eigen::Dynamic) and (Cols == 1 or Cols == Eigen::Dynamic) and one_by_one_matrix<XprType, b>);

    template<Likelihood b>
    static constexpr bool is_square =
      (b != Likelihood::definitely or (Rows != Eigen::Dynamic and Cols != Eigen::Dynamic) or
        ((Rows != Eigen::Dynamic or Cols != Eigen::Dynamic) and number_of_dynamic_indices_v<XprType> <= 1)) and
      (Rows == Eigen::Dynamic or Cols == Eigen::Dynamic or Rows == Cols) and
      (xprtypeprod == dynamic_size or (
        are_within_tolerance(xprtypeprod, internal::constexpr_sqrt(xprtypeprod) * internal::constexpr_sqrt(xprtypeprod)) and
        (Rows == Eigen::Dynamic or Rows * Rows == xprtypeprod) and
        (Cols == Eigen::Dynamic or Cols * Cols == xprtypeprod))) and
      (Rows == Eigen::Dynamic or xprtypemax == 0 or (Rows * Rows) % xprtypemax == 0) and
      (Cols == Eigen::Dynamic or xprtypemax == 0 or (Cols * Cols) % xprtypemax == 0);
  };


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
  struct Dependencies<Eigen::Reshaped<XprType, Rows, Cols, Order>>
  {
  private:
    using R = Eigen::Reshaped<XprType, Rows, Cols, Order>;

    static constexpr bool HasDirectAccess = Eigen::internal::traits<R>::HasDirectAccess;

    using Nested_t = typename detail::ReshapedNested<XprType, Rows, Cols, Order, HasDirectAccess>::type;

  public:

    static constexpr bool has_runtime_parameters = HasDirectAccess ? Rows == Eigen::Dynamic or Cols == Eigen::Dynamic : false;

    using type = std::tuple<Nested_t>;

    template<std::size_t i, typename Arg>
    static decltype(auto) get_nested_matrix(Arg&& arg)
    {
      static_assert(i == 0);
      return std::forward<Arg>(arg).nestedExpression();
    }

    template<typename Arg>
    static auto convert_to_self_contained(Arg&& arg)
    {
      return make_dense_writable_matrix_from(std::forward<Arg>(arg));
    }
  };


  template<typename XprType, int Rows, int Cols, int Order>
  struct SingleConstant<Eigen::Reshaped<XprType, Rows, Cols, Order>>
  {
    const Eigen::Reshaped<XprType, Rows, Cols, Order>& xpr;

    constexpr auto get_constant()
    {
      return constant_coefficient {xpr.nestedExpression()};
    }

    constexpr auto get_constant_diagonal()
    {
      if constexpr (
        (Rows != Eigen::Dynamic and (Rows == XprType::RowsAtCompileTime or Rows == XprType::ColsAtCompileTime or Rows == Cols)) or
        (Cols != Eigen::Dynamic and (Cols == XprType::RowsAtCompileTime or Cols == XprType::ColsAtCompileTime)))
      {
        return constant_diagonal_coefficient {xpr.nestedExpression()};
      }
      else if constexpr (((Rows == Eigen::Dynamic and Cols == Eigen::Dynamic) or
        (XprType::RowsAtCompileTime == Eigen::Dynamic and XprType::ColsAtCompileTime == Eigen::Dynamic)) and
        constant_diagonal_matrix<XprType, CompileTimeStatus::any, Likelihood::maybe>)
      {
        constant_diagonal_coefficient cd {xpr.nestedExpression()};
        return internal::ScalarConstant<Likelihood::maybe, std::decay_t<decltype(cd)>> {cd};
      }
      else return std::monostate{};
    }
  };


  template<typename XprType, int Rows, int Cols, int Order>
  struct TriangularTraits<Eigen::Reshaped<XprType, Rows, Cols, Order>>
  {
    template<TriangleType t, Likelihood b>
    static constexpr bool is_triangular = triangular_matrix<XprType, t, b> and
      (Rows == index_dimension_of_v<XprType, 0> or Rows == index_dimension_of_v<XprType, 1> or
        Cols == index_dimension_of_v<XprType, 1> or Cols == index_dimension_of_v<XprType, 0> or
        (Rows != Eigen::Dynamic and Rows == Cols));

    static constexpr bool is_triangular_adapter = false;
  };


#ifdef __cpp_concepts
  template<hermitian_matrix<Likelihood::maybe> XprType, int Rows, int Cols, int Order>
  struct HermitianTraits<Eigen::Reshaped<XprType, Rows, Cols, Order>>
#else
  template<typename XprType, int Rows, int Cols, int Order>
  struct HermitianTraits<Eigen::Reshaped<XprType, Rows, Cols, Order>, std::enable_if_t<
    hermitian_matrix<XprType, Likelihood::maybe>>>
#endif
  {
    static constexpr bool is_hermitian =
      Rows == index_dimension_of_v<XprType, 0> or Rows == index_dimension_of_v<XprType, 1> or
        Cols == index_dimension_of_v<XprType, 1> or Cols == index_dimension_of_v<XprType, 0> or
        (Rows != Eigen::Dynamic and Rows == Cols);
  };

} // namespace OpenKalman::interface

#endif //OPENKALMAN_EIGEN3_TRAITS_RESHAPED_HPP
