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
 * \brief Type traits as applied to native Eigen3 types.
 */

#ifndef OPENKALMAN_EIGEN_PARTIALREDUXEXPR_HPP
#define OPENKALMAN_EIGEN_PARTIALREDUXEXPR_HPP

#include <type_traits>

namespace OpenKalman::Eigen3::detail
{
#ifdef __cpp_concepts
  template<typename Dim, typename OtherDim>
  concept at_least_square = (std::decay_t<Dim>::value >= std::decay_t<OtherDim>::value);
#else
  template<typename Dim, typename OtherDim, typename = void>
    struct at_least_square_impl : std::false_type {};

    template<typename Dim, typename OtherDim>
    struct at_least_square_impl<Dim, OtherDim, std::enable_if_t<
      value::static_scalar<Dim> and value::static_scalar<OtherDim>>>
      : std::bool_constant<std::decay_t<Dim>::value >= std::decay_t<OtherDim>::value> {};

    template<typename Dim, typename OtherDim>
    constexpr bool at_least_square = at_least_square_impl<Dim, OtherDim>::value;
#endif


  template<typename MemberOp, std::size_t direction, typename XprType, typename Factor, typename DirDim, typename Func>
  constexpr auto get_constant_redux(const XprType& xpr, const Factor& factor, const DirDim& dir_dim, Func&& func)
  {
    auto dim = internal::best_vector_space_descriptor(dir_dim, get_index_dimension_of<direction>(xpr));

    if constexpr (Eigen3::eigen_MatrixWrapper<XprType> or Eigen3::eigen_ArrayWrapper<XprType> or
      internal::fixed_size_adapter<XprType> or Eigen3::eigen_wrapper<XprType>)
    {
      return get_constant_redux<MemberOp, direction>(nested_object(xpr), factor, dim, std::forward<Func>(func));
    }
    else if constexpr (Eigen3::eigen_CwiseUnaryOp<XprType> or Eigen3::eigen_CwiseUnaryView<XprType>)
    {
      auto new_func = Eigen3::functor_composition {std::forward<Func>(func), xpr.functor()};
      return get_constant_redux<MemberOp, direction>(xpr.nestedExpression(), factor, dim, std::move(new_func));
    }
    else if constexpr (Eigen3::eigen_Replicate<XprType>)
    {
      using F = eigen_Replicate_factor<XprType, direction>;
      const auto& n_xpr = xpr.nestedExpression();
      auto n_dim = get_index_dimension_of<direction>(n_xpr);

      auto f = [](const auto& dim, const auto& n_dim) {
        if constexpr (F::value != dynamic_size) return F{};
        else return value::static_scalar_operation {std::divides<std::size_t>{}, dim, n_dim};
      }(dim, n_dim);

      auto new_dim = [](const auto& dim, const auto& n_dim) {
        if constexpr (value::static_scalar<decltype(dim)> and F::value != dynamic_size and
            not value::static_scalar<decltype(n_dim)>)
          return value::static_scalar_operation {std::divides<std::size_t>{}, dim, F{}};
        else
          return n_dim;
      }(dim, n_dim);

      auto new_f = value::static_scalar_operation{std::multiplies<std::size_t>{}, factor, f};
      return get_constant_redux<MemberOp, direction>(n_xpr, new_f, new_dim, std::forward<Func>(func));
    }
    else
    {
      if constexpr (constant_matrix<XprType>)
      {
        auto c = value::static_scalar_operation {std::forward<Func>(func), constant_coefficient {xpr}};
        return Eigen3::ReduxTraits<MemberOp, direction>::get_constant(c, factor, dim);
      }
      else if constexpr (constant_diagonal_matrix<XprType>)
      {
        constexpr bool als = at_least_square<decltype(dim), decltype(get_index_dimension_of<direction == 1 ? 0 : 1>(xpr))>;
        auto c = value::static_scalar_operation {std::forward<Func>(func), constant_diagonal_coefficient {xpr}};
        return Eigen3::ReduxTraits<MemberOp, direction>::template get_constant_diagonal<als>(c, factor, dim);
      }
      else
      {
        return std::monostate {};
      }
    }
  }

} // namespace OpenKalman::Eigen3::detail


namespace OpenKalman::interface
{

  // ------------------------- //
  //  indexible_object_traits  //
  // ------------------------- //

  template<typename MatrixType, typename MemberOp, int Direction>
  struct indexible_object_traits<Eigen::PartialReduxExpr<MatrixType, MemberOp, Direction>>
    : Eigen3::indexible_object_traits_base<Eigen::PartialReduxExpr<MatrixType, MemberOp, Direction>>
  {
  private:

    using Base = Eigen3::indexible_object_traits_base<Eigen::PartialReduxExpr<MatrixType, MemberOp, Direction>>;

#if __cplusplus < 202002L
    struct Op
    {
      template<typename Scalar>
      constexpr Scalar&& operator()(Scalar&& arg) const { return std::forward<Scalar>(arg); }
    };
#endif

  public:

    template<typename Arg>
    static decltype(auto) nested_object(Arg&& arg)
    {
      return std::forward<Arg>(arg).nestedExpression();
    }


    // If a partial redux expression needs to be partially evaluated, it's probably faster to do a full evaluation.
    // Thus, we omit the conversion function.

    template<typename Arg>
    static constexpr auto get_constant(const Arg& arg)
    {
      // colwise (acting on columns) is Eigen::Vertical and rowwise (acting on rows) is Eigen::Horizontal
      constexpr std::size_t direction = Direction == Eigen::Horizontal ? 1 : 0;
      const auto& x = arg.nestedExpression();
      auto dim = get_index_dimension_of<direction>(x);
      std::integral_constant<std::size_t, 1> f;
#if __cplusplus >= 202002L
      return OpenKalman::Eigen3::detail::get_constant_redux<MemberOp, direction>(x, f, dim, std::identity{});
#else
      return OpenKalman::Eigen3::detail::get_constant_redux<MemberOp, direction>(x, f, dim, Op{});
#endif
    }

  };

} // namespace OpenKalman::interface

#endif //OPENKALMAN_EIGEN_PARTIALREDUXEXPR_HPP
