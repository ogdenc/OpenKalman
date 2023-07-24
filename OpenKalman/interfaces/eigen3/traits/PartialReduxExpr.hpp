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

#ifndef OPENKALMAN_EIGEN3_PARTIALREDUXEXPR_HPP
#define OPENKALMAN_EIGEN3_PARTIALREDUXEXPR_HPP

#include <type_traits>


namespace OpenKalman::interface
{

  namespace detail
  {
    template<typename XprType>
    struct is_diag : std::bool_constant<
      zero_matrix<XprType> or one_by_one_matrix<XprType> ? true :
      constant_matrix<XprType> ? false :
      constant_diagonal_matrix<XprType, CompileTimeStatus::any, Likelihood::maybe> ? true :
      constant_matrix<XprType, CompileTimeStatus::any, Likelihood::maybe> ? std::false_type{} : false> {};

    template<typename XprType>
    constexpr bool is_diag_v = is_diag<XprType>::value;


    template<typename XprType, int Direction>
    struct is_EigenReplicate : std::false_type {};

    template<typename MatrixType, int RowFactor, int ColFactor, int Direction>
    struct is_EigenReplicate<Eigen::Replicate<MatrixType, RowFactor, ColFactor>, Direction> : std::true_type
    {
    private:
      static constexpr int Efactor = Direction == Eigen::Horizontal ? ColFactor : RowFactor;
    public:
      static constexpr std::size_t direction = Direction == Eigen::Horizontal ? 1 : 0;
      static constexpr std::size_t factor = Efactor == Eigen::Dynamic ? dynamic_size : Efactor;
      static constexpr auto get_nested(const Eigen::Replicate<MatrixType, RowFactor, ColFactor>& xpr) { return xpr.nestedExpression(); }
    };

    template<typename XprType, int Direction>
    struct is_EigenReplicate<const XprType, Direction> : is_EigenReplicate<XprType, Direction> {};


    template<typename MemberOp, std::size_t direction, std::size_t factor,
      typename XprType, typename C, typename Dim>
    constexpr auto get_PartialReduxExpr_replicate(const XprType& xpr, const C& c, const Dim& dim)
    {
      if constexpr (factor == dynamic_size)
      {
        auto d = internal::index_dimension_value_of<direction>(xpr);
        auto f = get_scalar_constant_value(dim) / d;
        return Eigen3::SingleConstantPartialRedux<XprType, MemberOp>::get_constant(c, d, f);
      }
      else
      {
        std::integral_constant<std::size_t, factor> f;
        internal::scalar_constant_operation d {std::divides<>{}, dim, f};
        return Eigen3::SingleConstantPartialRedux<XprType, MemberOp>::get_constant(c, d, f);
      }
    }


#ifdef __cpp_concepts
    template<typename MemberOp, int Direction, typename XprType, typename Dim>
#else
    template<typename MemberOp, int Direction, typename XprType, typename Dim, std::enable_if_t<
      not is_EigenReplicate<XprType, Direction>::value and
      not Eigen3::eigen_MatrixWrapper<XprType> and not Eigen3::eigen_ArrayWrapper<XprType> and not Eigen3::eigen_wrapper<XprType>, int> = 0>
#endif
    constexpr auto get_PartialReduxExpr_constant(const XprType& xpr, const Dim& dim)
    {
      std::conditional_t<is_diag_v<XprType>, constant_diagonal_coefficient<XprType>, constant_coefficient<XprType>> c {xpr};
      std::integral_constant<std::size_t, 1> f;
      return Eigen3::SingleConstantPartialRedux<XprType, MemberOp>::get_constant(c, dim, f);
    }


#ifdef __cpp_concepts
    template<typename MemberOp, int Direction, typename XprType, typename Dim>
      requires Eigen3::eigen_MatrixWrapper<XprType> or Eigen3::eigen_ArrayWrapper<XprType> or Eigen3::eigen_wrapper<XprType>
#else
    template<typename MemberOp, int Direction, typename XprType, typename Dim, std::enable_if_t<
      not is_EigenReplicate<XprType, Direction>::value and
      (Eigen3::eigen_MatrixWrapper<XprType> or Eigen3::eigen_ArrayWrapper<XprType> or Eigen3::eigen_wrapper<XprType>), int> = 0>
#endif
    constexpr auto get_PartialReduxExpr_constant(const XprType& xpr, const Dim& dim)
    {
      return get_PartialReduxExpr_constant<MemberOp, Direction>(nested_matrix(xpr), dim);
    }


#ifdef __cpp_concepts
    template<typename MemberOp, int Direction, typename XprType, typename Dim>
      requires is_EigenReplicate<XprType, Direction>::value
#else
    template<typename MemberOp, int Direction, typename XprType, typename Dim, std::enable_if_t<
      is_EigenReplicate<XprType, Direction>::value, int> = 0>
#endif
    constexpr auto get_PartialReduxExpr_constant(const XprType& xpr, const Dim& dim)
    {
      constexpr std::size_t direction = is_EigenReplicate<XprType, Direction>::direction;
      constexpr std::size_t factor = is_EigenReplicate<XprType, Direction>::factor;
      decltype(auto) n_xpr = is_EigenReplicate<XprType, Direction>::get_nested(xpr);
      using NXprType = std::decay_t<decltype(n_xpr)>;
      std::conditional_t<is_diag_v<NXprType>, constant_diagonal_coefficient<NXprType>, constant_coefficient<NXprType>> c {n_xpr};
      return get_PartialReduxExpr_replicate<MemberOp, direction, factor>(n_xpr, c, dim);
    }


#ifdef __cpp_concepts
    template<typename MemberOp, int Direction, typename UnaryOp, typename XprType, typename Dim>
      requires is_EigenReplicate<XprType, Direction>::value
#else
    template<typename MemberOp, int Direction, typename UnaryOp, typename XprType, typename Dim, std::enable_if_t<
      is_EigenReplicate<XprType, Direction>::value, int> = 0>
#endif
    constexpr auto get_PartialReduxExpr_constant(const Eigen::CwiseUnaryOp<UnaryOp, XprType>& xpr, const Dim& dim)
    {
      constexpr std::size_t direction = is_EigenReplicate<XprType, Direction>::direction;
      constexpr std::size_t factor = is_EigenReplicate<XprType, Direction>::factor;
      decltype(auto) n_xpr = is_EigenReplicate<XprType, Direction>::get_nested(xpr.nestedExpression());
      using NXprType = std::decay_t<decltype(n_xpr)>;
      auto uop = Eigen::CwiseUnaryOp<UnaryOp, NXprType> {n_xpr};
      auto c = Eigen3::FunctorTraits<UnaryOp, NXprType>::template get_constant<is_diag_v<XprType>>(uop);
      return get_PartialReduxExpr_replicate<MemberOp, direction, factor>(n_xpr, c, dim);
    }


#ifdef __cpp_concepts
    template<typename MemberOp, int Direction, typename ViewOp, typename XprType, typename Dim>
      requires is_EigenReplicate<XprType, Direction>::value
#else
    template<typename MemberOp, int Direction, typename ViewOp, typename XprType, typename Dim, std::enable_if_t<
      is_EigenReplicate<XprType, Direction>::value, int> = 0>
#endif
    constexpr auto get_PartialReduxExpr_constant(const Eigen::CwiseUnaryView<ViewOp, XprType>& xpr, const Dim& dim)
    {
      constexpr std::size_t direction = is_EigenReplicate<XprType, Direction>::direction;
      constexpr std::size_t factor = is_EigenReplicate<XprType, Direction>::factor;
      decltype(auto) n_xpr = is_EigenReplicate<XprType, Direction>::get_nested(xpr.nestedExpression());
      using NXprType = std::decay_t<decltype(n_xpr)>;
      auto uop = Eigen::CwiseUnaryOp<ViewOp, const NXprType> {n_xpr};
      auto c = Eigen3::FunctorTraits<ViewOp, NXprType>::template get_constant<is_diag_v<XprType>>(uop);
      return get_PartialReduxExpr_replicate<MemberOp, direction, factor>(n_xpr, c, dim);
    }

  } // namespace detail


  template<typename MatrixType, typename MemberOp, int Direction>
  struct IndexibleObjectTraits<Eigen::PartialReduxExpr<MatrixType, MemberOp, Direction>>
    : Eigen3::IndexibleObjectTraitsBase<Eigen::PartialReduxExpr<MatrixType, MemberOp, Direction>>
  {
    static constexpr std::size_t max_indices = 2;

    template<std::size_t N, typename Arg>
    static constexpr auto get_index_descriptor(const Arg& arg)
    {
      using Xpr = Eigen::PartialReduxExpr<MatrixType, MemberOp, Direction>;
      constexpr Eigen::Index dim = N == 0 ? Xpr::RowsAtCompileTime : Xpr::ColsAtCompileTime;

      if constexpr (dim == Eigen::Dynamic)
      {
        if constexpr (N == 0) return static_cast<std::size_t>(arg.rows());
        else return static_cast<std::size_t>(arg.cols());
      }
      else return Dimensions<dim>{};
    }

    static constexpr bool has_runtime_parameters = false;

    using type = std::tuple<typename MatrixType::Nested, const MemberOp>;

    template<std::size_t i, typename Arg>
    static decltype(auto) get_nested_matrix(Arg&& arg)
    {
      if constexpr (i == 0)
        return std::forward<Arg>(arg).nestedExpression();
      else
        return std::forward<Arg>(arg).functor();
      static_assert(i <= 1);
    }

    // If a partial redux expression needs to be partially evaluated, it's probably faster to do a full evaluation.
    // Thus, we omit the conversion function.

    template<typename Arg>
    static constexpr auto get_constant(const Arg& arg)
    {
      // colwise (acting on columns) is Eigen::Vertical and rowwise (acting on rows) is Eigen::Horizontal
      constexpr std::size_t N = Direction == Eigen::Horizontal ? 1 : 0;
      const auto& x {arg.nestedExpression()};
      auto dim = internal::index_dimension_value_of<N>(x);

      return detail::get_PartialReduxExpr_constant<MemberOp, Direction>(x, dim);
    }
  };

} // namespace OpenKalman::interface

#endif //OPENKALMAN_EIGEN3_PARTIALREDUXEXPR_HPP
