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
 * \brief Traits for Eigen::CwiseBinaryOp.
 */

#ifndef OPENKALMAN_EIGEN3_TRAITS_CWISEBINARYOP_HPP
#define OPENKALMAN_EIGEN3_TRAITS_CWISEBINARYOP_HPP

#include <type_traits>

namespace OpenKalman::interface
{
  namespace EGI = Eigen::internal;

  template<typename BinaryOp, typename LhsType, typename RhsType>
  struct Dependencies<Eigen::CwiseBinaryOp<BinaryOp, LhsType, RhsType>>
  {
  private:

    using T = Eigen::CwiseBinaryOp<BinaryOp, LhsType, RhsType>;

  public:

    static constexpr bool has_runtime_parameters = false;
    using type = std::tuple<typename T::LhsNested, typename T::RhsNested>;

    template<std::size_t i, typename Arg>
    static decltype(auto) get_nested_matrix(Arg&& arg)
    {
      static_assert(i <= 2);
      if constexpr (i == 0)
        return std::forward<Arg>(arg).lhs();
      else if constexpr (i == 2)
        return std::forward<Arg>(arg).rhs();
      else
        return std::forward<Arg>(arg).functor();
    }

    template<typename Arg>
    static auto convert_to_self_contained(Arg&& arg)
    {
      using N = Eigen::CwiseBinaryOp<BinaryOp, equivalent_self_contained_t<LhsType>,
        equivalent_self_contained_t<RhsType>>;
      // Do a partial evaluation as long as at least one argument is already self-contained.
      if constexpr ((self_contained<LhsType> or self_contained<RhsType>) and
        not std::is_lvalue_reference_v<typename N::LhsNested> and
        not std::is_lvalue_reference_v<typename N::RhsNested>)
      {
        return N {make_self_contained(arg.lhs()), make_self_contained(arg.rhs()), arg.functor()};
      }
      else
      {
        return make_dense_writable_matrix_from(std::forward<Arg>(arg));
      }
    }
  };


  template<typename BinaryOp, typename LhsType, typename RhsType>
  struct SingleConstant<Eigen::CwiseBinaryOp<BinaryOp, LhsType, RhsType>>
  {
    const Eigen::CwiseBinaryOp<BinaryOp, LhsType, RhsType>& xpr;

    constexpr auto get_constant()
    {
      return Eigen3::FunctorTraits<BinaryOp, LhsType, RhsType>::template get_constant<constant_coefficient>(xpr);
    }

    constexpr auto get_constant_diagonal()
    {
      return Eigen3::FunctorTraits<BinaryOp, LhsType, RhsType>::template get_constant<constant_diagonal_coefficient>(xpr);
    }
  };


  template<typename BinaryOp, typename LhsType, typename RhsType>
  struct DiagonalTraits<Eigen::CwiseBinaryOp<BinaryOp, LhsType, RhsType>>
  {
    static constexpr bool is_diagonal = Eigen3::FunctorTraits<BinaryOp, LhsType, RhsType>::is_diagonal;
  };


  template<typename BinaryOp, typename LhsType, typename RhsType>
  struct TriangularTraits<Eigen::CwiseBinaryOp<BinaryOp, LhsType, RhsType>>
  {
    static constexpr TriangleType triangle_type = Eigen3::FunctorTraits<BinaryOp, LhsType, RhsType>::triangle_type;

    static constexpr bool is_triangular_adapter = false;
  };


  template<typename BinaryOp, typename LhsType, typename RhsType>
  struct HermitianTraits<Eigen::CwiseBinaryOp<BinaryOp, LhsType, RhsType>>
  {
    static constexpr bool is_hermitian = Eigen3::FunctorTraits<BinaryOp, LhsType, RhsType>::is_hermitian;

    static constexpr TriangleType adapter_type = TriangleType::none;
  };


} // namespace OpenKalman::interface

#endif //OPENKALMAN_EIGEN3_TRAITS_CWISEBINARYOP_HPP
