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
#ifdef __cpp_concepts
  struct IndexTraits<Eigen::CwiseBinaryOp<BinaryOp, LhsType, RhsType>>
#else
  struct IndexTraits<Eigen::CwiseBinaryOp<BinaryOp, LhsType, RhsType>, std::enable_if_t<native_eigen_general<Eigen::CwiseBinaryOp<BinaryOp, LhsType, RhsType>>>>
#endif
  {
    template<std::size_t N>
    static constexpr std::size_t dimension =
      not dynamic_dimension<LhsType, N> ? index_dimension_of_v<LhsType, N> :
      not dynamic_dimension<RhsType, N> ? index_dimension_of_v<RhsType, N> :
      dynamic_size;

    template<std::size_t N, typename Arg>
    static constexpr std::size_t dimension_at_runtime(const Arg& arg)
    {
      if constexpr (dimension<N> != dynamic_size) return dimension<N>;
      else return get_index_dimension_of<N>(arg.lhs());
    }

    template<Likelihood b>
    static constexpr bool is_one_by_one =
      one_by_one_matrix<LhsType, Likelihood::maybe> and one_by_one_matrix<RhsType, Likelihood::maybe> and
      (b != Likelihood::definitely or not has_dynamic_dimensions<Eigen::CwiseBinaryOp<BinaryOp, LhsType, RhsType>> or
        (square_matrix<LhsType, b> and (index_dimension_of_v<RhsType, 0> == 1 or index_dimension_of_v<RhsType, 1> == 1)) or
        ((index_dimension_of_v<LhsType, 0> == 1 or index_dimension_of_v<LhsType, 1> == 1) and square_matrix<RhsType, b>));

    template<Likelihood b>
    static constexpr bool is_square =
      square_matrix<LhsType, Likelihood::maybe> and square_matrix<RhsType, Likelihood::maybe> and
      (b != Likelihood::definitely or not has_dynamic_dimensions<Eigen::CwiseBinaryOp<BinaryOp, LhsType, RhsType>> or
        square_matrix<LhsType, b> or square_matrix<RhsType, b>);
  };


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
      return Eigen3::FunctorTraits<BinaryOp, LhsType, RhsType>::template get_constant<false>(xpr);
    }

    constexpr auto get_constant_diagonal()
    {
      return Eigen3::FunctorTraits<BinaryOp, LhsType, RhsType>::template get_constant<true>(xpr);
    }
  };


  template<typename BinaryOp, typename LhsType, typename RhsType>
  struct TriangularTraits<Eigen::CwiseBinaryOp<BinaryOp, LhsType, RhsType>>
  {
    template<TriangleType t, Likelihood b>
    static constexpr bool is_triangular = Eigen3::FunctorTraits<BinaryOp, LhsType, RhsType>::template is_triangular<t, b>;

    static constexpr bool is_triangular_adapter = false;
  };


  template<typename BinaryOp, typename LhsType, typename RhsType>
  struct HermitianTraits<Eigen::CwiseBinaryOp<BinaryOp, LhsType, RhsType>>
  {
    static constexpr bool is_hermitian = Eigen3::FunctorTraits<BinaryOp, LhsType, RhsType>::is_hermitian;
  };


} // namespace OpenKalman::interface

#endif //OPENKALMAN_EIGEN3_TRAITS_CWISEBINARYOP_HPP
