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
  template<typename BinaryOp, typename LhsType, typename RhsType>
  struct IndexibleObjectTraits<Eigen::CwiseBinaryOp<BinaryOp, LhsType, RhsType>>
    : Eigen3::IndexibleObjectTraitsBase<Eigen::CwiseBinaryOp<BinaryOp, LhsType, RhsType>>
  {
  private:

    using T = Eigen::CwiseBinaryOp<BinaryOp, LhsType, RhsType>;

  public:

    static constexpr std::size_t max_indices = std::max({max_indices_of_v<LhsType>, max_indices_of_v<RhsType>});

    template<std::size_t N, typename Arg>
    static constexpr auto get_index_descriptor(const Arg& arg)
    {
      if constexpr (not dynamic_dimension<LhsType, N>) return OpenKalman::get_index_descriptor<N>(arg.lhs());
      else return OpenKalman::get_index_descriptor<N>(arg.rhs());
    }

    template<Likelihood b>
    static constexpr bool is_one_by_one =
      one_by_one_matrix<LhsType, Likelihood::maybe> and one_by_one_matrix<RhsType, Likelihood::maybe> and
      (b != Likelihood::definitely or not has_dynamic_dimensions<Eigen::CwiseBinaryOp<BinaryOp, LhsType, RhsType>> or
        (square_matrix<LhsType, b> and (dimension_size_of_index_is<RhsType, 0, 1> or dimension_size_of_index_is<RhsType, 1, 1>)) or
        ((dimension_size_of_index_is<LhsType, 0, 1> or dimension_size_of_index_is<LhsType, 1, 1>) and square_matrix<RhsType, b>));

    template<Likelihood b>
    static constexpr bool is_square =
      square_matrix<LhsType, Likelihood::maybe> and square_matrix<RhsType, Likelihood::maybe> and
      (b != Likelihood::definitely or not has_dynamic_dimensions<Eigen::CwiseBinaryOp<BinaryOp, LhsType, RhsType>> or
        square_matrix<LhsType, b> or square_matrix<RhsType, b>);

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

    template<typename Arg>
    static constexpr auto get_constant(const Arg& arg)
    {
      return Eigen3::FunctorTraits<BinaryOp, LhsType, RhsType>::template get_constant<false>(arg);
    }

    template<typename Arg>
    static constexpr auto get_constant_diagonal(const Arg& arg)
    {
      return Eigen3::FunctorTraits<BinaryOp, LhsType, RhsType>::template get_constant<true>(arg);
    }

    template<TriangleType t, Likelihood b>
    static constexpr bool is_triangular = Eigen3::FunctorTraits<BinaryOp, LhsType, RhsType>::template is_triangular<t, b>;

    static constexpr bool is_triangular_adapter = false;

    static constexpr bool is_hermitian = Eigen3::FunctorTraits<BinaryOp, LhsType, RhsType>::is_hermitian;
  };


} // namespace OpenKalman::interface

#endif //OPENKALMAN_EIGEN3_TRAITS_CWISEBINARYOP_HPP
