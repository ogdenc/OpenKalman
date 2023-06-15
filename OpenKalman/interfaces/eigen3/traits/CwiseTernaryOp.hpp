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

#ifndef OPENKALMAN_EIGEN3_TRAITS_CWISETERNARYOP_HPP
#define OPENKALMAN_EIGEN3_TRAITS_CWISETERNARYOP_HPP

#include <type_traits>


namespace OpenKalman::interface
{
  namespace EGI = Eigen::internal;


  template<typename TernaryOp, typename Arg1, typename Arg2, typename Arg3>
  struct IndexTraits<Eigen::CwiseTernaryOp<TernaryOp, Arg1, Arg2, Arg3>>
  {
    static constexpr std::size_t max_indices = std::max({max_indices_of_v<Arg1>, max_indices_of_v<Arg2>, max_indices_of_v<Arg3>});

    template<std::size_t N, typename Arg>
    static constexpr auto get_index_descriptor(const Arg& arg)
    {
      if constexpr (not dynamic_dimension<Arg1, N>) return OpenKalman::get_index_descriptor<N>(arg.arg1());
      else if constexpr (not dynamic_dimension<Arg2, N>) return OpenKalman::get_index_descriptor<N>(arg.arg2());
      else return OpenKalman::get_index_descriptor<N>(arg.arg3());
    }

    template<Likelihood b>
    static constexpr bool is_one_by_one =
      one_by_one_matrix<Arg1, Likelihood::maybe> and one_by_one_matrix<Arg2, Likelihood::maybe> and one_by_one_matrix<Arg3, Likelihood::maybe> and
      (b != Likelihood::definitely or one_by_one_matrix<Arg1, b> or one_by_one_matrix<Arg2, b> or one_by_one_matrix<Arg3, b>);
  };


  template<typename TernaryOp, typename Arg1, typename Arg2, typename Arg3>
  struct Dependencies<Eigen::CwiseTernaryOp<TernaryOp, Arg1, Arg2, Arg3>>
  {
  private:

    using T = Eigen::CwiseTernaryOp<TernaryOp, Arg1, Arg2, Arg3>;

  public:

    static constexpr bool has_runtime_parameters = false;
    using type = std::tuple<typename T::Arg1Nested, typename T::Arg2Nested, typename T::Arg3Nested>;

    template<std::size_t i, typename Arg>
    static decltype(auto) get_nested_matrix(Arg&& arg)
    {
      static_assert(i < 3);
      if constexpr (i == 0)
        return std::forward<Arg>(arg).arg1();
      else if constexpr (i == 1)
        return std::forward<Arg>(arg).arg2();
      else
        return std::forward<Arg>(arg).arg3();
    }

    template<typename Arg>
    static auto convert_to_self_contained(Arg&& arg)
    {
      using N = Eigen::CwiseTernaryOp<TernaryOp,
        equivalent_self_contained_t<Arg1>, equivalent_self_contained_t<Arg2>, equivalent_self_contained_t<Arg3>>;
      // Do a partial evaluation as long as at least two arguments are already self-contained.
      if constexpr (
        ((self_contained<Arg1> ? 1 : 0) + (self_contained<Arg2> ? 1 : 0) + (self_contained<Arg3> ? 1 : 0) >= 2) and
        not std::is_lvalue_reference_v<typename N::Arg1Nested> and
        not std::is_lvalue_reference_v<typename N::Arg2Nested> and
        not std::is_lvalue_reference_v<typename N::Arg3Nested>)
      {
        return N {make_self_contained(arg.arg1()), make_self_contained(arg.arg2()), make_self_contained(arg.arg3()),
                  arg.functor()};
      }
      else
      {
        return make_dense_writable_matrix_from(std::forward<Arg>(arg));
      }
    }
  };


  template<typename TernaryOp, typename Arg1, typename Arg2, typename Arg3>
  struct SingleConstant<Eigen::CwiseTernaryOp<TernaryOp, Arg1, Arg2, Arg3>>
  {
    const Eigen::CwiseTernaryOp<TernaryOp, Arg1, Arg2, Arg3>& xpr;

    constexpr auto get_constant()
    {
      return Eigen3::FunctorTraits<TernaryOp, Arg1, Arg2, Arg3>::template get_constant<false>(xpr);
    }

    constexpr auto get_constant_diagonal()
    {
      return Eigen3::FunctorTraits<TernaryOp, Arg1, Arg2, Arg3>::template get_constant<true>(xpr);
    }
  };


  template<typename TernaryOp, typename Arg1, typename Arg2, typename Arg3>
  struct TriangularTraits<Eigen::CwiseTernaryOp<TernaryOp, Arg1, Arg2, Arg3>>
  {
    template<TriangleType t, Likelihood b>
    static constexpr bool is_triangular = Eigen3::FunctorTraits<TernaryOp, Arg1, Arg2, Arg3>::template is_triangular<t, b>;

    static constexpr bool is_triangular_adapter = false;
  };


  template<typename TernaryOp, typename Arg1, typename Arg2, typename Arg3>
  struct HermitianTraits<Eigen::CwiseTernaryOp<TernaryOp, Arg1, Arg2, Arg3>>
  {
    static constexpr bool is_hermitian = Eigen3::FunctorTraits<TernaryOp, Arg1, Arg2, Arg3>::is_hermitian;
  };

} // namespace OpenKalman::interface

#endif //OPENKALMAN_EIGEN3_TRAITS_CWISETERNARYOP_HPP
