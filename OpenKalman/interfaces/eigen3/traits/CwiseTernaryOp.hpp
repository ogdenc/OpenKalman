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
      return Eigen3::FunctorTraits<TernaryOp, Arg1, Arg2, Arg3>::template get_constant<constant_coefficient>(xpr);
    }

    constexpr auto get_constant_diagonal()
    {
      return Eigen3::FunctorTraits<TernaryOp, Arg1, Arg2, Arg3>::template get_constant<constant_diagonal_coefficient>(xpr);
    }
  };


  template<typename TernaryOp, typename Arg1, typename Arg2, typename Arg3>
  struct DiagonalTraits<Eigen::CwiseTernaryOp<TernaryOp, Arg1, Arg2, Arg3>>
  {
    static constexpr bool is_diagonal = Eigen3::FunctorTraits<TernaryOp, Arg1, Arg2, Arg3>::is_diagonal;
  };


  template<typename TernaryOp, typename Arg1, typename Arg2, typename Arg3>
  struct TriangularTraits<Eigen::CwiseTernaryOp<TernaryOp, Arg1, Arg2, Arg3>>
  {
    static constexpr TriangleType triangle_type = Eigen3::FunctorTraits<TernaryOp, Arg1, Arg2, Arg3>::triangle_type;

    static constexpr bool is_triangular_adapter = false;
  };


  template<typename TernaryOp, typename Arg1, typename Arg2, typename Arg3>
  struct HermitianTraits<Eigen::CwiseTernaryOp<TernaryOp, Arg1, Arg2, Arg3>>
  {
    static constexpr bool is_hermitian = Eigen3::FunctorTraits<TernaryOp, Arg1, Arg2, Arg3>::is_hermitian;

    static constexpr TriangleType adapter_type = TriangleType::none;
  };

} // namespace OpenKalman::interface

#endif //OPENKALMAN_EIGEN3_TRAITS_CWISETERNARYOP_HPP
