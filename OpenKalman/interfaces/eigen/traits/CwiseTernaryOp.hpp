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

#ifndef OPENKALMAN_EIGEN_TRAITS_CWISETERNARYOP_HPP
#define OPENKALMAN_EIGEN_TRAITS_CWISETERNARYOP_HPP

#include <type_traits>


namespace OpenKalman::interface
{
  template<typename TernaryOp, typename Arg1, typename Arg2, typename Arg3>
  struct indexible_object_traits<Eigen::CwiseTernaryOp<TernaryOp, Arg1, Arg2, Arg3>>
    : Eigen3::indexible_object_traits_base<Eigen::CwiseTernaryOp<TernaryOp, Arg1, Arg2, Arg3>>
  {
  private:

    using T = Eigen::CwiseTernaryOp<TernaryOp, Arg1, Arg2, Arg3>;
    using Base = Eigen3::indexible_object_traits_base<T>;

  public:

    template<typename Arg, typename N>
    static constexpr auto get_vector_space_descriptor(const Arg& arg, N n)
    {
      if constexpr (static_index_value<N>)
      {
        if constexpr (not dynamic_dimension<Arg1, n>)
          return OpenKalman::get_vector_space_descriptor(arg.arg1(), n);
        else if constexpr (not dynamic_dimension<Arg2, n>)
          return OpenKalman::get_vector_space_descriptor(arg.arg2(), n);
        else
          return OpenKalman::get_vector_space_descriptor(arg.arg3(), n);
      }
      else return OpenKalman::get_vector_space_descriptor(arg.arg1(), n);
    }

    using dependents = std::tuple<typename T::Arg1Nested, typename T::Arg2Nested, typename T::Arg3Nested>;

    static constexpr bool has_runtime_parameters = false;

    template<std::size_t i, typename Arg>
    static decltype(auto) get_nested_matrix(Arg&& arg)
    {
      static_assert(i <= 2);
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

    template<typename Arg>
    static constexpr auto get_constant(const Arg& arg)
    {
      return Eigen3::FunctorTraits<TernaryOp, Arg1, Arg2, Arg3>::template get_constant<false>(arg);
    }

    template<typename Arg>
    static constexpr auto get_constant_diagonal(const Arg& arg)
    {
      return Eigen3::FunctorTraits<TernaryOp, Arg1, Arg2, Arg3>::template get_constant<true>(arg);
    }

    template<Likelihood b>
    static constexpr bool is_one_by_one =
      one_by_one_matrix<Arg1, Likelihood::maybe> and one_by_one_matrix<Arg2, Likelihood::maybe> and one_by_one_matrix<Arg3, Likelihood::maybe> and
      (b != Likelihood::definitely or one_by_one_matrix<Arg1, b> or one_by_one_matrix<Arg2, b> or one_by_one_matrix<Arg3, b>);

    template<TriangleType t, Likelihood b>
    static constexpr bool is_triangular = Eigen3::FunctorTraits<TernaryOp, Arg1, Arg2, Arg3>::template is_triangular<t, b>;

    static constexpr bool is_triangular_adapter = false;

    static constexpr bool is_hermitian = Eigen3::FunctorTraits<TernaryOp, Arg1, Arg2, Arg3>::is_hermitian;
  };

} // namespace OpenKalman::interface

#endif //OPENKALMAN_EIGEN_TRAITS_CWISETERNARYOP_HPP
