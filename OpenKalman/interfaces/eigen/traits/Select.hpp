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
 * \brief Type traits as applied to Eigen::Select.
 */

#ifndef OPENKALMAN_EIGEN_TRAITS_SELECT_HPP
#define OPENKALMAN_EIGEN_TRAITS_SELECT_HPP

#include <type_traits>


namespace OpenKalman::interface
{
  template<typename ConditionMatrixType, typename ThenMatrixType, typename ElseMatrixType>
  struct indexible_object_traits<Eigen::Select<ConditionMatrixType, ThenMatrixType, ElseMatrixType>>
    : Eigen3::indexible_object_traits_base<Eigen::Select<ConditionMatrixType, ThenMatrixType, ElseMatrixType>>
  {
  private:

    using Xpr = Eigen::Select<ConditionMatrixType, ThenMatrixType, ElseMatrixType>;
    using Base = Eigen3::indexible_object_traits_base<Xpr>;

  public:

    template<typename Arg, typename N>
    static constexpr auto get_vector_space_descriptor(const Arg& arg, N n)
    {
      if constexpr (static_index_value<N>)
      {
        if constexpr (not dynamic_dimension<ConditionMatrixType, n>) return OpenKalman::get_vector_space_descriptor(arg.conditionMatrix(), n);
        else if constexpr (not dynamic_dimension<ThenMatrixType, n>) return OpenKalman::get_vector_space_descriptor(arg.thenMatrix(), n);
        else return OpenKalman::get_vector_space_descriptor(arg.elseMatrix(), n);
      }
      else
      {
        return OpenKalman::get_vector_space_descriptor(arg.conditionMatrix(), n);
      }
    }


    using dependents = std::tuple<typename ConditionMatrixType::Nested, typename ThenMatrixType::Nested,
      typename ElseMatrixType::Nested>;


    static constexpr bool has_runtime_parameters = false;


    // nested_object not defined


    template<typename Arg>
    static auto convert_to_self_contained(Arg&& arg)
    {
      using N = Eigen::Select<equivalent_self_contained_t<ConditionMatrixType>,
        equivalent_self_contained_t<ThenMatrixType>, equivalent_self_contained_t<ElseMatrixType>>;
      // Do a partial evaluation as long as at least two arguments are already self-contained.
      if constexpr (
        ((self_contained<ConditionMatrixType> ? 1 : 0) + (self_contained<ThenMatrixType> ? 1 : 0) +
          (self_contained<ElseMatrixType> ? 1 : 0) >= 2) and
        not std::is_lvalue_reference_v<typename equivalent_self_contained_t<ConditionMatrixType>::Nested> and
        not std::is_lvalue_reference_v<typename equivalent_self_contained_t<ThenMatrixType>::Nested> and
        not std::is_lvalue_reference_v<typename equivalent_self_contained_t<ElseMatrixType>::Nested>)
      {
        return N {make_self_contained(arg.arg1()), make_self_contained(arg.arg2()), make_self_contained(arg.arg3()),
                  arg.functor()};
      }
      else
      {
        return make_dense_object(std::forward<Arg>(arg));
      }
    }


    template<typename Arg>
    static constexpr auto get_constant(const Arg& arg)
    {
      if constexpr (constant_matrix<ConditionMatrixType, CompileTimeStatus::any, Likelihood::maybe>)
      {
        if constexpr (static_cast<bool>(constant_coefficient_v<ConditionMatrixType>))
          return constant_coefficient{arg.thenMatrix()};
        else
          return constant_coefficient{arg.elseMatrix()};
      }
      else if constexpr (constant_matrix<ThenMatrixType, CompileTimeStatus::any, Likelihood::maybe> and
        constant_matrix<ElseMatrixType, CompileTimeStatus::any, Likelihood::maybe>)
      {
        if constexpr (constant_coefficient_v<ThenMatrixType> == constant_coefficient_v<ElseMatrixType>)
          return constant_coefficient{arg.thenMatrix()};
        else return std::monostate{};
      }
      else return std::monostate{};
    }


    template<typename Arg>
    static constexpr auto get_constant_diagonal(const Arg& arg)
    {
      if constexpr (constant_matrix<ConditionMatrixType, CompileTimeStatus::any, Likelihood::maybe>)
      {
        if constexpr (static_cast<bool>(constant_coefficient_v<ConditionMatrixType>))
          return constant_diagonal_coefficient{arg.thenMatrix()};
        else
          return constant_diagonal_coefficient{arg.elseMatrix()};
      }
      else if constexpr (constant_diagonal_matrix<ThenMatrixType, CompileTimeStatus::any, Likelihood::maybe> and
        constant_diagonal_matrix<ElseMatrixType, CompileTimeStatus::any, Likelihood::maybe>)
      {
        if constexpr (constant_diagonal_coefficient_v<ThenMatrixType> == constant_diagonal_coefficient_v<ElseMatrixType>)
          return constant_diagonal_coefficient{arg.thenMatrix()};
        else return std::monostate{};
      }
      else return std::monostate{};
    }


    template<Likelihood b>
    static constexpr bool one_dimensional =
      OpenKalman::one_dimensional<ConditionMatrixType, Likelihood::maybe> and
      OpenKalman::one_dimensional<ThenMatrixType, Likelihood::maybe> and
      OpenKalman::one_dimensional<ElseMatrixType, Likelihood::maybe> and
      (b != Likelihood::definitely or
        OpenKalman::one_dimensional<ConditionMatrixType, b> or
        OpenKalman::one_dimensional<ThenMatrixType, b> or
        OpenKalman::one_dimensional<ElseMatrixType, b>);


    template<TriangleType t, Likelihood b>
    static constexpr bool is_triangular =
      [](){
        if constexpr (constant_matrix<ConditionMatrixType, CompileTimeStatus::any, Likelihood::maybe>)
          return triangular_matrix<std::conditional_t<static_cast<bool>(constant_coefficient_v<ConditionMatrixType>),
            ThenMatrixType, ElseMatrixType>, t, b>;
        else return false;
      }();


    static constexpr bool is_triangular_adapter = false;


    static constexpr bool is_hermitian =
      (constant_matrix<ConditionMatrixType, CompileTimeStatus::known, Likelihood::maybe> and
        [](){
          if constexpr (constant_matrix<ConditionMatrixType, CompileTimeStatus::any, Likelihood::maybe>)
            return hermitian_matrix<std::conditional_t<static_cast<bool>(constant_coefficient_v<ConditionMatrixType>),
              ThenMatrixType, ElseMatrixType>, Likelihood::maybe>;
          else return false;
        }()) or
      (hermitian_matrix<ConditionMatrixType, Likelihood::maybe> and hermitian_matrix<ThenMatrixType, Likelihood::maybe> and
        hermitian_matrix<ElseMatrixType, Likelihood::maybe> and
        (not constant_matrix<ConditionMatrixType, CompileTimeStatus::known, Likelihood::maybe>));
  };


} // namespace OpenKalman::interface

#endif //OPENKALMAN_EIGEN_TRAITS_SELECT_HPP
