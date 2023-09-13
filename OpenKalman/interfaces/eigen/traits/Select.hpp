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
  struct IndexibleObjectTraits<Eigen::Select<ConditionMatrixType, ThenMatrixType, ElseMatrixType>>
    : Eigen3::IndexibleObjectTraitsBase<Eigen::Select<ConditionMatrixType, ThenMatrixType, ElseMatrixType>>
  {
  private:

    using T = Eigen::Select<ConditionMatrixType, ThenMatrixType, ElseMatrixType>;

  public:

    template<typename Arg, typename N>
    static constexpr auto get_index_descriptor(const Arg& arg, N n)
    {
      if constexpr (static_index_value<N>)
      {
        constexpr auto i = static_index_value_of_v<N>;
        if constexpr (not dynamic_dimension<ConditionMatrixType, i>) return OpenKalman::get_index_descriptor(arg.conditionMatrix(), n);
        else if constexpr (not dynamic_dimension<ThenMatrixType, i>) return OpenKalman::get_index_descriptor(arg.thenMatrix(), n);
        else return OpenKalman::get_index_descriptor(arg.elseMatrix(), n);
      }
      else
      {
        return OpenKalman::get_index_descriptor(arg.conditionMatrix(), n);
      }
    }


    template<Likelihood b>
    static constexpr bool is_one_by_one =
      one_by_one_matrix<ConditionMatrixType, Likelihood::maybe> and one_by_one_matrix<ThenMatrixType, Likelihood::maybe> and one_by_one_matrix<ElseMatrixType, Likelihood::maybe> and
      (b != Likelihood::definitely or one_by_one_matrix<ConditionMatrixType, b> or one_by_one_matrix<ThenMatrixType, b> or one_by_one_matrix<ElseMatrixType, b>);


    static constexpr bool has_runtime_parameters = false;


    using type = std::tuple<typename ConditionMatrixType::Nested, typename ThenMatrixType::Nested,
      typename ElseMatrixType::Nested>;


    template<std::size_t i, typename Arg>
    static decltype(auto) get_nested_matrix(Arg&& arg)
    {
      static_assert(i < 3);
      if constexpr (i == 0)
        return std::forward<Arg>(arg).conditionMatrix();
      else if constexpr (i == 1)
        return std::forward<Arg>(arg).thenMatrix();
      else
        return std::forward<Arg>(arg).elseMatrix();
    }


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
        return make_dense_writable_matrix_from(std::forward<Arg>(arg));
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
