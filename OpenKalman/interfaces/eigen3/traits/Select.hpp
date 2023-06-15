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

#ifndef OPENKALMAN_EIGEN3_TRAITS_SELECT_HPP
#define OPENKALMAN_EIGEN3_TRAITS_SELECT_HPP

#include <type_traits>


namespace OpenKalman::interface
{
  template<typename ConditionMatrixType, typename ThenMatrixType, typename ElseMatrixType>
  struct IndexTraits<Eigen::Select<ConditionMatrixType, ThenMatrixType, ElseMatrixType>>
  {
    static constexpr std::size_t max_indices = max_indices_of_v<ConditionMatrixType>;

    template<std::size_t N, typename Arg>
    static constexpr auto get_index_descriptor(const Arg& arg)
    {
      if constexpr (not dynamic_dimension<ConditionMatrixType, N>) return OpenKalman::get_index_descriptor(arg.conditionMatrix());
      else if constexpr (not dynamic_dimension<ThenMatrixType, N>) return OpenKalman::get_index_descriptor(arg.thenMatrix());
      else return OpenKalman::get_index_descriptor<N>(arg.elseMatrix());
    }

    template<Likelihood b>
    static constexpr bool is_one_by_one =
      one_by_one_matrix<ConditionMatrixType, Likelihood::maybe> and one_by_one_matrix<ThenMatrixType, Likelihood::maybe> and one_by_one_matrix<ElseMatrixType, Likelihood::maybe> and
      (b != Likelihood::definitely or one_by_one_matrix<ConditionMatrixType, b> or one_by_one_matrix<ThenMatrixType, b> or one_by_one_matrix<ElseMatrixType, b>);
  };


  template<typename ConditionMatrixType, typename ThenMatrixType, typename ElseMatrixType>
  struct Dependencies<Eigen::Select<ConditionMatrixType, ThenMatrixType, ElseMatrixType>>
  {
  private:

    using T = Eigen::Select<ConditionMatrixType, ThenMatrixType, ElseMatrixType>;

  public:

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
  };


  // --- constant_coefficient --- //

  template<typename ConditionMatrixType, typename ThenMatrixType, typename ElseMatrixType>
  struct SingleConstant<Eigen::Select<ConditionMatrixType, ThenMatrixType, ElseMatrixType>>
  {
    const Eigen::Select<ConditionMatrixType, ThenMatrixType, ElseMatrixType>& xpr;

    constexpr auto get_constant()
    {
      if constexpr (constant_matrix<ConditionMatrixType, CompileTimeStatus::any, Likelihood::maybe>)
      {
        if constexpr (static_cast<bool>(constant_coefficient_v<ConditionMatrixType>))
          return constant_coefficient{xpr.thenMatrix()};
        else
          return constant_coefficient{xpr.elseMatrix()};
      }
      else if constexpr (constant_matrix<ThenMatrixType, CompileTimeStatus::any, Likelihood::maybe> and
        constant_matrix<ElseMatrixType, CompileTimeStatus::any, Likelihood::maybe>)
      {
        if constexpr (constant_coefficient_v<ThenMatrixType> == constant_coefficient_v<ElseMatrixType>)
          return constant_coefficient{xpr.thenMatrix()};
        else return std::monostate{};
      }
      else return std::monostate{};
    }

    constexpr auto get_constant_diagonal()
    {
      if constexpr (constant_matrix<ConditionMatrixType, CompileTimeStatus::any, Likelihood::maybe>)
      {
        if constexpr (static_cast<bool>(constant_coefficient_v<ConditionMatrixType>))
          return constant_diagonal_coefficient{xpr.thenMatrix()};
        else
          return constant_diagonal_coefficient{xpr.elseMatrix()};
      }
      else if constexpr (constant_diagonal_matrix<ThenMatrixType, CompileTimeStatus::any, Likelihood::maybe> and
        constant_diagonal_matrix<ElseMatrixType, CompileTimeStatus::any, Likelihood::maybe>)
      {
        if constexpr (constant_diagonal_coefficient_v<ThenMatrixType> == constant_diagonal_coefficient_v<ElseMatrixType>)
          return constant_diagonal_coefficient{xpr.thenMatrix()};
        else return std::monostate{};
      }
      else return std::monostate{};
    }
  };


#ifdef __cpp_concepts
  template<constant_matrix<CompileTimeStatus::known, Likelihood::maybe> ConditionMatrixType, typename ThenMatrixType, typename ElseMatrixType>
  struct TriangularTraits<Eigen::Select<ConditionMatrixType, ThenMatrixType, ElseMatrixType>>
#else
  template<typename ConditionMatrixType, typename ThenMatrixType, typename ElseMatrixType>
  struct TriangularTraits<Eigen::Select<ConditionMatrixType, ThenMatrixType, ElseMatrixType>,
    std::enable_if_t<constant_matrix<ConditionMatrixType, CompileTimeStatus::known, Likelihood::maybe>>>
#endif
  {
    template<TriangleType t, Likelihood b>
    static constexpr bool is_triangular =
      (triangular_matrix<ThenMatrixType, t, b> and constant_coefficient_v<ConditionMatrixType>) or
      (triangular_matrix<ElseMatrixType, t, b> and not constant_coefficient_v<ConditionMatrixType>);

    static constexpr bool is_triangular_adapter = false;
  };


#ifdef __cpp_concepts
  template<constant_matrix<CompileTimeStatus::known, Likelihood::maybe> ConditionMatrixType, typename ThenMatrixType, typename ElseMatrixType> requires
    (hermitian_matrix<ThenMatrixType, Likelihood::maybe> and static_cast<bool>(constant_coefficient_v<ConditionMatrixType>)) or
    (hermitian_matrix<ElseMatrixType, Likelihood::maybe> and not static_cast<bool>(constant_coefficient_v<ConditionMatrixType>))
  struct HermitianTraits<Eigen::Select<ConditionMatrixType, ThenMatrixType, ElseMatrixType>>
#else
  template<typename ConditionMatrixType, typename ThenMatrixType, typename ElseMatrixType>
  struct HermitianTraits<Eigen::Select<ConditionMatrixType, ThenMatrixType, ElseMatrixType>, std::enable_if_t<
    constant_matrix<ConditionMatrixType, CompileTimeStatus::known, Likelihood::maybe> and
    ((hermitian_matrix<ThenMatrixType, Likelihood::maybe> and static_cast<bool>(constant_coefficient<ConditionMatrixType>::value)) or
     (hermitian_matrix<ElseMatrixType, Likelihood::maybe> and not static_cast<bool>(constant_coefficient<ConditionMatrixType>::value)))>>
#endif
  {
    static constexpr bool is_hermitian = true;
  };


#ifdef __cpp_concepts
  template<hermitian_matrix<Likelihood::maybe> ConditionMatrixType, hermitian_matrix<Likelihood::maybe> ThenMatrixType,
      hermitian_matrix<Likelihood::maybe> ElseMatrixType> requires
    (not constant_matrix<ConditionMatrixType, CompileTimeStatus::known, Likelihood::maybe>)
  struct HermitianTraits<Eigen::Select<ConditionMatrixType, ThenMatrixType, ElseMatrixType>>
#else
  template<typename ConditionMatrixType, typename ThenMatrixType, typename ElseMatrixType>
  struct HermitianTraits<Eigen::Select<ConditionMatrixType, ThenMatrixType, ElseMatrixType>,
    std::enable_if_t<hermitian_matrix<ConditionMatrixType, Likelihood::maybe> and hermitian_matrix<ThenMatrixType, Likelihood::maybe> and
      hermitian_matrix<ElseMatrixType, Likelihood::maybe> and
      (not constant_matrix<ConditionMatrixType, CompileTimeStatus::known, Likelihood::maybe>)>>
#endif
  {
    static constexpr bool is_hermitian = true;
  };


} // namespace OpenKalman::interface

#endif //OPENKALMAN_EIGEN3_TRAITS_SELECT_HPP
