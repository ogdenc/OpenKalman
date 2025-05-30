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
      if constexpr (square_shaped<ConditionMatrixType> or square_shaped<ThenMatrixType> or square_shaped<ElseMatrixType>)
        return internal::best_vector_space_descriptor(
          OpenKalman::get_vector_space_descriptor<0>(arg.conditionMatrix()),
          OpenKalman::get_vector_space_descriptor<0>(arg.thenMatrix()),
          OpenKalman::get_vector_space_descriptor<0>(arg.elseMatrix()),
          OpenKalman::get_vector_space_descriptor<1>(arg.conditionMatrix()),
          OpenKalman::get_vector_space_descriptor<1>(arg.thenMatrix()),
          OpenKalman::get_vector_space_descriptor<1>(arg.elseMatrix()));
      else
        return internal::best_vector_space_descriptor(
          OpenKalman::get_vector_space_descriptor(arg.conditionMatrix(), n),
          OpenKalman::get_vector_space_descriptor(arg.thenMatrix(), n),
          OpenKalman::get_vector_space_descriptor(arg.elseMatrix(), n));
    }


    // nested_object not defined


    template<typename Arg>
    static constexpr auto get_constant(const Arg& arg)
    {
      if constexpr (constant_matrix<ConditionMatrixType>)
      {
        if constexpr (static_cast<bool>(constant_coefficient_v<ConditionMatrixType>))
          return constant_coefficient{arg.thenMatrix()};
        else
          return constant_coefficient{arg.elseMatrix()};
      }
      else if constexpr (constant_matrix<ThenMatrixType> and constant_matrix<ElseMatrixType>)
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
      if constexpr (constant_matrix<ConditionMatrixType>)
      {
        if constexpr (static_cast<bool>(constant_coefficient_v<ConditionMatrixType>))
          return constant_diagonal_coefficient{arg.thenMatrix()};
        else
          return constant_diagonal_coefficient{arg.elseMatrix()};
      }
      else if constexpr (constant_diagonal_matrix<ThenMatrixType> and constant_diagonal_matrix<ElseMatrixType>)
      {
        if constexpr (constant_diagonal_coefficient_v<ThenMatrixType> == constant_diagonal_coefficient_v<ElseMatrixType>)
          return constant_diagonal_coefficient{arg.thenMatrix()};
        else return std::monostate{};
      }
      else return std::monostate{};
    }


    template<Applicability b>
    static constexpr bool one_dimensional =
      OpenKalman::one_dimensional<ConditionMatrixType, Applicability::permitted> and
      OpenKalman::one_dimensional<ThenMatrixType, Applicability::permitted> and
      OpenKalman::one_dimensional<ElseMatrixType, Applicability::permitted> and
      (b != Applicability::guaranteed or
        not has_dynamic_dimensions<Xpr> or
        OpenKalman::one_dimensional<ConditionMatrixType> or
        OpenKalman::one_dimensional<ThenMatrixType> or
        OpenKalman::one_dimensional<ElseMatrixType>);


    template<Applicability b>
    static constexpr bool is_square =
      square_shaped<ConditionMatrixType, Applicability::permitted> and
      square_shaped<ThenMatrixType, Applicability::permitted> and
      square_shaped<ElseMatrixType, Applicability::permitted> and
      (b != Applicability::guaranteed or
        not has_dynamic_dimensions<Xpr> or
        square_shaped<ConditionMatrixType, b> or
        square_shaped<ThenMatrixType, b> or
        square_shaped<ElseMatrixType, b>);


    template<TriangleType t>
    static constexpr bool is_triangular =
      [](){
        if constexpr (constant_matrix<ConditionMatrixType>)
          return triangular_matrix<std::conditional_t<static_cast<bool>(constant_coefficient_v<ConditionMatrixType>),
            ThenMatrixType, ElseMatrixType>, t>;
        else return false;
      }();


    static constexpr bool is_triangular_adapter = false;


    static constexpr bool is_hermitian =
      (values::fixed<constant_coefficient<ConditionMatrixType>> and
        [](){
          if constexpr (constant_matrix<ConditionMatrixType>)
            return hermitian_matrix<std::conditional_t<static_cast<bool>(constant_coefficient_v<ConditionMatrixType>),
              ThenMatrixType, ElseMatrixType>, Applicability::permitted>;
          else return false;
        }()) or
      (hermitian_matrix<ConditionMatrixType, Applicability::permitted> and hermitian_matrix<ThenMatrixType, Applicability::permitted> and
        hermitian_matrix<ElseMatrixType, Applicability::permitted> and
        (not values::fixed<constant_coefficient<ConditionMatrixType>>));
  };


} // namespace OpenKalman::interface

#endif //OPENKALMAN_EIGEN_TRAITS_SELECT_HPP
