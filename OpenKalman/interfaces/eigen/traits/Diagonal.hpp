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
 * \brief Type traits as applied to Eigen::Diagonal.
 */

#ifndef OPENKALMAN_EIGEN_TRAITS_DIAGONAL_HPP
#define OPENKALMAN_EIGEN_TRAITS_DIAGONAL_HPP

#include <type_traits>


namespace OpenKalman::interface
{
  template<typename MatrixType, int DiagIndex>
  struct indexible_object_traits<Eigen::Diagonal<MatrixType, DiagIndex>>
    : Eigen3::indexible_object_traits_base<Eigen::Diagonal<MatrixType, DiagIndex>>
  {
  private:

    using Xpr = Eigen::Diagonal<MatrixType, DiagIndex>;
    using Base = Eigen3::indexible_object_traits_base<Xpr>;

  public:

    using dependents = std::tuple<typename Eigen::internal::ref_selector<MatrixType>::non_const_type>;

    static constexpr bool has_runtime_parameters = DiagIndex == Eigen::DynamicIndex;


    template<typename Arg>
    static decltype(auto) nested_object(Arg&& arg)
    {
      return std::forward<Arg>(arg).nestedExpression();
    }


    // Rely on default for convert_to_self_contained. Should always convert to a dense, writable matrix.


    template<typename Arg>
    static constexpr auto get_constant(const Arg& arg)
    {
      using Scalar = scalar_type_of_t<MatrixType>;

      if constexpr (constant_diagonal_matrix<MatrixType> and DiagIndex == 0)
      {
        return constant_diagonal_coefficient{arg.nestedExpression()};
      }
      else if constexpr (zero<MatrixType> or (constant_diagonal_matrix<MatrixType> and DiagIndex != Eigen::DynamicIndex and
        ((DiagIndex < 0 and (dynamic_dimension<MatrixType, 0> or DiagIndex > -int(index_dimension_of_v<MatrixType, 0>))) or
          (DiagIndex > 0 and (dynamic_dimension<MatrixType, 1> or DiagIndex < int(index_dimension_of_v<MatrixType, 1>))))))
      {
        if constexpr (DiagIndex < 0 and dynamic_dimension<MatrixType, 0>)
        {
          if (DiagIndex <= -arg.nestedExpression().rows()) throw std::out_of_range{"DiagIndex in Eigen::Diagonal is too low for MatrixType."};
        }
        else if constexpr (DiagIndex > 0 and dynamic_dimension<MatrixType, 1>)
        {
          if (DiagIndex >= arg.nestedExpression().cols()) throw std::out_of_range{"DiagIndex in Eigen::Diagonal is too high for MatrixType."};
        }

        return values::ScalarConstant<Scalar, 0>{};
      }
      else if constexpr (constant_diagonal_matrix<MatrixType> and DiagIndex == Eigen::DynamicIndex)
      {
        if (arg.index() == 0) return static_cast<Scalar>(constant_diagonal_coefficient{arg.nestedExpression()});
        else if (arg.index() > -arg.nestedExpression().rows() and arg.index() < arg.nestedExpression().cols()) return Scalar(0);
        else throw std::out_of_range {"Dynamic index for Eigen::Diagonal is out of range."};
      }
      else if constexpr (constant_matrix<MatrixType, ConstantType::any>)
      {
        return constant_coefficient{arg.nestedExpression()};
      }
      else
      {
        return std::monostate{};
      }
    }


    template<Qualification b>
    static constexpr bool one_dimensional = dimension_size_of_index_is<Xpr, 0, 1, b>;


    template<Qualification b>
    static constexpr bool is_square = one_dimensional<b>;

  };

} // namespace OpenKalman::interface

#endif //OPENKALMAN_EIGEN_TRAITS_DIAGONAL_HPP
