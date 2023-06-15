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

#ifndef OPENKALMAN_EIGEN3_TRAITS_DIAGONAL_HPP
#define OPENKALMAN_EIGEN3_TRAITS_DIAGONAL_HPP

#include <type_traits>


namespace OpenKalman::interface
{
#ifndef __cpp_concepts
  template<typename MatrixType, int DiagIndex>
  struct IndexTraits<Eigen::Diagonal<MatrixType, DiagIndex>>
    : detail::IndexTraits_Eigen_default<Eigen::Diagonal<MatrixType, DiagIndex>> {};
#endif


  template<typename MatrixType, int DiagIndex>
  struct Dependencies<Eigen::Diagonal<MatrixType, DiagIndex>>
  {
    static constexpr bool has_runtime_parameters = DiagIndex == Eigen::DynamicIndex;

    using type = std::tuple<typename Eigen::internal::ref_selector<MatrixType>::non_const_type>;

    template<std::size_t i, typename Arg>
    static decltype(auto) get_nested_matrix(Arg&& arg)
    {
      static_assert(i == 0);
      return std::forward<Arg>(arg).nestedExpression();
    }

    // Rely on default for convert_to_self_contained. Should always convert to a dense, writable matrix.

  };


  template<typename MatrixType, int DiagIndex>
  struct SingleConstant<Eigen::Diagonal<MatrixType, DiagIndex>>
  {
    const Eigen::Diagonal<MatrixType, DiagIndex>& xpr;

    constexpr auto get_constant()
    {
      if constexpr (constant_diagonal_matrix<MatrixType, CompileTimeStatus::any, Likelihood::maybe>)
      {
        if constexpr (DiagIndex == Eigen::DynamicIndex)
        {
          if (xpr.index() == 0)
            return constant_diagonal_coefficient{xpr.nestedExpression()}();
          else
            return scalar_type_of_t<MatrixType>{0};
        }
        else if constexpr (DiagIndex == 0)
          return constant_diagonal_coefficient{xpr.nestedExpression()};
        else
          return internal::ScalarConstant<Likelihood::definitely, scalar_type_of_t<MatrixType>, 0>{};
      }
      else if constexpr (constant_matrix<MatrixType, CompileTimeStatus::any, Likelihood::maybe>)
      {
        return constant_coefficient{xpr.nestedExpression()};
      }
      else
      {
        return std::monostate{};
      }
    }
  };

} // namespace OpenKalman::interface

#endif //OPENKALMAN_EIGEN3_TRAITS_DIAGONAL_HPP
