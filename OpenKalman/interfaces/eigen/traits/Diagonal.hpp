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
  struct IndexibleObjectTraits<Eigen::Diagonal<MatrixType, DiagIndex>>
    : Eigen3::IndexibleObjectTraitsBase<Eigen::Diagonal<MatrixType, DiagIndex>>
  {
    static constexpr std::size_t max_indices = 2;

    template<std::size_t N, typename Arg>
    static constexpr auto get_index_descriptor(const Arg& arg)
    {
      using Xpr = Eigen::Diagonal<MatrixType, DiagIndex>;
      constexpr Eigen::Index dim = N == 0 ? Xpr::RowsAtCompileTime : Xpr::ColsAtCompileTime;

      if constexpr (dim == Eigen::Dynamic)
      {
        if constexpr (N == 0) return static_cast<std::size_t>(arg.rows());
        else return static_cast<std::size_t>(arg.cols());
      }
      else return Dimensions<dim>{};
    }

    static constexpr bool has_runtime_parameters = DiagIndex == Eigen::DynamicIndex;

    using type = std::tuple<typename Eigen::internal::ref_selector<MatrixType>::non_const_type>;

    template<std::size_t i, typename Arg>
    static decltype(auto) get_nested_matrix(Arg&& arg)
    {
      static_assert(i == 0);
      return std::forward<Arg>(arg).nestedExpression();
    }

    // Rely on default for convert_to_self_contained. Should always convert to a dense, writable matrix.

    template<typename Arg>
    static constexpr auto get_constant(const Arg& arg)
    {
      if constexpr (constant_diagonal_matrix<MatrixType, CompileTimeStatus::any, Likelihood::maybe>)
      {
        if constexpr (DiagIndex == Eigen::DynamicIndex)
        {
          if (arg.index() == 0)
            return constant_diagonal_coefficient{arg.nestedExpression()}();
          else
            return scalar_type_of_t<MatrixType>{0};
        }
        else if constexpr (DiagIndex == 0)
          return constant_diagonal_coefficient{arg.nestedExpression()};
        else
          return internal::ScalarConstant<Likelihood::definitely, scalar_type_of_t<MatrixType>, 0>{};
      }
      else if constexpr (constant_matrix<MatrixType, CompileTimeStatus::any, Likelihood::maybe>)
      {
        return constant_coefficient{arg.nestedExpression()};
      }
      else
      {
        return std::monostate{};
      }
    }
  };

} // namespace OpenKalman::interface

#endif //OPENKALMAN_EIGEN_TRAITS_DIAGONAL_HPP
