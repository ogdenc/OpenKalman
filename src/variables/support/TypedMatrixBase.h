/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2018-2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef OPENKALMAN_TYPEDMATRIXBASE_H
#define OPENKALMAN_TYPEDMATRIXBASE_H

#include "variables/support/MatrixBase.h"
#include "variables/support/OpenKalman-coefficients.h"

namespace OpenKalman::internal
{
  /// Base class for means or matrices.
  template<
    typename Derived,
    typename RowCoeffs,
    typename ColCoeffs,
    typename NestedType>
  struct TypedMatrixBase : internal::MatrixBase<Derived, NestedType>
  {
    using Base = internal::MatrixBase<Derived, NestedType>;
    using RowCoefficients = RowCoeffs;
    using ColumnCoefficients = ColCoeffs;
    using BaseMatrix = NestedType; ///< The nested class.
    using Scalar = typename MatrixTraits<BaseMatrix>::Scalar; ///< Scalar type for this variable.
    static constexpr auto dimension = MatrixTraits<BaseMatrix>::dimension; ///< Dimension of the vector.
    static constexpr auto columns = MatrixTraits<BaseMatrix>::columns; ///< Number of columns.

    /***************
     * Constructors
     ***************/

    /// Default constructor.
    TypedMatrixBase() : Base() {}

    /// Construct from a typed matrix base.
    template<typename Arg, std::enable_if_t<is_typed_matrix_base_v<Arg>, int> = 0>
    TypedMatrixBase(Arg&& arg) noexcept : Base(std::forward<Arg>(arg))
    {
      static_assert(MatrixTraits<Arg>::dimension == dimension);
      static_assert(MatrixTraits<Arg>::columns == columns);
    }

    /// Construct from a list of coefficients.
    template<typename ... Args, std::enable_if_t<std::conjunction_v<std::is_convertible<Args, const Scalar>...>, int> = 0>
    TypedMatrixBase(Args ... args) : Base(MatrixTraits<BaseMatrix>::make(args...)) {}

    /**********************
     * Assignment Operators
     **********************/

    /// Increment from another vector.
    template<typename Arg, std::enable_if_t<is_typed_matrix_v<Arg>, int> = 0>
    auto& operator+=(Arg&& other) noexcept
    {
      static_assert(OpenKalman::is_equivalent_v<typename MatrixTraits<Arg>::RowCoefficients, RowCoefficients>);
      static_assert(OpenKalman::is_equivalent_v<typename MatrixTraits<Arg>::ColumnCoefficients, ColumnCoefficients>);
      this->base_matrix() += std::forward<Arg>(other).base_matrix();
      return *this;
    }

    /// Add a stochastic value to each column of the matrix, based on a distribution.
    template<typename Arg, std::enable_if_t<is_distribution_v<Arg>, int> = 0>
    auto& operator+=(const Arg& arg) noexcept
    {
      static_assert(is_equivalent_v<typename DistributionTraits<Arg>::Coefficients, RowCoefficients>);
      static_assert(ColumnCoefficients::axes_only);
      static_assert(not is_Euclidean_transformed_v<Derived>);
      apply_columnwise(this->base_matrix(), [&arg](auto& col){ col += arg().base_matrix(); });
      return *this;
    }

    /// Decrement from another vector.
    template<typename Arg, std::enable_if_t<is_typed_matrix_v<Arg>, int> = 0>
    auto& operator-=(Arg&& other) noexcept
    {
      static_assert(OpenKalman::is_equivalent_v<typename MatrixTraits<Arg>::RowCoefficients, RowCoefficients>);
      static_assert(OpenKalman::is_equivalent_v<typename MatrixTraits<Arg>::ColumnCoefficients, ColumnCoefficients>);
      this->base_matrix() -= std::forward<Arg>(other).base_matrix();
      return *this;
    }

    /// Subtract a stochastic value to each column of the matrix, based on a distribution.
    template<typename Arg, std::enable_if_t<is_distribution_v<Arg>, int> = 0>
    auto& operator-=(const Arg& arg) noexcept
    {
      static_assert(is_equivalent_v<typename DistributionTraits<Arg>::Coefficients, RowCoefficients>);
      static_assert(ColumnCoefficients::axes_only);
      static_assert(not is_Euclidean_transformed_v<Derived>);
      apply_columnwise(this->base_matrix(), [&arg](auto& col){ col -= arg().base_matrix(); });
      return *this;
    }

    /// Multiply by a scale factor.
    template<typename S, std::enable_if_t<std::is_convertible_v<S, Scalar>, int> = 0>
    auto& operator*=(const S s)
    {
      this->base_matrix() *= s;
      return *this;
    }

    /// Divide by a scale factor.
    template<typename S, std::enable_if_t<std::is_convertible_v<S, Scalar>, int> = 0>
    auto& operator/=(const S s)
    {
      this->base_matrix() /= s;
      return *this;
    }

  };


} // namespace OpenKalman


#endif //OPENKALMAN_TYPEDMATRIXBASE_H
