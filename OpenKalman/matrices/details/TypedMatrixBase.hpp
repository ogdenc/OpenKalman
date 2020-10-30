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

    /// Copy constructor.
    TypedMatrixBase(const TypedMatrixBase& other) : Base(other.base_matrix()) {}

    /// Move constructor.
    TypedMatrixBase(TypedMatrixBase&& other) noexcept : Base(std::move(other).base_matrix()) {}

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

    using Base::operator=;

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

    auto operator()(std::size_t i, std::size_t j) &
    {
      return make_ElementSetter<not is_element_settable_v<Derived, 2>>(static_cast<Derived&>(*this), i, j);
    }

    auto operator()(std::size_t i, std::size_t j) &&
    {
      return make_ElementSetter<true>(static_cast<Derived&&>(*this), i, j);
    }

    auto operator()(std::size_t i, std::size_t j) const &
    {
      return make_ElementSetter<true>(static_cast<const Derived&>(*this), i, j);
    }

    auto operator()(std::size_t i, std::size_t j) const &&
    {
      return make_ElementSetter<true>(static_cast<const Derived&&>(*this), i, j);
    }

    auto operator[](std::size_t i) &
    {
      return make_ElementSetter<not is_element_settable_v<Derived, 1>>(static_cast<Derived&>(*this), i);
    }

    auto operator[](std::size_t i) &&
    {
      return make_ElementSetter<true>(static_cast<Derived&&>(*this), i);
    }

    auto operator[](std::size_t i) const &
    {
      return make_ElementSetter<true>(static_cast<const Derived&>(*this), i);
    }

    auto operator[](std::size_t i) const &&
    {
      return make_ElementSetter<true>(static_cast<const Derived&&>(*this), i);
    }

    auto operator()(std::size_t i) { return operator[](i); }

    auto operator()(std::size_t i) const { return operator[](i); }
  };


} // namespace OpenKalman


#endif //OPENKALMAN_TYPEDMATRIXBASE_H