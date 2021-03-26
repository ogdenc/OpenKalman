/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2018-2021 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef OPENKALMAN_TYPEDMATRIXBASE_H
#define OPENKALMAN_TYPEDMATRIXBASE_H

namespace OpenKalman::internal
{
  /**
   * \internal
   * \brief Base class for means or matrices.
   * \tparam Derived The derived class (e.g., Matrix, Mean, EuclideanMean).
   * \tparam RowCoefficients The \ref coefficients representing the rows of the matrix.
   * \tparam ColumnCoefficients The \ref coefficients representing the columns of the matrix.
   * \tparam NestedMatrix The nested matrix.
   */
#ifdef __cpp_concepts
  template<typename Derived, coefficients RowCoefficients, coefficients ColumnCoefficients,
    typed_matrix_nestable NestedMatrix> requires (not std::is_rvalue_reference_v<NestedMatrix>)
#else
  template<typename Derived, typename RowCoefficients, typename ColumnCoefficients, typename NestedMatrix>
#endif
  struct TypedMatrixBase : internal::MatrixBase<Derived, NestedMatrix>
  {

#ifndef __cpp_concepts
    static_assert(coefficients<RowCoefficients>);
    static_assert(coefficients<ColumnCoefficients>);
    static_assert(typed_matrix_nestable<NestedMatrix>);
    static_assert(not std::is_rvalue_reference_v<NestedMatrix>);
#endif

  private:

    using Base = internal::MatrixBase<Derived, NestedMatrix>;

  protected:

    using Scalar = typename MatrixTraits<NestedMatrix>::Scalar; ///< Scalar type for this variable.
    static constexpr auto dimension = MatrixTraits<NestedMatrix>::dimension; ///< Dimension of the vector.
    static constexpr auto columns = MatrixTraits<NestedMatrix>::columns; ///< Number of columns.

  public:

    /***************
     * Constructors
     ***************/

    /// Default constructor.
#ifdef __cpp_concepts
    TypedMatrixBase() requires std::default_initializable<Base> : Base {} {}
#else
    template<typename T = Base, std::enable_if_t<std::is_default_constructible_v<T>, int> = 0>
    TypedMatrixBase() : Base {} {}
#endif


    /// Copy constructor.
    TypedMatrixBase(const TypedMatrixBase& other) : Base {other.nested_matrix()} {}


    /// Move constructor.
    TypedMatrixBase(TypedMatrixBase&& other) noexcept : Base {std::move(other).nested_matrix()} {}


    /// Construct from a typed_matrix_nestable.
#ifdef __cpp_concepts
    template<typed_matrix_nestable Arg> requires (MatrixTraits<Arg>::dimension == dimension) and
      (MatrixTraits<Arg>::columns == columns) and std::is_constructible_v<Base, Arg>
#else
    template<typename Arg, std::enable_if_t<typed_matrix_nestable<Arg> and
      (MatrixTraits<Arg>::dimension == dimension) and (MatrixTraits<Arg>::columns == columns) and
      std::is_constructible_v<Base, Arg>, int> = 0>
#endif
    TypedMatrixBase(Arg&& arg) noexcept : Base {std::forward<Arg>(arg)} {}


    /// Construct from a list of coefficients.
#ifdef __cpp_concepts
    template<std::convertible_to<Scalar> ... Args> requires (sizeof...(Args) > 0) and
      requires {std::is_constructible_v<Base,
        decltype(MatrixTraits<NestedMatrix>::make(static_cast<const Scalar>(std::declval<const Args>())...))>; }
    TypedMatrixBase(const Args ... args)
      : Base {MatrixTraits<NestedMatrix>::make(static_cast<const Scalar>(args)...)} {}
#else
    // Note: std::is_constructible_v cannot be used here with ::make.
    template<typename ... Args, std::enable_if_t<(std::is_convertible_v<Args, Scalar> and ...) and
      (sizeof...(Args) == dimension) and diagonal_matrix<NestedMatrix> and
      std::is_constructible_v<Base, NestedMatrix&&>, int> = 0>
    TypedMatrixBase(const Args ... args)
      : Base {MatrixTraits<NestedMatrix>::make(static_cast<const Scalar>(args)...)} {}

    template<typename ... Args, std::enable_if_t<(std::is_convertible_v<Args, Scalar> and ...) and
      (not one_by_one_matrix<NestedMatrix>) and (sizeof...(Args) == dimension * columns) and
      std::is_constructible_v<Base, NestedMatrix&&>, int> = 0>
    TypedMatrixBase(const Args ... args)
      : Base {MatrixTraits<NestedMatrix>::make(static_cast<const Scalar>(args)...)} {}
#endif


    /**********************
     * Assignment Operators
     **********************/

    /// Copy assignment operator.
    auto& operator=(const TypedMatrixBase& other)
    {
      Base::operator=(other);
      return *this;
    }


    /// Move assignment operator.
    auto& operator=(TypedMatrixBase&& other)
    {
      Base::operator=(std::move(other));
      return *this;
    }


    /// Assign from a \ref typed_matrix_nestable.
#ifdef __cpp_concepts
    template<typed_matrix_nestable Arg> requires modifiable<NestedMatrix, Arg>
#else
    template<typename Arg, std::enable_if_t<typed_matrix_nestable<Arg> and modifiable<NestedMatrix, Arg>, int> = 0>
#endif
    auto& operator=(Arg&& arg)
    {
      if constexpr (not zero_matrix<NestedMatrix> and not identity_matrix<NestedMatrix>)
      {
        Base::operator=(std::forward<Arg>(arg));
        return *this;
      }
    }


    /// Multiply by a scale factor.
#ifdef __cpp_concepts
    template<std::convertible_to<Scalar> S>
#else
    template<typename S, std::enable_if_t<std::is_convertible_v<S, Scalar>, int> = 0>
#endif
    auto& operator*=(const S s)
    {
      this->nested_matrix() *= s;
      return *this;
    }


    /// Divide by a scale factor.
#ifdef __cpp_concepts
    template<std::convertible_to<Scalar> S>
#else
    template<typename S, std::enable_if_t<std::is_convertible_v<S, Scalar>, int> = 0>
#endif
    auto& operator/=(const S s)
    {
      this->nested_matrix() /= s;
      return *this;
    }


    /// Subscript (two indices)
    auto operator()(std::size_t i, std::size_t j) &
    {
      return make_ElementSetter<not element_settable<Derived, 2>>(static_cast<Derived&>(*this), i, j);
    }


    /// \overload
    auto operator()(std::size_t i, std::size_t j) &&
    {
      return make_ElementSetter<true>(static_cast<Derived&&>(*this), i, j);
    }


    /// \overload
    auto operator()(std::size_t i, std::size_t j) const &
    {
      return make_ElementSetter<true>(static_cast<const Derived&>(*this), i, j);
    }


    /// \overload
    auto operator()(std::size_t i, std::size_t j) const &&
    {
      return make_ElementSetter<true>(static_cast<const Derived&&>(*this), i, j);
    }


    /// Subscript (one index)
    auto operator[](std::size_t i) &
    {
      return make_ElementSetter<not element_settable<Derived, 1>>(static_cast<Derived&>(*this), i);
    }


    /// \overload
    auto operator[](std::size_t i) &&
    {
      return make_ElementSetter<true>(static_cast<Derived&&>(*this), i);
    }


    /// \overload
    auto operator[](std::size_t i) const &
    {
      return make_ElementSetter<true>(static_cast<const Derived&>(*this), i);
    }


    /// \overload
    auto operator[](std::size_t i) const &&
    {
      return make_ElementSetter<true>(static_cast<const Derived&&>(*this), i);
    }


    /// \overload
    auto operator()(std::size_t i) { return operator[](i); }


    /// \overload
    auto operator()(std::size_t i) const { return operator[](i); }

  };


} // namespace OpenKalman


#endif //OPENKALMAN_TYPEDMATRIXBASE_H
