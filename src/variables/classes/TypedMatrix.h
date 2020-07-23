/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2018-2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef OPENKALMAN_TYPEDMATRIX_H
#define OPENKALMAN_TYPEDMATRIX_H

namespace OpenKalman
{
  ////////////////////////////////
  //        TypedMatrix         //
  ////////////////////////////////

  /// A typed matrix.
  template<
    typename RowCoefficients,
    typename ColumnCoefficients,
    typename BaseMatrix>
  struct TypedMatrix : internal::TypedMatrixBase<TypedMatrix<RowCoefficients, ColumnCoefficients, BaseMatrix>,
    RowCoefficients, ColumnCoefficients, BaseMatrix>
  {
    using Base = internal::TypedMatrixBase<TypedMatrix, RowCoefficients, ColumnCoefficients, BaseMatrix>;
    static_assert(is_typed_matrix_base_v<BaseMatrix>);
    static_assert(Base::dimension == RowCoefficients::size);
    static_assert(Base::columns == ColumnCoefficients::size);

    using Base::base_matrix;
    using Base::Base;

    /// Copy constructor.
    TypedMatrix(const TypedMatrix& other) : Base(other.base_matrix()) {}

    /// Move constructor.
    TypedMatrix(TypedMatrix&& other) noexcept : Base(std::move(other).base_matrix()) {}

    /// Construct from a compatible matrix.
    template<typename Arg, std::enable_if_t<is_typed_matrix_v<Arg> and not is_Euclidean_transformed_v<Arg>, int> = 0>
    TypedMatrix(Arg&& other) noexcept : Base(std::forward<Arg>(other).base_matrix())
    {
      static_assert(is_equivalent_v<typename MatrixTraits<Arg>::RowCoefficients, RowCoefficients>);
      static_assert(is_equivalent_v<typename MatrixTraits<Arg>::ColumnCoefficients, ColumnCoefficients>);
    }

    /// Construct from a compatible Euclidean-transformed matrix.
    template<typename Arg, std::enable_if_t<is_typed_matrix_v<Arg> and is_Euclidean_transformed_v<Arg>, int> = 0>
    TypedMatrix(Arg&& other) noexcept : Base(OpenKalman::from_Euclidean<RowCoefficients>(std::forward<Arg>(other).base_matrix()))
    {
      static_assert(is_equivalent_v<typename MatrixTraits<Arg>::RowCoefficients, RowCoefficients>);
      static_assert(is_equivalent_v<typename MatrixTraits<Arg>::ColumnCoefficients, ColumnCoefficients>);
    }

    /// Construct from compatible typed matrix base.
    template<typename Arg, std::enable_if_t<is_typed_matrix_base_v<Arg>, int> = 0>
    TypedMatrix(Arg&& arg) noexcept : Base(std::forward<Arg>(arg))
    {
      static_assert(MatrixTraits<Arg>::dimension == Base::dimension);
      static_assert(MatrixTraits<Arg>::columns == Base::columns);
    }

    /// Construct from compatible covariance.
    template<typename Arg, std::enable_if_t<is_covariance_v<Arg>, int> = 0>
    TypedMatrix(Arg&& arg) noexcept : Base(strict_matrix(std::forward<Arg>(arg)))
    {
      static_assert(is_equivalent_v<typename MatrixTraits<Arg>::Coefficients, RowCoefficients>);
      static_assert(is_equivalent_v<RowCoefficients, ColumnCoefficients>);
    }

    /// Copy assignment operator.
    TypedMatrix& operator=(const TypedMatrix& other)
    {
      this->base_matrix() = other.base_matrix();
      return *this;
    }

    /// Move assignment operator.
    TypedMatrix& operator=(TypedMatrix&& other)
    {
      if (this != &other) this->base_matrix() = std::move(other).base_matrix();
      return *this;
    }

    /// Assign from a compatible typed matrix.
    template<typename Arg, std::enable_if_t<is_typed_matrix_v<Arg>, int> = 0>
    auto& operator=(Arg&& other) noexcept
    {
      static_assert(is_equivalent_v<typename MatrixTraits<Arg>::RowCoefficients, RowCoefficients>);
      static_assert(is_equivalent_v<typename MatrixTraits<Arg>::ColumnCoefficients, ColumnCoefficients>);
      if constexpr (std::is_same_v<std::decay_t<Arg>, TypedMatrix>) if (this == &other) return *this;
      if constexpr(is_Euclidean_transformed_v<Arg>)
        this->base_matrix() = from_Euclidean<RowCoefficients>(std::forward<Arg>(other).base_matrix());
      else
        this->base_matrix() = std::forward<Arg>(other).base_matrix();
      return *this;
    }

  protected:
    template<typename CR = RowCoefficients, typename CC = ColumnCoefficients, typename Arg>
    static auto
    make(Arg&& arg) noexcept { return TypedMatrix<CR, CC, std::decay_t<Arg>>(std::forward<Arg>(arg)); }

  public:
    static auto zero() { return make(MatrixTraits<BaseMatrix>::zero()); }

    static auto identity() { return make(MatrixTraits<BaseMatrix>::identity()); }
  };


  /////////////////////////////////////
  //        Deduction Guides         //
  /////////////////////////////////////

  /// Deduce parameter types from a typed matrix base.
  template<typename M, std::enable_if_t<is_typed_matrix_base_v<M>, int> = 0>
  TypedMatrix(M&&)
  -> TypedMatrix<Axes<MatrixTraits<M>::dimension>, Axes<MatrixTraits<M>::columns>, std::decay_t<M>>;

  /// Deduce template parameters from a non-Euclidean-transformed typed matrix.
  template<typename V, std::enable_if_t<is_typed_matrix_v<V> and not is_Euclidean_transformed_v<V>, int> = 0>
  TypedMatrix(V&&) -> TypedMatrix<
    typename MatrixTraits<V>::RowCoefficients,
    typename MatrixTraits<V>::ColumnCoefficients,
    typename MatrixTraits<V>::BaseMatrix>;

  /// Deduce template parameters from a Euclidean-transformed typed matrix.
  template<typename V, std::enable_if_t<is_typed_matrix_v<V> and is_Euclidean_transformed_v<V> and
  MatrixTraits<V>::ColumnCoefficients::axes_only, int> = 0>
  TypedMatrix(V&&) -> TypedMatrix<
    typename MatrixTraits<V>::RowCoefficients,
    typename MatrixTraits<V>::ColumnCoefficients,
    decltype(from_Euclidean<typename MatrixTraits<V>::RowCoefficients>(std::forward<V>(std::declval<V>()).base_matrix()))>;

  /// Deduce parameter types from a Covariance.
  template<typename V, std::enable_if_t<is_covariance_v<V>, int> = 0>
  TypedMatrix(V&&) -> TypedMatrix<
    typename MatrixTraits<V>::Coefficients,
    typename MatrixTraits<V>::Coefficients,
    typename MatrixTraits<V>::template StrictMatrix<>>;


  ///////////////////////////////////
  //        Make functions         //
  ///////////////////////////////////

  /// Make a TypedMatrix object from a typed matrix base.
  template<
    typename RowCoefficients = void, typename ColumnCoefficients = void, typename M,
      std::enable_if_t<is_typed_matrix_base_v<M>, int> = 0>
  inline auto make_Matrix(M&& arg)
  {
    using RowCoeffs = std::conditional_t<std::is_void_v<RowCoefficients>,
      Axes<MatrixTraits<M>::dimension>,
      RowCoefficients>;
    using ColCoeffs = std::conditional_t<std::is_void_v<ColumnCoefficients>,
      Axes<MatrixTraits<M>::columns>,
      ColumnCoefficients>;
    static_assert(MatrixTraits<M>::dimension == RowCoeffs::size);
    static_assert(MatrixTraits<M>::columns == ColCoeffs::size);
    return TypedMatrix<RowCoeffs, ColCoeffs, std::decay_t<M>>(std::forward<M>(arg));
  }


  /// Make a TypedMatrix object from a covariance object.
  template<typename M, std::enable_if_t<is_covariance_v<M>, int> = 0>
  inline auto make_Matrix(M&& arg)
  {
    using C = typename MatrixTraits<M>::Coefficients;
    return make_Matrix<C, C>(strict_matrix(std::forward<M>(arg)));
  }


  /// Make a TypedMatrix object from another typed matrix.
  template<typename Arg, std::enable_if_t<is_typed_matrix_v<Arg>, int> = 0>
  inline auto make_Matrix(Arg&& arg)
  {
    using RowCoeffs = typename MatrixTraits<Arg>::RowCoefficients;
    using ColCoeffs = typename MatrixTraits<Arg>::ColumnCoefficients;
    if constexpr(is_Euclidean_transformed_v<Arg>)
      return make_Matrix<RowCoeffs, ColCoeffs>(base_matrix(from_Euclidean<RowCoeffs>(std::forward<Arg>(arg))));
    else
      return make_Matrix<RowCoeffs, ColCoeffs>(base_matrix(std::forward<Arg>(arg)));
  }


  /// Make a default, strict TypedMatrix object.
  template<typename RowCoefficients, typename ColumnCoefficients, typename M,
    std::enable_if_t<is_typed_matrix_base_v<M>, int> = 0>
  inline auto make_Matrix()
  {
    static_assert(RowCoefficients::size == MatrixTraits<M>::dimension);
    return TypedMatrix<RowCoefficients, ColumnCoefficients, typename MatrixTraits<M>::template StrictMatrix<>>();
  }


  /// Make a default, strict TypedMatrix object with axis coefficients.
  template<typename M, std::enable_if_t<is_typed_matrix_base_v<M>, int> = 0>
  inline auto make_Matrix()
  {
    using RowCoeffs = Axes<MatrixTraits<M>::dimension>;
    using ColCoeffs = Axes<MatrixTraits<M>::columns>;
    return make_Matrix<RowCoeffs, ColCoeffs, M>();
  }


  ///////////////////////////
  //        Traits         //
  ///////////////////////////

  /// MatrixTraits for TypedMatrix
  template<typename RowCoeffs, typename ColCoeffs, typename NestedType>
  struct MatrixTraits<OpenKalman::TypedMatrix<RowCoeffs, ColCoeffs, NestedType>>
  {
    using BaseMatrix = NestedType;
    using Coefficients = RowCoeffs;
    using RowCoefficients = RowCoeffs;
    using ColumnCoefficients = ColCoeffs;
    static constexpr std::size_t dimension = MatrixTraits<BaseMatrix>::dimension;
    static constexpr std::size_t columns = MatrixTraits<BaseMatrix>::columns;
    using Scalar = typename MatrixTraits<BaseMatrix>::Scalar;
    static_assert(RowCoefficients::size == dimension);
    static_assert(ColumnCoefficients::size == columns);

    template<std::size_t rows = dimension, std::size_t cols = columns, typename S = Scalar>
    using StrictMatrix = typename MatrixTraits<BaseMatrix>::template StrictMatrix<rows, cols, S>;

    template<
      typename RC = RowCoefficients,
      typename CC = ColumnCoefficients,
      typename Arg, std::enable_if_t<is_typed_matrix_base_v<Arg>, int> = 0>
    static auto make(Arg&& arg)
    {
      static_assert(MatrixTraits<Arg>::dimension == RC::size);
      if constexpr(MatrixTraits<Arg>::columns == CC::size)
      {
        return TypedMatrix<RC, CC, std::decay_t<Arg>>(std::forward<Arg>(arg));
      }
      else
      {
        return Mean<RC, std::decay_t<Arg>>(std::forward<Arg>(arg));
      }
    }

    static auto zero() { return make(MatrixTraits<BaseMatrix>::zero()); }

    static auto identity() { return make<RowCoefficients, RowCoefficients>(MatrixTraits<BaseMatrix>::identity()); }

  };

} // namespace OpenKalman


#endif //OPENKALMAN_TYPEDMATRIX_H
