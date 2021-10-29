/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2018-2021 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef OPENKALMAN_MATRIX_HPP
#define OPENKALMAN_MATRIX_HPP

namespace OpenKalman
{
  namespace oin = OpenKalman::internal;

  // --------------------- //
  //        Matrix         //
  // --------------------- //

#ifdef __cpp_concepts
  template<coefficients RowCoefficients, coefficients ColumnCoefficients, typed_matrix_nestable NestedMatrix>
  requires (RowCoefficients::dimensions == MatrixTraits<NestedMatrix>::rows) and
    (ColumnCoefficients::dimensions == MatrixTraits<NestedMatrix>::columns) and
    (not std::is_rvalue_reference_v<NestedMatrix>) and
    (dynamic_coefficients<RowCoefficients> == dynamic_rows<NestedMatrix>) and
    (dynamic_coefficients<ColumnCoefficients> == dynamic_columns<NestedMatrix>)
#else
  template<typename RowCoefficients, typename ColumnCoefficients, typename NestedMatrix>
#endif
  struct Matrix : oin::TypedMatrixBase<Matrix<RowCoefficients, ColumnCoefficients, NestedMatrix>, NestedMatrix,
    RowCoefficients, ColumnCoefficients>
  {

#ifndef __cpp_concepts
    static_assert(coefficients<RowCoefficients>);
    static_assert(coefficients<ColumnCoefficients>);
    static_assert(typed_matrix_nestable<NestedMatrix>);
    static_assert(RowCoefficients::dimensions == MatrixTraits<NestedMatrix>::rows);
    static_assert(ColumnCoefficients::dimensions == MatrixTraits<NestedMatrix>::columns);
    static_assert(not std::is_rvalue_reference_v<NestedMatrix>);
    static_assert(dynamic_coefficients<RowCoefficients> == dynamic_rows<NestedMatrix>);
    static_assert(dynamic_coefficients<ColumnCoefficients> == dynamic_columns<NestedMatrix>);
#endif

    using Scalar = typename MatrixTraits<NestedMatrix>::Scalar; ///< Scalar type for this matrix.

  private:

    using Base = oin::TypedMatrixBase<Matrix, NestedMatrix, RowCoefficients, ColumnCoefficients>;

  public:

    using Base::Base;


    /// Construct from a compatible \ref OpenKalman::typed_matrix "typed_matrix".
#ifdef __cpp_concepts
    template<typed_matrix Arg> requires (not std::derived_from<std::decay_t<Arg>, Matrix>) and
      (not euclidean_transformed<Arg>) and
      equivalent_to<typename MatrixTraits<Arg>::RowCoefficients, RowCoefficients> and
      equivalent_to<typename MatrixTraits<Arg>::ColumnCoefficients, ColumnCoefficients> and
      //requires(Arg&& arg) { NestedMatrix {nested_matrix(std::forward<Arg>(arg))}; } // \todo 't work in GCC 10
      std::constructible_from<NestedMatrix, decltype(nested_matrix(std::declval<Arg&&>()))>
#else
    template<typename Arg, std::enable_if_t<typed_matrix<Arg> and not std::is_base_of_v<Matrix, std::decay_t<Arg>> and
      not euclidean_transformed<Arg> and
      equivalent_to<typename MatrixTraits<Arg>::RowCoefficients, RowCoefficients> and
      equivalent_to<typename MatrixTraits<Arg>::ColumnCoefficients, ColumnCoefficients> and
      std::is_constructible_v<NestedMatrix, decltype(nested_matrix(std::declval<Arg&&>()))>, int> = 0>
#endif
    Matrix(Arg&& arg) noexcept : Base {nested_matrix(std::forward<Arg>(arg))} {}


    /// Construct from a compatible \ref OpenKalman::euclidean_transformed "euclidean_transformed".
#ifdef __cpp_concepts
    template<euclidean_transformed Arg> requires
      equivalent_to<typename MatrixTraits<Arg>::RowCoefficients, RowCoefficients> and
      equivalent_to<typename MatrixTraits<Arg>::ColumnCoefficients, ColumnCoefficients> and
      requires(Arg&& arg) { NestedMatrix {from_euclidean<RowCoefficients>(nested_matrix(std::forward<Arg>(arg)))}; }
#else
    template<typename Arg, std::enable_if_t<typed_matrix<Arg> and euclidean_transformed<Arg> and
      equivalent_to<typename MatrixTraits<Arg>::RowCoefficients, RowCoefficients> and
      equivalent_to<typename MatrixTraits<Arg>::ColumnCoefficients, ColumnCoefficients> and
      std::is_constructible_v<NestedMatrix,
        decltype(from_euclidean<RowCoefficients>(nested_matrix(std::declval<Arg&&>())))>, int> = 0>
#endif
    Matrix(Arg&& arg) noexcept
      : Base {from_euclidean<RowCoefficients>(nested_matrix(std::forward<Arg>(arg)))} {}


    /// Construct from compatible \ref OpenKalman::typed_matrix_nestable "typed_matrix_nestable".
#ifdef __cpp_concepts
    template<typed_matrix_nestable Arg> requires (MatrixTraits<Arg>::rows == MatrixTraits<NestedMatrix>::rows) and
      (MatrixTraits<Arg>::columns == MatrixTraits<NestedMatrix>::columns) and
      std::constructible_from<NestedMatrix, Arg&&>
#else
    template<typename Arg, std::enable_if_t<typed_matrix_nestable<Arg> and
      (MatrixTraits<Arg>::rows == MatrixTraits<NestedMatrix>::rows) and
      (MatrixTraits<Arg>::columns == MatrixTraits<NestedMatrix>::columns) and
      std::is_constructible_v<NestedMatrix, Arg&&>, int> = 0>
#endif
    explicit Matrix(Arg&& arg) noexcept : Base {std::forward<Arg>(arg)} {}


    /// Construct from compatible \ref OpenKalman::covariance "covariance".
#ifdef __cpp_concepts
    template<covariance Arg> requires
      equivalent_to<typename MatrixTraits<Arg>::RowCoefficients, RowCoefficients> and
      equivalent_to<typename MatrixTraits<Arg>::RowCoefficients, ColumnCoefficients> and
      requires(Arg&& arg) { NestedMatrix {make_native_matrix(std::forward<Arg>(arg))}; }
#else
    template<typename Arg, std::enable_if_t<covariance<Arg> and
      equivalent_to<typename MatrixTraits<Arg>::RowCoefficients, RowCoefficients> and
      equivalent_to<typename MatrixTraits<Arg>::RowCoefficients, ColumnCoefficients> and
      std::is_constructible_v<NestedMatrix, native_matrix_t<Arg>>, int> = 0>
#endif
    Matrix(Arg&& arg) noexcept : Base {make_native_matrix(std::forward<Arg>(arg))} {}


    /// Assign from a compatible \ref OpenKalman::typed_matrix "typed_matrix".
#ifdef __cpp_concepts
    template<typed_matrix Arg> requires (not euclidean_transformed<Arg>) and
      (not std::derived_from<std::decay_t<Arg>, Matrix>) and
      equivalent_to<typename MatrixTraits<Arg>::RowCoefficients, RowCoefficients> and
      equivalent_to<typename MatrixTraits<Arg>::ColumnCoefficients, ColumnCoefficients> and
      modifiable<NestedMatrix, nested_matrix_t<Arg>>
#else
    template<typename Arg, std::enable_if_t<typed_matrix<Arg> and (not euclidean_transformed<Arg>) and
      (not std::is_base_of_v<Matrix, std::decay_t<Arg>>) and
      equivalent_to<typename MatrixTraits<Arg>::RowCoefficients, RowCoefficients> and
      equivalent_to<typename MatrixTraits<Arg>::ColumnCoefficients, ColumnCoefficients> and
      modifiable<NestedMatrix, nested_matrix_t<Arg>>, int> = 0>
#endif
    auto& operator=(Arg&& other) noexcept
    {
      if constexpr (not zero_matrix<NestedMatrix> and not identity_matrix<NestedMatrix>)
      {
        Base::operator=(nested_matrix(std::forward<Arg>(other)));
      }
      return *this;
    }


    /// Assign from a compatible \ref OpenKalman::euclidean_transformed "euclidean_transformed" matrix.
#ifdef __cpp_concepts
    template<euclidean_transformed Arg> requires
      equivalent_to<typename MatrixTraits<Arg>::RowCoefficients, RowCoefficients> and
      equivalent_to<typename MatrixTraits<Arg>::ColumnCoefficients, ColumnCoefficients> and
      modifiable<NestedMatrix, decltype(from_euclidean<RowCoefficients>(std::declval<nested_matrix_t<Arg>>()))>
#else
    template<typename Arg, std::enable_if_t<euclidean_transformed<Arg> and
      equivalent_to<typename MatrixTraits<Arg>::RowCoefficients, RowCoefficients> and
      equivalent_to<typename MatrixTraits<Arg>::ColumnCoefficients, ColumnCoefficients> and
      modifiable<NestedMatrix, decltype(from_euclidean<RowCoefficients>(std::declval<nested_matrix_t<Arg>>()))>,
        int> = 0>
#endif
    auto& operator=(Arg&& other) noexcept
    {
      if constexpr (not zero_matrix<NestedMatrix> and not identity_matrix<NestedMatrix>)
      {
        Base::operator=(from_euclidean<RowCoefficients>(nested_matrix(std::forward<Arg>(other))));
      }
      return *this;
    }


    /// Assign from a compatible \ref OpenKalman::typed_matrix_nestable "typed_matrix_nestable".
#ifdef __cpp_concepts
    template<typed_matrix_nestable Arg> requires modifiable<NestedMatrix, Arg>
#else
    template<typename Arg, std::enable_if_t<typed_matrix_nestable<Arg> and modifiable<NestedMatrix, Arg>, int> = 0>
#endif
    auto& operator=(Arg&& arg) noexcept
    {
      if constexpr (not zero_matrix<NestedMatrix> and not identity_matrix<NestedMatrix>)
      {
        Base::operator=(std::forward<Arg>(arg));
      }
      return *this;
    }


    /// Increment from another Matrix.
    auto& operator+=(const Matrix& other)
    {
      this->nested_matrix() += other.nested_matrix();
      return *this;
    }

    /// Increment from another typed matrix.
#ifdef __cpp_concepts
    template<typed_matrix Arg> requires
      equivalent_to<typename MatrixTraits<Arg>::RowCoefficients, RowCoefficients> and
      equivalent_to<typename MatrixTraits<Arg>::ColumnCoefficients, ColumnCoefficients>
#else
    template<typename Arg, std::enable_if_t<typed_matrix<Arg> and
      equivalent_to<typename MatrixTraits<Arg>::RowCoefficients, RowCoefficients> and
      equivalent_to<typename MatrixTraits<Arg>::ColumnCoefficients, ColumnCoefficients>, int> = 0>
#endif
    auto& operator+=(Arg&& other) noexcept
    {
      this->nested_matrix() += nested_matrix(std::forward<Arg>(other));
      return *this;
    }


    /// Add a stochastic value to each column of the matrix, based on a distribution.
#ifdef __cpp_concepts
    template<distribution Arg> requires (ColumnCoefficients::axes_only) and
      (equivalent_to<typename DistributionTraits<Arg>::Coefficients, RowCoefficients>)
#else
    template<typename Arg, std::enable_if_t<distribution<Arg> and (ColumnCoefficients::axes_only) and
      (equivalent_to<typename DistributionTraits<Arg>::Coefficients, RowCoefficients>), int> = 0>
#endif
    auto& operator+=(const Arg& arg) noexcept
    {
      apply_columnwise(this->nested_matrix(), [&arg](auto& col) { col += arg().nested_matrix(); });
      return *this;
    }


    /// Decrement from another Matrix.
    auto& operator-=(const Matrix& other)
    {
      this->nested_matrix() -= other.nested_matrix();
      return *this;
    }


    /// Decrement from another typed matrix.
#ifdef __cpp_concepts
    template<typed_matrix Arg> requires
      equivalent_to<typename MatrixTraits<Arg>::RowCoefficients, RowCoefficients> and
      equivalent_to<typename MatrixTraits<Arg>::ColumnCoefficients, ColumnCoefficients>
#else
    template<typename Arg, std::enable_if_t<typed_matrix<Arg> and
      equivalent_to<typename MatrixTraits<Arg>::RowCoefficients, RowCoefficients> and
      equivalent_to<typename MatrixTraits<Arg>::ColumnCoefficients, ColumnCoefficients>, int> = 0>
#endif
    auto& operator-=(Arg&& other) noexcept
    {
      this->nested_matrix() -= nested_matrix(std::forward<Arg>(other));
      return *this;
    }


    /// Subtract a stochastic value to each column of the matrix, based on a distribution.
#ifdef __cpp_concepts
    template<distribution Arg> requires (ColumnCoefficients::axes_only) and
      (equivalent_to<typename DistributionTraits<Arg>::Coefficients, RowCoefficients>)
#else
    template<typename Arg, std::enable_if_t<distribution<Arg> and (ColumnCoefficients::axes_only) and
      (equivalent_to<typename DistributionTraits<Arg>::Coefficients, RowCoefficients>), int> = 0>
#endif
    auto& operator-=(const Arg& arg) noexcept
    {
      apply_columnwise(this->nested_matrix(), [&arg](auto& col){ col -= arg().nested_matrix(); });
      return *this;
    }

  private:

    template<typename CR = RowCoefficients, typename CC = ColumnCoefficients, typename Arg>
    static auto make(Arg&& arg) noexcept
    {
      return Matrix<CR, CC, std::decay_t<Arg>>(std::forward<Arg>(arg));
    }

  };


  // ------------------------------- //
  //        Deduction Guides         //
  // ------------------------------- //

  /// Deduce parameter types from a typed_matrix_nestable.
#ifdef __cpp_concepts
  template<typed_matrix_nestable M>
#else
  template<typename M, std::enable_if_t<typed_matrix_nestable<M>, int> = 0>
#endif
  explicit Matrix(M&&) -> Matrix<Axes<MatrixTraits<M>::rows>, Axes<MatrixTraits<M>::columns>, passable_t<M>>;


  /// Deduce template parameters from a non-Euclidean-transformed typed matrix.
#ifdef __cpp_concepts
  template<typed_matrix V> requires (not euclidean_transformed<V>)
#else
  template<typename V, std::enable_if_t<typed_matrix<V> and not euclidean_transformed<V>, int> = 0>
#endif
  Matrix(V&&) -> Matrix<
    typename MatrixTraits<V>::RowCoefficients,
    typename MatrixTraits<V>::ColumnCoefficients,
    passable_t<nested_matrix_t<V>>>;


  /// Deduce template parameters from a Euclidean-transformed typed matrix.
#if defined(__cpp_concepts) and false
  // \todo Unlike SFINAE version, this incorrectly matches V==Mean and V==Matrix in both GCC 10.1.0 and clang 10.0.0:
  template<euclidean_transformed V> requires untyped_columns<V>
#else
  template<typename V, std::enable_if_t<euclidean_transformed<V> and untyped_columns<V>, int> = 0>
#endif
  Matrix(V&&) -> Matrix<
    typename MatrixTraits<V>::RowCoefficients,
    typename MatrixTraits<V>::ColumnCoefficients,
    decltype(from_euclidean<typename MatrixTraits<V>::RowCoefficients>(
      nested_matrix(std::forward<V>(std::declval<V>()))))>;


  /// Deduce parameter types from a Covariance.
#ifdef __cpp_concepts
  template<covariance V>
#else
  template<typename V, std::enable_if_t<covariance<V>, int> = 0>
#endif
  Matrix(V&&) -> Matrix<
    typename MatrixTraits<V>::RowCoefficients,
    typename MatrixTraits<V>::RowCoefficients,
    native_matrix_t<nested_matrix_t<V>>>;


  // ----------------------------- //
  //        Make functions         //
  // ----------------------------- //

  /**
   * \brief Make a Matrix object from a typed_matrix_nestable, specifying the row and column coefficients.
   * \tparam RowCoefficients The coefficient types corresponding to the rows.
   * \tparam ColumnCoefficients The coefficient types corresponding to the columns.
   * \tparam M A typed_matrix_nestable with size matching RowCoefficients and ColumnCoefficients.
   */
#ifdef __cpp_concepts
  template<coefficients RowCoefficients, coefficients ColumnCoefficients, typed_matrix_nestable M>
    requires (MatrixTraits<M>::rows == RowCoefficients::dimensions) and
    (MatrixTraits<M>::columns == ColumnCoefficients::dimensions)
#else
  template<typename RowCoefficients, typename ColumnCoefficients, typename M, std::enable_if_t<
    coefficients<RowCoefficients> and coefficients<ColumnCoefficients> and typed_matrix_nestable<M> and
    (MatrixTraits<M>::rows == RowCoefficients::dimensions) and
    (MatrixTraits<M>::columns == ColumnCoefficients::dimensions), int> = 0>
#endif
  inline auto make_matrix(M&& m)
  {
    return Matrix<RowCoefficients, ColumnCoefficients, passable_t<M>>(std::forward<M>(m));
  }


  /**
   * \brief Make a Matrix object from a typed_matrix_nestable, specifying only the row coefficients.
   * \details The column coefficients are default Axis.
   * \tparam RowCoefficients The coefficient types corresponding to the rows.
   * \tparam M A typed_matrix_nestable with size matching RowCoefficients and ColumnCoefficients.
   */
#ifdef __cpp_concepts
  template<coefficients RowCoefficients, typed_matrix_nestable M>
  requires (MatrixTraits<M>::rows == RowCoefficients::dimensions)
#else
  template<typename RowCoefficients, typename M, std::enable_if_t<
    coefficients<RowCoefficients> and typed_matrix_nestable<M> and
    (MatrixTraits<M>::rows == RowCoefficients::dimensions), int> = 0>
#endif
  inline auto make_matrix(M&& m)
  {
    using ColumnCoefficients = Axes<MatrixTraits<M>::columns>;
    return Matrix<RowCoefficients, ColumnCoefficients, passable_t<M>>(std::forward<M>(m));
  }


  /**
   * \overload
   * \brief Make a Matrix object from a typed_matrix_nestable object, with default Axis coefficients.
   */
#ifdef __cpp_concepts
  template<typed_matrix_nestable M>
#else
  template<typename M, std::enable_if_t<typed_matrix_nestable<M>, int> = 0>
#endif
  inline auto make_matrix(M&& m)
  {
    using RowCoeffs = Axes<MatrixTraits<M>::rows>;
    using ColCoeffs = Axes<MatrixTraits<M>::columns>;
    return make_matrix<RowCoeffs, ColCoeffs>(std::forward<M>(m));
  }


  /**
   * \overload
   * \brief Make a Matrix object from a covariance object.
   * \tparam M A covariance object (i.e., Covariance, SquareRootCovariance).
   */
#ifdef __cpp_concepts
  template<covariance M>
#else
  template<typename M, std::enable_if_t<covariance<M>, int> = 0>
#endif
  inline auto make_matrix(M&& arg)
  {
    using C = typename MatrixTraits<M>::RowCoefficients;
    return make_matrix<C, C>(make_native_matrix(std::forward<M>(arg)));
  }


  /**
   * \overload
   * \brief Make a Matrix object from another typed_matrix.
   * \tparam Arg A typed_matrix (i.e., Matrix, Mean, or EuclideanMean).
   */
#ifdef __cpp_concepts
  template<typed_matrix Arg>
#else
  template<typename Arg, std::enable_if_t<typed_matrix<Arg>, int> = 0>
#endif
  inline auto make_matrix(Arg&& arg)
  {
    using RowCoeffs = typename MatrixTraits<Arg>::RowCoefficients;
    using ColCoeffs = typename MatrixTraits<Arg>::ColumnCoefficients;
    if constexpr(euclidean_transformed<Arg>)
      return make_matrix<RowCoeffs, ColCoeffs>(nested_matrix(from_euclidean<RowCoeffs>(std::forward<Arg>(arg))));
    else
      return make_matrix<RowCoeffs, ColCoeffs>(nested_matrix(std::forward<Arg>(arg)));
  }


  /**
   * \overload
   * \brief Make a default, self-contained Matrix object.
   * \tparam RowCoefficients The coefficient types corresponding to the rows.
   * \tparam ColumnCoefficients The coefficient types corresponding to the columns.
   * \tparam M a typed_matrix_nestable on which the new matrix is based. It will be converted to a self_contained type
   * if it is not already self-contained.
   */
#ifdef __cpp_concepts
  template<coefficients RowCoefficients, coefficients ColumnCoefficients, typed_matrix_nestable M> requires
    (MatrixTraits<M>::rows == RowCoefficients::dimensions) and
    (MatrixTraits<M>::columns == ColumnCoefficients::dimensions)
#else
  template<typename RowCoefficients, typename ColumnCoefficients, typename M, std::enable_if_t<
    coefficients<RowCoefficients> and coefficients<ColumnCoefficients> and typed_matrix_nestable<M> and
    (MatrixTraits<M>::rows == RowCoefficients::dimensions) and
    (MatrixTraits<M>::columns == ColumnCoefficients::dimensions), int> = 0>
#endif
  inline auto make_matrix()
  {
    return Matrix<RowCoefficients, ColumnCoefficients, native_matrix_t<M>>();
  }


  /**
   * \overload
   * \brief Make a self-contained Matrix object with default Axis coefficients.
   * \tparam M a typed_matrix_nestable on which the new matrix is based. It will be converted to a self_contained type
   * if it is not already self-contained.
   */
#ifdef __cpp_concepts
  template<typed_matrix_nestable M>
#else
  template<typename M, std::enable_if_t<typed_matrix_nestable<M>, int> = 0>
#endif
  inline auto make_matrix()
  {
    using RowCoeffs = Axes<MatrixTraits<M>::rows>;
    using ColCoeffs = Axes<MatrixTraits<M>::columns>;
    return make_matrix<RowCoeffs, ColCoeffs, M>();
  }


  // -------------- //
  //  MatrixTraits  //
  // -------------- //

  template<typename RowCoeffs, typename ColCoeffs, typename NestedType>
  struct MatrixTraits<Matrix<RowCoeffs, ColCoeffs, NestedType>>
  {
    using NestedMatrix = NestedType;
    using Coefficients = RowCoeffs;
    using RowCoefficients = RowCoeffs;
    using ColumnCoefficients = ColCoeffs;
    static constexpr std::size_t rows = MatrixTraits<NestedMatrix>::rows;
    static constexpr std::size_t columns = MatrixTraits<NestedMatrix>::columns;
    using Scalar = typename MatrixTraits<NestedMatrix>::Scalar;
    static_assert(RowCoefficients::dimensions == rows);
    static_assert(ColumnCoefficients::dimensions == columns);

    template<std::size_t r = rows, std::size_t c = columns, typename S = Scalar>
    using NativeMatrixFrom = native_matrix_t<NestedMatrix, r, c, S>;

    using SelfContainedFrom = Matrix<RowCoefficients, ColumnCoefficients, self_contained_t<NestedMatrix>>;


#ifdef __cpp_concepts
    template<coefficients RC = RowCoefficients, coefficients CC = ColumnCoefficients, typed_matrix_nestable Arg>
    requires (MatrixTraits<Arg>::rows == RC::dimensions) and (MatrixTraits<Arg>::columns == CC::dimensions)
#else
    template<typename RC = RowCoefficients, typename CC = ColumnCoefficients, typename Arg, std::enable_if_t<
      coefficients<RC> and coefficients<CC> and typed_matrix_nestable<Arg> and
      (MatrixTraits<Arg>::rows == RC::dimensions) and (MatrixTraits<Arg>::columns == CC::dimensions), int> = 0>
#endif
    static auto make(Arg&& arg)
    {
      return Matrix<RC, CC, std::decay_t<Arg>>(std::forward<Arg>(arg));
    }


#ifdef __cpp_concepts
    template<std::convertible_to<std::size_t> ... Args> requires
    (sizeof...(Args) == (rows == 0 ? 1 : 0) + (columns == 0 ? 1 : 0))
#else
    template<typename...Args, std::enable_if_t<(std::is_convertible_v<Args, std::size_t> and ...) and
      (sizeof...(Args) == (rows == 0 ? 1 : 0) + (columns == 0 ? 1 : 0)), int> = 0>
#endif
    static auto zero(const Args...args)
    {
      return make(MatrixTraits<NestedMatrix>::zero(static_cast<std::size_t>(args)...));
    }


#ifdef __cpp_concepts
    template<std::convertible_to<std::size_t> ... Args> requires (sizeof...(Args) == (rows == 0 ? 1 : 0))
#else
    template<typename...Args, std::enable_if_t<(std::is_convertible_v<Args, std::size_t> and ...) and
      (sizeof...(Args) == (rows == 0 ? 1 : 0)), int> = 0>
#endif
    static auto identity(const Args...args)
    {
      auto b = MatrixTraits<NestedMatrix>::identity(args...);
      return make<RowCoefficients, RowCoefficients>(std::move(b));
    }

  };

} // namespace OpenKalman


#endif //OPENKALMAN_MATRIX_HPP
