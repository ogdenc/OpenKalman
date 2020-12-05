/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2018-2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef OPENKALMAN_MATRIX_HPP
#define OPENKALMAN_MATRIX_HPP

namespace OpenKalman
{
  ///////////////////////////
  //        Matrix         //
  ///////////////////////////

#ifdef __cpp_concepts
  template<coefficients RowCoefficients, coefficients ColumnCoefficients, typed_matrix_nestable NestedMatrix> requires
    (RowCoefficients::size == MatrixTraits<NestedMatrix>::dimension) and
    (ColumnCoefficients::size == MatrixTraits<NestedMatrix>::columns) and (not std::is_rvalue_reference_v<NestedMatrix>)
#else
  template<typename RowCoefficients, typename ColumnCoefficients, typename NestedMatrix>
#endif
  struct Matrix : internal::TypedMatrixBase<Matrix<RowCoefficients, ColumnCoefficients, NestedMatrix>,
    RowCoefficients, ColumnCoefficients, NestedMatrix>
  {
    using Base = internal::TypedMatrixBase<Matrix, RowCoefficients, ColumnCoefficients, NestedMatrix>;
    static_assert(typed_matrix_nestable<NestedMatrix>);
    static_assert(Base::dimension == RowCoefficients::size);
    static_assert(Base::columns == ColumnCoefficients::size);
    static_assert(not std::is_rvalue_reference_v<NestedMatrix>);

    using Base::Base;

    /// Copy constructor.
    Matrix(const Matrix& other) : Base(other.nested_matrix()) {}

    /// Move constructor.
    Matrix(Matrix&& other) noexcept : Base(std::move(other).nested_matrix()) {}

    /// Construct from a compatible matrix.
#ifdef __cpp_concepts
    template<typed_matrix Arg> requires (not euclidean_transformed<Arg>)
#else
    template<typename Arg, std::enable_if_t<typed_matrix<Arg> and not euclidean_transformed<Arg>, int> = 0>
#endif
    Matrix(Arg&& other) noexcept : Base(std::forward<Arg>(other).nested_matrix())
    {
      static_assert(equivalent_to<typename MatrixTraits<Arg>::RowCoefficients, RowCoefficients>);
      static_assert(equivalent_to<typename MatrixTraits<Arg>::ColumnCoefficients, ColumnCoefficients>);
    }

    /// Construct from a compatible Euclidean-transformed matrix.
#ifdef __cpp_concepts
    template<euclidean_transformed Arg>
#else
    template<typename Arg, std::enable_if_t<typed_matrix<Arg> and euclidean_transformed<Arg>, int> = 0>
#endif
    Matrix(Arg&& other) noexcept : Base(OpenKalman::from_euclidean<RowCoefficients>(std::forward<Arg>(other).nested_matrix()))
    {
      static_assert(equivalent_to<typename MatrixTraits<Arg>::RowCoefficients, RowCoefficients>);
      static_assert(equivalent_to<typename MatrixTraits<Arg>::ColumnCoefficients, ColumnCoefficients>);
    }

    /// Construct from compatible typed_matrix_nestable.
#ifdef __cpp_concepts
    template<typed_matrix_nestable Arg>
#else
    template<typename Arg, std::enable_if_t<typed_matrix_nestable<Arg>, int> = 0>
#endif
    Matrix(Arg&& arg) noexcept : Base(std::forward<Arg>(arg))
    {
      static_assert(MatrixTraits<Arg>::dimension == Base::dimension);
      static_assert(MatrixTraits<Arg>::columns == Base::columns);
    }

    /// Construct from compatible covariance.
#ifdef __cpp_concepts
    template<covariance Arg>
#else
    template<typename Arg, std::enable_if_t<covariance<Arg>, int> = 0>
#endif
    Matrix(Arg&& arg) noexcept : Base(make_native_matrix(std::forward<Arg>(arg)))
    {
      static_assert(equivalent_to<typename MatrixTraits<Arg>::RowCoefficients, RowCoefficients>);
      static_assert(equivalent_to<RowCoefficients, ColumnCoefficients>);
    }

    /// Copy assignment operator.
    auto& operator=(const Matrix& other)
    {
      if constexpr (not zero_matrix<NestedMatrix> and not identity_matrix<NestedMatrix>) if (this != &other)
        this->nested_matrix() = other.nested_matrix();
      return *this;
    }

    /// Move assignment operator.
    auto& operator=(Matrix&& other)
    {
      if constexpr (not zero_matrix<NestedMatrix> and not identity_matrix<NestedMatrix>) if (this != &other)
        this->nested_matrix() = std::move(other).nested_matrix();
      return *this;
    }

    /// Assign from a compatible typed matrix.
#ifdef __cpp_concepts
    template<typed_matrix Arg>
#else
    template<typename Arg, std::enable_if_t<typed_matrix<Arg>, int> = 0>
#endif
    auto& operator=(Arg&& other) noexcept
    {
      static_assert(equivalent_to<typename MatrixTraits<Arg>::RowCoefficients, RowCoefficients>);
      static_assert(equivalent_to<typename MatrixTraits<Arg>::ColumnCoefficients, ColumnCoefficients>);
      if constexpr (zero_matrix<NestedMatrix>)
      {
        static_assert(zero_matrix<Arg>);
      }
      else if constexpr (identity_matrix<NestedMatrix>)
      {
        static_assert(identity_matrix<Arg>);
      }
      else if constexpr(euclidean_transformed<Arg>)
      {
        this->nested_matrix() = from_euclidean<RowCoefficients>(std::forward<Arg>(other).nested_matrix());
      }
      else
      {
        this->nested_matrix() = std::forward<Arg>(other).nested_matrix();
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
    template<typed_matrix Arg>
#else
    template<typename Arg, std::enable_if_t<typed_matrix<Arg>, int> = 0>
#endif
    auto& operator+=(Arg&& other) noexcept
    {
      static_assert(equivalent_to<typename MatrixTraits<Arg>::RowCoefficients, RowCoefficients>);
      static_assert(equivalent_to<typename MatrixTraits<Arg>::ColumnCoefficients, ColumnCoefficients>);
      this->nested_matrix() += std::forward<Arg>(other).nested_matrix();
      return *this;
    }

    /// Add a stochastic value to each column of the matrix, based on a distribution.
#ifdef __cpp_concepts
    template<typename Arg> requires distribution<Arg>
#else
    template<typename Arg, std::enable_if_t<distribution<Arg>, int> = 0>
#endif
    auto& operator+=(const Arg& arg) noexcept
    {
      static_assert(equivalent_to<typename DistributionTraits<Arg>::Coefficients, RowCoefficients>);
      static_assert(ColumnCoefficients::axes_only);
      static_assert(not euclidean_transformed<Matrix>);
      apply_columnwise(this->nested_matrix(), [&arg](auto& col){ col += arg().nested_matrix(); });
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
    template<typed_matrix Arg>
#else
    template<typename Arg, std::enable_if_t<typed_matrix<Arg>, int> = 0>
#endif
    auto& operator-=(Arg&& other) noexcept
    {
      static_assert(equivalent_to<typename MatrixTraits<Arg>::RowCoefficients, RowCoefficients>);
      static_assert(equivalent_to<typename MatrixTraits<Arg>::ColumnCoefficients, ColumnCoefficients>);
      this->nested_matrix() -= std::forward<Arg>(other).nested_matrix();
      return *this;
    }

    /// Subtract a stochastic value to each column of the matrix, based on a distribution.
#ifdef __cpp_concepts
    template<typename Arg> requires distribution<Arg>
#else
    template<typename Arg, std::enable_if_t<distribution<Arg>, int> = 0>
#endif
    auto& operator-=(const Arg& arg) noexcept
    {
      static_assert(equivalent_to<typename DistributionTraits<Arg>::Coefficients, RowCoefficients>);
      static_assert(ColumnCoefficients::axes_only);
      static_assert(not euclidean_transformed<Matrix>);
      apply_columnwise(this->nested_matrix(), [&arg](auto& col){ col -= arg().nested_matrix(); });
      return *this;
    }

  private:
    template<typename CR = RowCoefficients, typename CC = ColumnCoefficients, typename Arg>
    static auto
    make(Arg&& arg) noexcept { return Matrix<CR, CC, self_contained_t<Arg>>(std::forward<Arg>(arg)); }

  public:
    static auto zero() { return make(MatrixTraits<NestedMatrix>::zero()); }

    static auto identity() { return make(MatrixTraits<NestedMatrix>::identity()); }
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
  Matrix(M&&)
  -> Matrix<Axes<MatrixTraits<M>::dimension>, Axes<MatrixTraits<M>::columns>, passable_t<M>>;


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
  // \TODO Unlike SFINAE version, this incorrectly matches V==Mean and V==Matrix in both GCC 10.1.0 and clang 10.0.0:
  template<euclidean_transformed V> requires MatrixTraits<V>::ColumnCoefficients::axes_only
#else
  template<typename V, std::enable_if_t<typed_matrix<V> and euclidean_transformed<V> and
    MatrixTraits<V>::ColumnCoefficients::axes_only, int> = 0>
#endif
  Matrix(V&&) -> Matrix<
    typename MatrixTraits<V>::RowCoefficients,
    typename MatrixTraits<V>::ColumnCoefficients,
    decltype(from_euclidean<typename MatrixTraits<V>::RowCoefficients>(std::forward<V>(std::declval<V>()).nested_matrix()))>;


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
    requires (MatrixTraits<M>::dimension == RowCoefficients::size) and
    (MatrixTraits<M>::columns == ColumnCoefficients::size)
#else
  template<typename RowCoefficients, typename ColumnCoefficients, typename M, std::enable_if_t<
    coefficients<RowCoefficients> and coefficients<ColumnCoefficients> and typed_matrix_nestable<M> and
    (MatrixTraits<M>::dimension == RowCoefficients::size) and
    (MatrixTraits<M>::columns == ColumnCoefficients::size), int> = 0>
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
  requires (MatrixTraits<M>::dimension == RowCoefficients::size)
#else
  template<typename RowCoefficients, typename M, std::enable_if_t<
    coefficients<RowCoefficients> and typed_matrix_nestable<M> and
    (MatrixTraits<M>::dimension == RowCoefficients::size), int> = 0>
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
    using RowCoeffs = Axes<MatrixTraits<M>::dimension>;
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
    (MatrixTraits<M>::dimension == RowCoefficients::size) and
    (MatrixTraits<M>::columns == ColumnCoefficients::size)
#else
  template<typename RowCoefficients, typename ColumnCoefficients, typename M, std::enable_if_t<
    coefficients<RowCoefficients> and coefficients<ColumnCoefficients> and typed_matrix_nestable<M> and
    (MatrixTraits<M>::dimension == RowCoefficients::size) and
    (MatrixTraits<M>::columns == ColumnCoefficients::size), int> = 0>
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
    using RowCoeffs = Axes<MatrixTraits<M>::dimension>;
    using ColCoeffs = Axes<MatrixTraits<M>::columns>;
    return make_matrix<RowCoeffs, ColCoeffs, M>();
  }


  // -------------- //
  //  MatrixTraits  //
  // -------------- //

  template<typename RowCoeffs, typename ColCoeffs, typename NestedType>
  struct MatrixTraits<OpenKalman::Matrix<RowCoeffs, ColCoeffs, NestedType>>
  {
    using NestedMatrix = NestedType;
    using Coefficients = RowCoeffs;
    using RowCoefficients = RowCoeffs;
    using ColumnCoefficients = ColCoeffs;
    static constexpr std::size_t dimension = MatrixTraits<NestedMatrix>::dimension;
    static constexpr std::size_t columns = MatrixTraits<NestedMatrix>::columns;
    using Scalar = typename MatrixTraits<NestedMatrix>::Scalar;
    static_assert(RowCoefficients::size == dimension);
    static_assert(ColumnCoefficients::size == columns);

    template<std::size_t rows = dimension, std::size_t cols = columns, typename S = Scalar>
    using NativeMatrix = typename MatrixTraits<NestedMatrix>::template NativeMatrix<rows, cols, S>;

    using SelfContained = Matrix<RowCoefficients, ColumnCoefficients, self_contained_t<NestedMatrix>>;

#ifdef __cpp_concepts
    template<coefficients RC = RowCoefficients, coefficients CC = ColumnCoefficients, typed_matrix_nestable Arg>
#else
    template<typename RC = RowCoefficients, typename CC = ColumnCoefficients, typename Arg, std::enable_if_t<
      typed_matrix_nestable<Arg>, int> = 0>
#endif
    static auto make(Arg&& arg)
    {
      static_assert(MatrixTraits<Arg>::dimension == RC::size);
      if constexpr(MatrixTraits<Arg>::columns == CC::size)
      {
        return Matrix<RC, CC, std::decay_t<Arg>>(std::forward<Arg>(arg));
      }
      else
      {
        return Mean<RC, std::decay_t<Arg>>(std::forward<Arg>(arg));
      }
    }

    static auto zero() { return make(MatrixTraits<NestedMatrix>::zero()); }

    static auto identity() { return make<RowCoefficients, RowCoefficients>(MatrixTraits<NestedMatrix>::identity()); }

  };

} // namespace OpenKalman


#endif //OPENKALMAN_MATRIX_HPP
