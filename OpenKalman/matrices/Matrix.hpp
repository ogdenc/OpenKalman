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
  requires (RowCoefficients::dimension == row_dimension_of_v<NestedMatrix>) and
    (ColumnCoefficients::dimension == column_dimension_of_v<NestedMatrix>) and
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
    static_assert(RowCoefficients::dimension == row_dimension_of_v<NestedMatrix>);
    static_assert(ColumnCoefficients::dimension == column_dimension_of_v<NestedMatrix>);
    static_assert(not std::is_rvalue_reference_v<NestedMatrix>);
    static_assert(dynamic_coefficients<RowCoefficients> == dynamic_rows<NestedMatrix>);
    static_assert(dynamic_coefficients<ColumnCoefficients> == dynamic_columns<NestedMatrix>);
#endif

    using Scalar = scalar_type_of_t<NestedMatrix>; ///< Scalar type for this matrix.

  private:

    using Base = oin::TypedMatrixBase<Matrix, NestedMatrix, RowCoefficients, ColumnCoefficients>;

  public:

    using Base::Base;


    /// Construct from a compatible \ref OpenKalman::typed_matrix "typed_matrix".
#ifdef __cpp_concepts
    template<typed_matrix Arg> requires (not std::derived_from<std::decay_t<Arg>, Matrix>) and
      (not euclidean_transformed<Arg>) and
      equivalent_to<row_coefficient_types_of_t<Arg>, RowCoefficients> and
      equivalent_to<column_coefficient_types_of_t<Arg>, ColumnCoefficients> and
      //requires(Arg&& arg) { NestedMatrix {nested_matrix(std::forward<Arg>(arg))}; } // \todo 't work in GCC 10
      std::constructible_from<NestedMatrix, decltype(nested_matrix(std::declval<Arg&&>()))>
#else
    template<typename Arg, std::enable_if_t<typed_matrix<Arg> and not std::is_base_of_v<Matrix, std::decay_t<Arg>> and
      not euclidean_transformed<Arg> and
      equivalent_to<row_coefficient_types_of_t<Arg>, RowCoefficients> and
      equivalent_to<column_coefficient_types_of_t<Arg>, ColumnCoefficients> and
      std::is_constructible_v<NestedMatrix, decltype(nested_matrix(std::declval<Arg&&>()))>, int> = 0>
#endif
    Matrix(Arg&& arg) noexcept : Base {nested_matrix(std::forward<Arg>(arg))} {}


    /// Construct from a compatible \ref OpenKalman::euclidean_transformed "euclidean_transformed".
#ifdef __cpp_concepts
    template<euclidean_transformed Arg> requires
      equivalent_to<row_coefficient_types_of_t<Arg>, RowCoefficients> and
      equivalent_to<column_coefficient_types_of_t<Arg>, ColumnCoefficients> and
      requires(Arg&& arg) { NestedMatrix {from_euclidean<RowCoefficients>(nested_matrix(std::forward<Arg>(arg)))}; }
#else
    template<typename Arg, std::enable_if_t<typed_matrix<Arg> and euclidean_transformed<Arg> and
      equivalent_to<row_coefficient_types_of_t<Arg>, RowCoefficients> and
      equivalent_to<column_coefficient_types_of_t<Arg>, ColumnCoefficients> and
      std::is_constructible_v<NestedMatrix,
        decltype(from_euclidean<RowCoefficients>(nested_matrix(std::declval<Arg&&>())))>, int> = 0>
#endif
    Matrix(Arg&& arg) noexcept
      : Base {from_euclidean<RowCoefficients>(nested_matrix(std::forward<Arg>(arg)))} {}


    /// Construct from compatible \ref OpenKalman::typed_matrix_nestable "typed_matrix_nestable".
#ifdef __cpp_concepts
    template<typed_matrix_nestable Arg> requires (row_dimension_of_v<Arg> == row_dimension_of_v<NestedMatrix>) and
      (column_dimension_of_v<Arg> == column_dimension_of_v<NestedMatrix>) and
      std::constructible_from<NestedMatrix, Arg&&>
#else
    template<typename Arg, std::enable_if_t<typed_matrix_nestable<Arg> and
      (row_dimension_of<Arg>::value == row_dimension_of<NestedMatrix>::value) and
      (column_dimension_of<Arg>::value == column_dimension_of<NestedMatrix>::value) and
      std::is_constructible_v<NestedMatrix, Arg&&>, int> = 0>
#endif
    explicit Matrix(Arg&& arg) noexcept : Base {std::forward<Arg>(arg)} {}


    /// Construct from compatible \ref OpenKalman::covariance "covariance".
#ifdef __cpp_concepts
    template<covariance Arg> requires
      equivalent_to<row_coefficient_types_of_t<Arg>, RowCoefficients> and
      equivalent_to<row_coefficient_types_of_t<Arg>, ColumnCoefficients> and
      requires(Arg&& arg) { NestedMatrix {make_dense_writable_matrix_from(std::forward<Arg>(arg))}; }
#else
    template<typename Arg, std::enable_if_t<covariance<Arg> and
      equivalent_to<row_coefficient_types_of_t<Arg>, RowCoefficients> and
      equivalent_to<row_coefficient_types_of_t<Arg>, ColumnCoefficients> and
      std::is_constructible_v<NestedMatrix, dense_writable_matrix_t<Arg>>, int> = 0>
#endif
    Matrix(Arg&& arg) noexcept : Base {make_dense_writable_matrix_from(std::forward<Arg>(arg))} {}


    /// Assign from a compatible \ref OpenKalman::typed_matrix "typed_matrix".
#ifdef __cpp_concepts
    template<typed_matrix Arg> requires (not euclidean_transformed<Arg>) and
      (not std::derived_from<std::decay_t<Arg>, Matrix>) and
      equivalent_to<row_coefficient_types_of_t<Arg>, RowCoefficients> and
      equivalent_to<column_coefficient_types_of_t<Arg>, ColumnCoefficients> and
      modifiable<NestedMatrix, nested_matrix_of_t<Arg>>
#else
    template<typename Arg, std::enable_if_t<typed_matrix<Arg> and (not euclidean_transformed<Arg>) and
      (not std::is_base_of_v<Matrix, std::decay_t<Arg>>) and
      equivalent_to<row_coefficient_types_of_t<Arg>, RowCoefficients> and
      equivalent_to<column_coefficient_types_of_t<Arg>, ColumnCoefficients> and
      modifiable<NestedMatrix, nested_matrix_of_t<Arg>>, int> = 0>
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
      equivalent_to<row_coefficient_types_of_t<Arg>, RowCoefficients> and
      equivalent_to<column_coefficient_types_of_t<Arg>, ColumnCoefficients> and
      modifiable<NestedMatrix, decltype(from_euclidean<RowCoefficients>(std::declval<nested_matrix_of_t<Arg>>()))>
#else
    template<typename Arg, std::enable_if_t<euclidean_transformed<Arg> and
      equivalent_to<row_coefficient_types_of_t<Arg>, RowCoefficients> and
      equivalent_to<column_coefficient_types_of_t<Arg>, ColumnCoefficients> and
      modifiable<NestedMatrix, decltype(from_euclidean<RowCoefficients>(std::declval<nested_matrix_of_t<Arg>>()))>,
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
      equivalent_to<row_coefficient_types_of_t<Arg>, RowCoefficients> and
      equivalent_to<column_coefficient_types_of_t<Arg>, ColumnCoefficients>
#else
    template<typename Arg, std::enable_if_t<typed_matrix<Arg> and
      equivalent_to<row_coefficient_types_of_t<Arg>, RowCoefficients> and
      equivalent_to<column_coefficient_types_of_t<Arg>, ColumnCoefficients>, int> = 0>
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
      apply_columnwise([&arg](auto& col) { col += arg().nested_matrix(); }, this->nested_matrix());
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
      equivalent_to<row_coefficient_types_of_t<Arg>, RowCoefficients> and
      equivalent_to<column_coefficient_types_of_t<Arg>, ColumnCoefficients>
#else
    template<typename Arg, std::enable_if_t<typed_matrix<Arg> and
      equivalent_to<row_coefficient_types_of_t<Arg>, RowCoefficients> and
      equivalent_to<column_coefficient_types_of_t<Arg>, ColumnCoefficients>, int> = 0>
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
      apply_columnwise([&arg](auto& col){ col -= arg().nested_matrix(); }, this->nested_matrix());
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
  explicit Matrix(M&&) -> Matrix<Axes<row_dimension_of_v<M>>, Axes<column_dimension_of_v<M>>, passable_t<M>>;


  /// Deduce template parameters from a non-Euclidean-transformed typed matrix.
#ifdef __cpp_concepts
  template<typed_matrix V> requires (not euclidean_transformed<V>)
#else
  template<typename V, std::enable_if_t<typed_matrix<V> and not euclidean_transformed<V>, int> = 0>
#endif
  Matrix(V&&) -> Matrix<
    row_coefficient_types_of_t<V>,
    column_coefficient_types_of_t<V>,
    passable_t<nested_matrix_of_t<V>>>;


  /// Deduce template parameters from a Euclidean-transformed typed matrix.
#if defined(__cpp_concepts) and OPENKALMAN_CPP_FEATURE_CONCEPTS_2
  // \todo Unlike SFINAE version, this incorrectly matches V==Mean and V==Matrix in both GCC 10.1.0 and clang 10.0.0:
  template<euclidean_transformed V> requires untyped_columns<V>
#else
  template<typename V, std::enable_if_t<euclidean_transformed<V> and untyped_columns<V>, int> = 0>
#endif
  Matrix(V&&) -> Matrix<
    row_coefficient_types_of_t<V>,
    column_coefficient_types_of_t<V>,
    decltype(from_euclidean<row_coefficient_types_of_t<V>>(
      nested_matrix(std::forward<V>(std::declval<V>()))))>;


  /// Deduce parameter types from a Covariance.
#ifdef __cpp_concepts
  template<covariance V>
#else
  template<typename V, std::enable_if_t<covariance<V>, int> = 0>
#endif
  Matrix(V&&) -> Matrix<
    row_coefficient_types_of_t<V>,
    row_coefficient_types_of_t<V>,
    dense_writable_matrix_t<nested_matrix_of_t<V>>>;


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
    requires (row_dimension_of_v<M> == RowCoefficients::dimension) and
    (column_dimension_of_v<M> == ColumnCoefficients::dimension)
#else
  template<typename RowCoefficients, typename ColumnCoefficients, typename M, std::enable_if_t<
    coefficients<RowCoefficients> and coefficients<ColumnCoefficients> and typed_matrix_nestable<M> and
    (row_dimension_of<M>::value == RowCoefficients::dimension) and
    (column_dimension_of<M>::value == ColumnCoefficients::dimension), int> = 0>
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
  requires (row_dimension_of_v<M> == RowCoefficients::dimension)
#else
  template<typename RowCoefficients, typename M, std::enable_if_t<
    coefficients<RowCoefficients> and typed_matrix_nestable<M> and
    (row_dimension_of<M>::value == RowCoefficients::dimension), int> = 0>
#endif
  inline auto make_matrix(M&& m)
  {
    using ColumnCoefficients = Axes<column_dimension_of_v<M>>;
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
    using RowCoeffs = Axes<row_dimension_of_v<M>>;
    using ColCoeffs = Axes<column_dimension_of_v<M>>;
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
    using C = row_coefficient_types_of_t<M>;
    return make_matrix<C, C>(make_dense_writable_matrix_from(std::forward<M>(arg)));
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
    using RowCoeffs = row_coefficient_types_of_t<Arg>;
    using ColCoeffs = column_coefficient_types_of_t<Arg>;
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
    (row_dimension_of_v<M> == RowCoefficients::dimension) and
    (column_dimension_of_v<M> == ColumnCoefficients::dimension)
#else
  template<typename RowCoefficients, typename ColumnCoefficients, typename M, std::enable_if_t<
    coefficients<RowCoefficients> and coefficients<ColumnCoefficients> and typed_matrix_nestable<M> and
    (row_dimension_of<M>::value == RowCoefficients::dimension) and
    (column_dimension_of<M>::value == ColumnCoefficients::dimension), int> = 0>
#endif
  inline auto make_matrix()
  {
    return Matrix<RowCoefficients, ColumnCoefficients, dense_writable_matrix_t<M>>();
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
    using RowCoeffs = Axes<row_dimension_of_v<M>>;
    using ColCoeffs = Axes<column_dimension_of_v<M>>;
    return make_matrix<RowCoeffs, ColCoeffs, M>();
  }


} // namespace OpenKalman


#endif //OPENKALMAN_MATRIX_HPP
