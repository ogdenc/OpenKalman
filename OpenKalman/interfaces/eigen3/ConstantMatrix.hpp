/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2021 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definitions for Eigen3::ConstantMatrix
 */

#ifndef OPENKALMAN_EIGEN3_CONSTANTMATRIX_HPP
#define OPENKALMAN_EIGEN3_CONSTANTMATRIX_HPP

#include <type_traits>

namespace OpenKalman::Eigen3
{

  // ConstantMatrix is declared in eigen3-forward-declarations.hpp.

#ifdef __cpp_concepts
  template<arithmetic_or_complex Scalar, auto constant, std::size_t rows_, std::size_t columns>
  requires std::convertible_to<decltype(constant), Scalar>
#else
  template<typename Scalar, auto constant, std::size_t rows_, std::size_t columns>
#endif
  struct ConstantMatrix : Eigen3::internal::Eigen3Base<ConstantMatrix<Scalar, constant, rows_, columns>>,
    Eigen3::internal::EigenDynamicBase<Scalar, rows_, columns>
  {

  private:

#ifndef __cpp_concepts
    static_assert(arithmetic_or_complex<Scalar>);
    static_assert(std::is_convertible_v<decltype(constant), Scalar>);
#endif

    using Base = Eigen3::internal::EigenDynamicBase<Scalar, rows_, columns>;

  public:

    using Base::rows;

    using Base::cols;

    /**
     * \brief Construct a ConstantMatrix.
     * \details The constructor can take a number of arguments representing the number of dynamic dimensions.
     * For example, ConstantMatrix {2, 3} constructs a 2-by-3 dynamic matrix, ConstantMatrix {3} constructs a
     * 2-by-3 matrix in which there are two fixed row dimensions and three dynamic column dimensions, and
     * ConstantMatrix {} constructs a fixed matrix.
     */
#ifdef __cpp_concepts
    template<std::convertible_to<std::size_t> ... Args> requires
    (sizeof...(Args) == (rows_ == 0 ? 1 : 0) + (columns == 0 ? 1 : 0))
#else
    template<typename...Args, std::enable_if_t<(std::is_convertible_v<Args, std::size_t> and ...) and
      (sizeof...(Args) == (rows_ == 0 ? 1 : 0) + (columns == 0 ? 1 : 0)), int> = 0>
#endif
    ConstantMatrix(const Args...args) : Base {static_cast<std::size_t>(args)...} {}


    /**
     * \internal
     * \brief Construct a ConstantMatrix from another constant matrix.
     * \tparam M A constant_matrix with a compatible shape.
     */
#ifdef __cpp_concepts
    template<constant_matrix M>
    requires (not std::same_as<M, ConstantMatrix>) and (constant_coefficient_v<M> == constant) and
      (dynamic_rows<M> or rows_ == 0 or MatrixTraits<M>::rows == rows_) and
      (dynamic_columns<M> or columns == 0 or MatrixTraits<M>::columns == columns)
#else
    template<typename M, std::enable_if_t<constant_matrix<M> and
      (not std::is_same_v<M, ConstantMatrix>) and (constant_coefficient_v<M> == constant) and
      (dynamic_rows<M> or rows_ == 0 or MatrixTraits<M>::rows == rows_) and
      (dynamic_columns<M> or columns == 0 or MatrixTraits<M>::columns == columns), int> = 0>
#endif
    ConstantMatrix(M&& m) : Base {std::forward<M>(m)} {}


    /**
     * \internal
     * \brief Assign from another compatible constant_matrix.
     */
#ifdef __cpp_concepts
    template<constant_matrix M>
    requires (not std::same_as<M, ConstantMatrix>) and (constant_coefficient_v<M> == constant) and
      (dynamic_rows<M> or rows_ == 0 or MatrixTraits<M>::rows == rows_) and
      (dynamic_columns<M> or columns == 0 or MatrixTraits<M>::columns == columns)
#else
    template<typename M, std::enable_if_t<constant_matrix<M> and
      (not std::is_same_v<M, ConstantMatrix>) and (constant_coefficient_v<M> == constant) and
      (dynamic_rows<M> or rows_ == 0 or MatrixTraits<M>::rows == rows_) and
      (dynamic_columns<M> or columns == 0 or MatrixTraits<M>::columns == columns), int> = 0>
#endif
    auto& operator=(M&& m)
    {
      Base::operator=(std::forward<M>(m));
      return *this;
    }


    /**
     * \brief Element accessor.
     * \note Does not do any bounds checking.
     * \param r The row.
     * \param c The column.
     * \return The element at row r and column c (always the constant).
     */
    constexpr Scalar operator()(std::size_t r, std::size_t c) const
    {
      assert(Eigen::Index(r) < this->rows());
      assert(Eigen::Index(c) < this->cols());
      return constant;
    }


    /**
     * \brief Element accessor for a row or column vector.
     * \param i The row or column.
     * \return The element at row or column i (always the constant).
     */
#ifdef __cpp_concepts
    constexpr Scalar operator[](std::size_t i) const requires (rows_ == 1) or (columns == 1)
#else
    template<std::size_t r = rows_, std::enable_if_t<(r == 1) or (columns == 1), int> = 0>
    constexpr Scalar operator[](std::size_t i) const
#endif
    {
      assert(rows_ == 1 or Eigen::Index(i) < this->rows());
      assert(columns == 1 or Eigen::Index(i) < this->cols());
      return constant;
    }


    /**
     * \brief Element accessor for a row or column vector.
     * \param i The row or column.
     * \return The element at row or column i (always the constant).
     */
#ifdef __cpp_concepts
    constexpr Scalar operator()(std::size_t i) const requires (rows_ == 1) or (columns == 1)
#else
    template<std::size_t r = rows_, std::enable_if_t<(r == 1) or (columns == 1), int> = 0>
    constexpr Scalar operator()(std::size_t i) const
#endif
    {
      assert(rows_ == 1 or Eigen::Index(i) < this->rows());
      assert(columns == 1 or Eigen::Index(i) < this->cols());
      return constant;
    }


    /// \internal \note Eigen 3 requires this for it to be used in an Eigen::CwiseBinaryOp.
    using Nested = ConstantMatrix;

  };


  // ------------------------------ //
  //        Deduction guide         //
  // ------------------------------ //

#ifdef __cpp_concepts
  template<constant_matrix Arg>
#else
  template<typename Arg, std::enable_if_t<constant_matrix<Arg>, int> = 0>
#endif
  ConstantMatrix(Arg&&) -> ConstantMatrix<typename MatrixTraits<Arg>::Scalar,
    constant_coefficient_v<Arg>, MatrixTraits<Arg>::rows, MatrixTraits<Arg>::columns>;


} // OpenKalman::Eigen3


namespace OpenKalman
{
  // -------- //
  //  Traits  //
  // -------- //

  template<typename Scalar_, auto constant, std::size_t rows_, std::size_t columns_>
  struct MatrixTraits<Eigen3::ConstantMatrix<Scalar_, constant, rows_, columns_>>
  {
    using Scalar = Scalar_;

    static constexpr std::size_t rows = rows_;

    static constexpr std::size_t columns = columns_;


  private:

    using Matrix = Eigen3::ConstantMatrix<Scalar, constant, rows, columns>;

  public:

    template<std::size_t r = rows, std::size_t c = columns, typename S = Scalar>
    using NativeMatrixFrom = Eigen3::eigen_matrix_t<S, r, c>;


    using SelfContainedFrom = Matrix;


    template<TriangleType storage_triangle = TriangleType::diagonal>
    using SelfAdjointMatrixFrom = Eigen3::SelfAdjointMatrix<Matrix, storage_triangle>;


    template<TriangleType triangle_type = TriangleType::diagonal>
    using TriangularMatrixFrom = Eigen3::TriangularMatrix<Matrix, triangle_type>;


    template<std::size_t dim = rows, typename S = Scalar>
    using DiagonalMatrixFrom = Eigen3::DiagonalMatrix<Eigen3::ConstantMatrix<S, constant, dim, 1>>;


    template<typename Derived>
    using MatrixBaseFrom = Eigen3::internal::Eigen3MatrixBase<Derived, Matrix>;


#ifdef __cpp_concepts
    template<std::convertible_to<std::size_t> ... Args> requires (sizeof...(Args) == (rows == 0 ? 1 : 0) + (columns == 0 ? 1 : 0))
#else
    template<typename...Args, std::enable_if_t<(std::is_convertible_v<Args, std::size_t> and ...) and
      (sizeof...(Args) == (rows == 0 ? 1 : 0) + (columns == 0 ? 1 : 0)), int> = 0>
#endif
    static auto zero(const Args...args)
    {
      return Eigen3::ZeroMatrix<Scalar, rows, columns> {static_cast<std::size_t>(args)...};
    }


#ifdef __cpp_concepts
    template<std::convertible_to<Eigen::Index> ... Args>
    requires (sizeof...(Args) >= (rows == 0 and columns == 0 ? 1 : 0)) and (sizeof...(Args) <= 1)
#else
    template<typename...Args, std::enable_if_t<(std::is_convertible_v<Args, Eigen::Index> and ...) and
      (sizeof...(Args) >= (rows == 0 and columns == 0 ? 1 : 0)) and (sizeof...(Args) <= 1), int> = 0>
#endif
    static auto identity(const Args...args)
    {
      constexpr auto r = sizeof...(Args) == 0 ? (rows == 0 ? columns : rows) : 0;

      return Eigen3::eigen_matrix_t<Scalar, r, r>::Identity(
        static_cast<Eigen::Index>(args)..., static_cast<Eigen::Index>(args)...);
    }


  };

} // namespace OpenKalman



#endif //OPENKALMAN_EIGEN3_CONSTANTMATRIX_HPP
