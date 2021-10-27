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
#else
  template<typename Scalar, auto constant, std::size_t rows_, std::size_t columns>
#endif
  struct ConstantMatrix : Eigen3::internal::Eigen3Base<ConstantMatrix<Scalar, constant, rows_, columns>>,
    Eigen3::internal::EigenDynamicBase<Scalar, rows_, columns>
  {

  private:

#ifndef __cpp_concepts
    static_assert(arithmetic_or_complex<Scalar>);
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


} // OpenKalman::Eigen3


namespace OpenKalman
{
  // -------- //
  //  Traits  //
  // -------- //

  template<typename Scalar_, auto constant_, std::size_t rows_, std::size_t columns_>
  struct MatrixTraits<Eigen3::ConstantMatrix<Scalar_, constant_, rows_, columns_>>
  {
    using Scalar = Scalar_;

    static constexpr auto constant = constant_;

    static constexpr std::size_t rows = rows_;

    static constexpr std::size_t columns = columns_;


  private:

    using Matrix = Eigen3::ConstantMatrix<Scalar, constant, rows, columns>;

  public:

    template<std::size_t r = rows, std::size_t c = columns, typename S = Scalar>
    using NativeMatrixFrom = Eigen::Matrix<S, r, c>;


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
    template<std::convertible_to<Eigen::Index> ... Args> requires (sizeof...(Args) == (rows == 0 ? 1 : 0))
#else
    template<typename...Args, std::enable_if_t<(std::is_convertible_v<Args, Eigen::Index> and ...) and
      (sizeof...(Args) == (rows == 0 ? 1 : 0)), int> = 0>
#endif
    static constexpr auto identity(const Args...args)
    {
      return Eigen3::eigen_matrix_t<Scalar, rows, rows>::Identity(
        static_cast<Eigen::Index>(args)..., static_cast<Eigen::Index>(args)...);
    }


  };

} // namespace OpenKalman



#endif //OPENKALMAN_EIGEN3_CONSTANTMATRIX_HPP
