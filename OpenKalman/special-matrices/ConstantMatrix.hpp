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
    (sizeof...(Args) == (rows_ == dynamic_extent ? 1 : 0) + (columns == dynamic_extent ? 1 : 0))
#else
    template<typename...Args, std::enable_if_t<(std::is_convertible_v<Args, std::size_t> and ...) and
      (sizeof...(Args) == (rows_ == dynamic_extent ? 1 : 0) + (columns == dynamic_extent ? 1 : 0)), int> = 0>
#endif
    ConstantMatrix(const Args...args) : Base {static_cast<std::size_t>(args)...} {}

#ifndef __cpp_concepts
  private:

    template<typename T, typename = void>
    struct constant_arg_matches : std::false_type {};

    template<typename T>
    struct constant_arg_matches<T, std::enable_if_t<(constant_coefficient<std::decay_t<T>>::value == constant) and
      (dynamic_rows<T> or rows_ == dynamic_extent or row_extent_of<std::decay_t<T>>::value == rows_) and
      (dynamic_columns<T> or columns == dynamic_extent or column_extent_of<std::decay_t<T>>::value == columns)>>
      : std::true_type {};

  public:
#endif

    /**
     * \internal
     * \brief Construct a ConstantMatrix from another constant matrix.
     * \tparam M A constant_matrix with a compatible shape.
     */
#ifdef __cpp_concepts
    template<constant_matrix M>
    requires (not std::same_as<M, ConstantMatrix>) and (constant_coefficient_v<M> == constant) and
      (dynamic_rows<M> or rows_ == dynamic_extent or row_extent_of_v<M> == rows_) and
      (dynamic_columns<M> or columns == dynamic_extent or column_extent_of_v<M> == columns)
#else
    template<typename M, std::enable_if_t<constant_matrix<M> and
      (not std::is_same_v<M, ConstantMatrix>) and constant_arg_matches<M>::value, int> = 0>
#endif
    ConstantMatrix(M&& m) : Base {std::forward<M>(m)} {}


    /**
     * \internal
     * \brief Assign from another compatible constant_matrix.
     */
#ifdef __cpp_concepts
    template<constant_matrix M>
    requires (not std::same_as<M, ConstantMatrix>) and (constant_coefficient_v<M> == constant) and
      (dynamic_rows<M> or rows_ == dynamic_extent or row_extent_of_v<M> == rows_) and
      (dynamic_columns<M> or columns == dynamic_extent or column_extent_of_v<M> == columns)
#else
    template<typename M, std::enable_if_t<constant_matrix<M> and
      (not std::is_same_v<M, ConstantMatrix>) and constant_arg_matches<M>::value, int> = 0>
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

  };


  // ------------------------------ //
  //        Deduction guide         //
  // ------------------------------ //

#ifdef __cpp_concepts
  template<constant_matrix Arg>
#else
  template<typename Arg, std::enable_if_t<constant_matrix<Arg>, int> = 0>
#endif
  ConstantMatrix(Arg&&) -> ConstantMatrix<scalar_type_of_t<Arg>,
    constant_coefficient_v<Arg>, row_extent_of_v<Arg>, column_extent_of_v<Arg>>;


} // OpenKalman::Eigen3


#endif //OPENKALMAN_EIGEN3_CONSTANTMATRIX_HPP
