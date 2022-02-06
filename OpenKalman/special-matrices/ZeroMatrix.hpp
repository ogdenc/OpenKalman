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
 * \brief Definitions for Eigen3::ZeroMatrix
 */

#ifndef OPENKALMAN_EIGEN3_ZEROMATRIX_HPP
#define OPENKALMAN_EIGEN3_ZEROMATRIX_HPP

#include <type_traits>

namespace OpenKalman::Eigen3
{
  // ------------ //
  //  ZeroMatrix  //
  // ------------ //

  // ZeroMatrix is declared in eigen3-forward-declarations.hpp.

  template<typename Scalar, std::size_t rows_, std::size_t columns>
  struct ZeroMatrix : Eigen3::internal::Eigen3Base<ZeroMatrix<Scalar, rows_, columns>>,
    Eigen3::internal::EigenDynamicBase<Scalar, rows_, columns>
  {

  private:

    using Base = Eigen3::internal::EigenDynamicBase<Scalar, rows_, columns>;

  public:

    using Base::rows;

    using Base::cols;

    /**
     * \brief Construct a ZeroMatrix.
     * \details The constructor can take a number of arguments representing the number of dynamic dimensions.
     * For example, ZeroMatrix {2, 3} constructs a 2-by-3 dynamic matrix, ZeroMatrix {3} constructs a
     * 2-by-3 matrix in which there are two fixed row dimensions and three dynamic column dimensions, and
     * ZeroMatrix {} constructs a fixed matrix.
     */
#ifdef __cpp_concepts
    template<std::convertible_to<std::size_t> ... Args> requires
      (sizeof...(Args) == (rows_ == dynamic_extent ? 1 : 0) + (columns == dynamic_extent ? 1 : 0))
#else
    template<typename...Args, std::enable_if_t<(std::is_convertible_v<Args, std::size_t> and ...) and
      (sizeof...(Args) == (rows_ == dynamic_extent ? 1 : 0) + (columns == dynamic_extent ? 1 : 0)), int> = 0>
#endif
    ZeroMatrix(const Args...args) : Base {static_cast<std::size_t>(args)...} {}


    /**
     * \internal
     * \brief Construct a ZeroMatrix from another zero_matrix.
     * \tparam M A zero_matrix with a compatible shape.
     */
#ifdef __cpp_concepts
    template<zero_matrix M> requires (not std::same_as<M, ZeroMatrix>) and
      (dynamic_rows<M> or rows_ == dynamic_extent or row_extent_of_v<M> == rows_) and
      (dynamic_columns<M> or columns == dynamic_extent or column_extent_of_v<M> == columns)
#else
    template<typename M, std::enable_if_t<zero_matrix<M> and (not std::is_same_v<M, ZeroMatrix>) and
      (dynamic_rows<M> or rows_ == dynamic_extent or row_extent_of<M>::value == rows_) and
      (dynamic_columns<M> or columns == dynamic_extent or column_extent_of<M>::value == columns), int> = 0>
#endif
    ZeroMatrix(M&& m) : Base {std::forward<M>(m)} {}


    /**
     * \internal
     * \brief Assign from another compatible zero_matrix.
     */
#ifdef __cpp_concepts
    template<zero_matrix M> requires (not std::same_as<M, ZeroMatrix>) and
      (dynamic_rows<M> or rows_ == dynamic_extent or row_extent_of_v<M> == rows_) and
      (dynamic_columns<M> or columns == dynamic_extent or column_extent_of_v<M> == columns)
#else
    template<typename M, std::enable_if_t<zero_matrix<M> and (not std::is_same_v<M, ZeroMatrix>) and
      (dynamic_rows<M> or rows_ == dynamic_extent or row_extent_of<M>::value == rows_) and
      (dynamic_columns<M> or columns == dynamic_extent or column_extent_of<M>::value == columns), int> = 0>
#endif
    auto& operator=(M&& m)
    {
      Base::operator=(std::forward<M>(m));
      return *this;
    }


    /**
     * \brief Element accessor.
     * \param r The row.
     * \param c The column.
     * \return The element at row r and column c (always zero of type Scalar).
     */
    constexpr Scalar operator()(std::size_t r, std::size_t c) const
    {
      assert(Eigen::Index(r) < rows());
      assert(Eigen::Index(c) < cols());
      return 0;
    }


    /**
     * \brief Element accessor for a row or column vector.
     * \param i The row or column.
     * \return The element at row or column i (always zero of type Scalar).
     */
#ifdef __cpp_concepts
    constexpr Scalar operator[](std::size_t i) const
    requires (rows_ == 1) or (columns == 1)
#else
    template<std::size_t r = rows_, std::enable_if_t<(r == 1) or (columns == 1), int> = 0>
      constexpr Scalar operator[](std::size_t i) const
#endif
    {
      assert(rows_ == 1 or Eigen::Index(i) < rows());
      assert(columns == 1 or Eigen::Index(i) < cols());
      return 0;
    }


    /**
     * \brief Element accessor for a row or column vector.
     * \param i The row or column.
     * \return The element at row or column i (always zero of type Scalar).
     */
#ifdef __cpp_concepts
    constexpr Scalar operator()(std::size_t i) const
    requires (rows_ == 1) or (columns == 1)
#else
    template<std::size_t r = rows_, std::enable_if_t<(r == 1) or (columns == 1), int> = 0>
      constexpr Scalar operator()(std::size_t i) const
#endif
    {
      assert(rows_ == 1 or Eigen::Index(i) < rows());
      assert(columns == 1 or Eigen::Index(i) < cols());
      return 0;
    }

  };


  // ------------------------------ //
  //        Deduction guide         //
  // ------------------------------ //

#ifdef __cpp_concepts
  template<zero_matrix Arg>
#else
  template<typename Arg, std::enable_if_t<zero_matrix<Arg>, int> = 0>
#endif
  ZeroMatrix(Arg&&)
    -> ZeroMatrix<scalar_type_of_t<Arg>, row_extent_of_v<Arg>, column_extent_of_v<Arg>>;


} // OpenKalman::Eigen3



#endif //OPENKALMAN_EIGEN3_ZEROMATRIX_HPP
