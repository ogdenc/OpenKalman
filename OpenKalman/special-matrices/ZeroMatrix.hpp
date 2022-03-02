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

#ifdef __cpp_concepts
  template<indexible PatternMatrix>
#else
  template<typename PatternMatrix>
#endif
  struct ZeroMatrix : Eigen3::internal::EigenDynamicBase<ZeroMatrix<PatternMatrix>, PatternMatrix>
  {

  private:

    using nested_scalar = scalar_type_of_t<PatternMatrix>;
    static constexpr auto nested_rows = row_dimension_of_v<PatternMatrix>;
    static constexpr auto nested_cols = column_dimension_of_v<PatternMatrix>;
    using Base = Eigen3::internal::EigenDynamicBase<ZeroMatrix, PatternMatrix>;

  public:

    /**
     * \brief Construct a ZeroMatrix.
     * \details The constructor can take a number of arguments representing the number of dynamic dimensions.
     * For example, ZeroMatrix {2, 3} constructs a 2-by-3 dynamic matrix, ZeroMatrix {3} constructs a
     * 2-by-3 matrix in which there are two fixed row dimensions and three dynamic column dimensions, and
     * ZeroMatrix {} constructs a fixed matrix.
     */
#ifdef __cpp_concepts
    template<std::convertible_to<std::size_t> ... Args> requires
      (sizeof...(Args) == (nested_rows == dynamic_size ? 1 : 0) + (nested_cols == dynamic_size ? 1 : 0))
#else
    template<typename...Args, std::enable_if_t<(std::is_convertible_v<Args, std::size_t> and ...) and
      (sizeof...(Args) == (nested_rows == dynamic_size ? 1 : 0) + (nested_cols == dynamic_size ? 1 : 0)), int> = 0>
#endif
    ZeroMatrix(const Args...args) : Base {static_cast<std::size_t>(args)...} {}


    /**
     * \internal
     * \brief Construct a ZeroMatrix from another zero_matrix.
     * \tparam M A zero_matrix with a compatible shape.
     */
#ifdef __cpp_concepts
    template<zero_matrix M> requires (not std::same_as<M, ZeroMatrix>) and
      (dynamic_rows<M> or nested_rows == dynamic_size or row_dimension_of_v<M> == nested_rows) and
      (dynamic_columns<M> or nested_cols == dynamic_size or column_dimension_of_v<M> == nested_cols)
#else
    template<typename M, std::enable_if_t<zero_matrix<M> and (not std::is_same_v<M, ZeroMatrix>) and
      (dynamic_rows<M> or nested_rows == dynamic_size or row_dimension_of<M>::value == nested_rows) and
      (dynamic_columns<M> or nested_cols == dynamic_size or column_dimension_of<M>::value == nested_cols), int> = 0>
#endif
    ZeroMatrix(M&& m) : Base {std::forward<M>(m)} {}


    /**
     * \internal
     * \brief Assign from another compatible zero_matrix.
     */
#ifdef __cpp_concepts
    template<zero_matrix M> requires (not std::same_as<M, ZeroMatrix>) and
      (dynamic_rows<M> or nested_rows == dynamic_size or row_dimension_of_v<M> == nested_rows) and
      (dynamic_columns<M> or nested_cols == dynamic_size or column_dimension_of_v<M> == nested_cols)
#else
    template<typename M, std::enable_if_t<zero_matrix<M> and (not std::is_same_v<M, ZeroMatrix>) and
      (dynamic_rows<M> or nested_rows == dynamic_size or row_dimension_of<M>::value == nested_rows) and
      (dynamic_columns<M> or nested_cols == dynamic_size or column_dimension_of<M>::value == nested_cols), int> = 0>
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
     * \return The element at row r and column c (always zero of type nested_scalar).
     */
    constexpr nested_scalar operator()(std::size_t r, std::size_t c) const
    {
      assert(Eigen::Index(r) < runtime_dimension_of<0>(*this));
      assert(Eigen::Index(c) < runtime_dimension_of<1>(*this));
      return 0;
    }


    /**
     * \brief Element accessor for a row or column vector.
     * \param i The row or column.
     * \return The element at row or column i (always zero of type nested_scalar).
     */
#ifdef __cpp_concepts
    constexpr nested_scalar
    operator[](std::size_t i) const requires (nested_rows == 1) or (nested_cols == 1)
#else
    template<std::size_t r = nested_rows, std::enable_if_t<(r == 1) or (nested_cols == 1), int> = 0>
    constexpr nested_scalar
    operator[](std::size_t i) const
#endif
    {
      if constexpr (nested_rows != 1) assert(Eigen::Index(i) < runtime_dimension_of<0>(*this));
      if constexpr (nested_cols != 1) assert(Eigen::Index(i) < runtime_dimension_of<1>(*this));
      return 0;
    }


    /**
     * \brief Element accessor for a row or column vector.
     * \param i The row or column.
     * \return The element at row or column i (always zero of type nested_scalar).
     */
#ifdef __cpp_concepts
    constexpr nested_scalar
    operator()(std::size_t i) const requires (nested_rows == 1) or (nested_cols == 1)
#else
    template<std::size_t r = nested_rows, std::enable_if_t<(r == 1) or (nested_cols == 1), int> = 0>
    constexpr nested_scalar
    operator()(std::size_t i) const
#endif
    {
      if constexpr (nested_rows != 1) assert(Eigen::Index(i) < runtime_dimension_of<0>(*this));
      if constexpr (nested_cols != 1) assert(Eigen::Index(i) < runtime_dimension_of<1>(*this));
      return 0;
    }

  };


  // ------------------------------- //
  //        Deduction guides         //
  // ------------------------------- //

#ifdef __cpp_concepts
  template<zero_matrix Arg>
#else
  template<typename Arg, std::enable_if_t<zero_matrix<Arg>, int> = 0>
#endif
  ZeroMatrix(Arg&&) -> ZeroMatrix<std::conditional_t<eigen_zero_expr<Arg>, pattern_matrix_of_t<Arg>, std::decay_t<Arg>>>;


} // OpenKalman::Eigen3



#endif //OPENKALMAN_EIGEN3_ZEROMATRIX_HPP
