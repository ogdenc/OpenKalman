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
# if __cpp_nontype_template_args >= 201911L
  template<indexible PatternMatrix, scalar_type_of_t<PatternMatrix> constant>
# else
  template<indexible PatternMatrix, auto constant> requires
    std::convertible_to<decltype(constant), scalar_type_of_t<PatternMatrix>>
# endif
#else
  template<typename PatternMatrix, auto constant>
#endif
  struct ConstantMatrix : Eigen3::internal::EigenDynamicBase<ConstantMatrix<PatternMatrix, constant>, PatternMatrix>
  {

  private:

    using nested_scalar = scalar_type_of_t<PatternMatrix>;
    static constexpr auto nested_rows = index_dimension_of_v<PatternMatrix, 0>;
    static constexpr auto nested_cols = index_dimension_of_v<PatternMatrix, 1>;
    using Base = Eigen3::internal::EigenDynamicBase<ConstantMatrix, PatternMatrix>;

#ifndef __cpp_concepts
    static_assert(std::is_convertible_v<decltype(constant), nested_scalar>);
#endif

  public:

    /**
     * \brief Construct a ConstantMatrix.
     * \details The constructor can take a number of arguments representing the number of dynamic dimensions.
     * For example, ConstantMatrix {2, 3} constructs a 2-by-3 dynamic matrix, ConstantMatrix {3} constructs a
     * 2-by-3 matrix in which there are two fixed row dimensions and three dynamic column dimensions, and
     * ConstantMatrix {} constructs a fixed matrix.
     */
#ifdef __cpp_concepts
    template<std::convertible_to<std::size_t> ... Args> requires
    (sizeof...(Args) == (nested_rows == dynamic_size ? 1 : 0) + (nested_cols == dynamic_size ? 1 : 0))
#else
    template<typename...Args, std::enable_if_t<(std::is_convertible_v<Args, std::size_t> and ...) and
      (sizeof...(Args) == (nested_rows == dynamic_size ? 1 : 0) + (nested_cols == dynamic_size ? 1 : 0)), int> = 0>
#endif
    ConstantMatrix(const Args...args) : Base {static_cast<std::size_t>(args)...} {}

#ifndef __cpp_concepts
  private:

    template<typename T, typename = void>
    struct constant_arg_matches : std::false_type {};

    template<typename T>
    struct constant_arg_matches<T, std::enable_if_t<(constant_coefficient<std::decay_t<T>>::value == constant) and
      (dynamic_rows<T> or nested_rows == dynamic_size or row_dimension_of<std::decay_t<T>>::value == nested_rows) and
      (dynamic_columns<T> or nested_cols == dynamic_size or column_dimension_of<std::decay_t<T>>::value == nested_cols)>>
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
      (dynamic_rows<M> or nested_rows == dynamic_size or row_dimension_of_v<M> == nested_rows) and
      (dynamic_columns<M> or nested_cols == dynamic_size or column_dimension_of_v<M> == nested_cols)
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
      (dynamic_rows<M> or nested_rows == dynamic_size or row_dimension_of_v<M> == nested_rows) and
      (dynamic_columns<M> or nested_cols == dynamic_size or column_dimension_of_v<M> == nested_cols)
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
    constexpr nested_scalar
    operator()(std::size_t r, std::size_t c) const
    {
      assert(Eigen::Index(r) < runtime_dimension_of<0>(*this));
      assert(Eigen::Index(c) < runtime_dimension_of<1>(*this));
      return constant;
    }


    /**
     * \brief Element accessor for a row or column vector.
     * \param i The row or column.
     * \return The element at row or column i (always the constant).
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
      return constant;
    }


    /**
     * \brief Element accessor for a row or column vector.
     * \param i The row or column.
     * \return The element at row or column i (always the constant).
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
      return constant;
    }

  };


  // ------------------------------- //
  //        Deduction guides         //
  // ------------------------------- //

  template<typename NestedMatrix, auto constant>
  ConstantMatrix(ConstantMatrix<NestedMatrix, constant>&) -> ConstantMatrix<NestedMatrix, constant>;

  template<typename NestedMatrix, auto constant>
  ConstantMatrix(const ConstantMatrix<NestedMatrix, constant>&) -> ConstantMatrix<NestedMatrix, constant>;

  template<typename NestedMatrix, auto constant>
  ConstantMatrix(ConstantMatrix<NestedMatrix, constant>&&) -> ConstantMatrix<NestedMatrix, constant>;

  template<typename NestedMatrix, auto constant>
  ConstantMatrix(const ConstantMatrix<NestedMatrix, constant>&&) -> ConstantMatrix<NestedMatrix, constant>;


  #ifdef __cpp_concepts
#  if __cpp_nontype_template_args >= 201911L
  template<constant_matrix Arg> requires (not eigen_constant_expr<Arg>)
  ConstantMatrix(Arg&&) -> ConstantMatrix<std::decay_t<Arg>, constant_coefficient_v<Arg>>;
#  else
  template<constant_matrix Arg> requires (not eigen_constant_expr<Arg>) and std::is_integral_v<scalar_type_of_t<Arg>>
  ConstantMatrix(Arg&&) -> ConstantMatrix<std::decay_t<Arg>, constant_coefficient_v<Arg>>;

  template<constant_matrix Arg> requires (not eigen_constant_expr<Arg>) and
    (not std::is_integral_v<scalar_type_of_t<Arg>>) and
    (constant_coefficient_v<Arg> == static_cast<std::intmax_t>(constant_coefficient_v<Arg>))
  ConstantMatrix(Arg&&) -> ConstantMatrix<std::decay_t<Arg>, static_cast<std::intmax_t>(constant_coefficient_v<Arg>)>;
#  endif
#else
  template<typename Arg, std::enable_if_t<constant_matrix<Arg> and not eigen_constant_expr<Arg> and
    constant_coefficient_v<Arg> == static_cast<std::intmax_t>(constant_coefficient_v<Arg>), int> = 0>
  ConstantMatrix(Arg&&) -> ConstantMatrix<std::decay_t<Arg>, static_cast<std::intmax_t>(constant_coefficient_v<Arg>)>;
#endif


} // namespace OpenKalman::Eigen3



#endif //OPENKALMAN_EIGEN3_CONSTANTMATRIX_HPP
