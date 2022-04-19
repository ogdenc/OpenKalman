/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2018-2021 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief ToEuclideanExpr and related definitions.
 */

#ifndef OPENKALMAN_EIGEN3_TOEUCLIDEANEXPR_HPP
#define OPENKALMAN_EIGEN3_TOEUCLIDEANEXPR_HPP

namespace OpenKalman::Eigen3
{

  /// \todo Remove nested diagonal matrix option
#ifdef __cpp_concepts
  template<typed_index_descriptor Coefficients, typename NestedMatrix> requires (not from_euclidean_expr<NestedMatrix>) and
    (dynamic_coefficients<Coefficients> == dynamic_rows<NestedMatrix>) and
    (not fixed_coefficients<Coefficients> or dimension_size_of_v<Coefficients> == row_dimension_of_v<NestedMatrix>) and
    (not dynamic_coefficients<Coefficients> or
      std::same_as<typename Coefficients::Scalar, scalar_type_of_t<NestedMatrix>>)
#else
  template<typename Coefficients, typename NestedMatrix>
#endif
  struct ToEuclideanExpr : OpenKalman::internal::TypedMatrixBase<
    ToEuclideanExpr<Coefficients, NestedMatrix>, NestedMatrix, Coefficients>
  {

#ifndef __cpp_concepts
    static_assert(typed_index_descriptor<Coefficients>);
    static_assert(not from_euclidean_expr<NestedMatrix>);
    static_assert(dynamic_coefficients<Coefficients> == dynamic_rows<NestedMatrix>);
    static_assert(not fixed_coefficients<Coefficients> or dimension_size_of_v<Coefficients> == row_dimension_of_v<NestedMatrix>);
#endif

    using Scalar = scalar_type_of_t<NestedMatrix>;

  private:

    static constexpr auto columns = column_dimension_of_v<NestedMatrix>; ///< Number of columns.

    using Base = OpenKalman::internal::TypedMatrixBase<ToEuclideanExpr, NestedMatrix, Coefficients>;

  public:

    using Base::Base;

    /// Construct from a compatible to-Euclidean expression.
#ifdef __cpp_concepts
    template<to_euclidean_expr Arg> requires (not std::derived_from<std::decay_t<Arg>, ToEuclideanExpr>) and
      equivalent_to<row_coefficient_types_of_t<Arg>, Coefficients> and
      std::constructible_from<NestedMatrix, decltype(nested_matrix(std::declval<Arg&&>()))>
      //alt: requires(Arg&& arg) { NestedMatrix {nested_matrix(std::forward<Arg>(arg))}; } -- not accepted in GCC 10
#else
    template<typename Arg, std::enable_if_t<to_euclidean_expr<Arg> and
      (not std::is_base_of_v<ToEuclideanExpr, std::decay_t<Arg>>) and
      equivalent_to<row_coefficient_types_of_t<Arg>, Coefficients> and
      std::is_constructible_v<NestedMatrix, decltype(nested_matrix(std::declval<Arg&&>()))>, int> = 0>
#endif
    ToEuclideanExpr(Arg&& arg) noexcept : Base {nested_matrix(std::forward<Arg>(arg))} {}


    /// Construct from compatible matrix object.
#ifdef __cpp_concepts
    template<eigen_matrix Arg> requires std::constructible_from<NestedMatrix, Arg&&>
#else
    template<typename Arg, std::enable_if_t<
      eigen_matrix<Arg> and std::is_constructible_v<NestedMatrix, Arg&&>, int> = 0>
#endif
    explicit ToEuclideanExpr(Arg&& arg) noexcept : Base {std::forward<Arg>(arg)} {}


#ifndef __cpp_concepts
    /**
     * /brief Construct from a list of coefficients.
     * /note If c++ concepts are available, this functionality is inherited from the base class.
     */
    template<typename ... Args, std::enable_if_t<std::conjunction_v<std::is_convertible<Args, const Scalar>...> and
      (sizeof...(Args) == columns * dimension_size_of_v<Coefficients>), int> = 0>
    ToEuclideanExpr(Args ... args) : Base {MatrixTraits<NestedMatrix>::make(static_cast<const Scalar>(args)...)} {}
#endif


    /// Assign from a compatible to-Euclidean expression.
#ifdef __cpp_concepts
    template<to_euclidean_expr Arg> requires (not std::derived_from<std::decay_t<Arg>, ToEuclideanExpr>) and
      (equivalent_to<row_coefficient_types_of_t<Arg>, Coefficients>) and
      (column_dimension_of_v<Arg> == columns) and
      modifiable<NestedMatrix, nested_matrix_of_t<Arg>>
#else
    template<typename Arg, std::enable_if_t<to_euclidean_expr<Arg> and
      (not std::is_base_of_v<ToEuclideanExpr, std::decay_t<Arg>>) and
      (equivalent_to<row_coefficient_types_of_t<Arg>, Coefficients>) and
      (column_dimension_of<Arg>::value == columns) and
      modifiable<NestedMatrix, nested_matrix_of_t<Arg>>, int> = 0>
#endif
    auto& operator=(Arg&& other) noexcept
    {
      if constexpr (not zero_matrix<NestedMatrix> and not identity_matrix<NestedMatrix>)
      {
        this->nested_matrix() = nested_matrix(std::forward<Arg>(other));
      }
      return *this;
    }


    /// Assign from a general Eigen matrix.
#ifdef __cpp_concepts
    template<eigen_matrix Arg> requires (row_dimension_of_v<Arg> == euclidean_dimension_size_of_v<Coefficients>) and
      (column_dimension_of_v<Arg> == columns) and
      modifiable<NestedMatrix, decltype(from_euclidean<Coefficients>(std::declval<Arg>()))>
#else
    template<typename Arg, std::enable_if_t<eigen_matrix<Arg> and
      (row_dimension_of<Arg>::value == euclidean_dimension_size_of_v<Coefficients>) and (column_dimension_of<Arg>::value == columns) and
      modifiable<NestedMatrix, decltype(from_euclidean<Coefficients>(std::declval<Arg>()))>, int> = 0>
#endif
    auto& operator=(Arg&& arg) noexcept
    {
      if constexpr (not zero_matrix<NestedMatrix> and not identity_matrix<NestedMatrix>)
      {
        this->nested_matrix() = from_euclidean<Coefficients>(std::forward<Arg>(arg));
      }
      return *this;
    }


    /// Increment from another \ref to_euclidean_expr.
#ifdef __cpp_concepts
    template<to_euclidean_expr Arg> requires (column_dimension_of_v<Arg> == columns) and
      equivalent_to<row_coefficient_types_of_t<Arg>, Coefficients>
#else
    template<typename Arg, std::enable_if_t<to_euclidean_expr<Arg> and (column_dimension_of<Arg>::value == columns) and
      equivalent_to<row_coefficient_types_of_t<Arg>, Coefficients>, int> = 0>
#endif
    auto& operator+=(const Arg& arg) noexcept
    {
      this->nested_matrix() = from_euclidean<Coefficients>(*this + arg);
      return *this;
    }


    /// Increment from another \ref eigen_matrix.
#ifdef __cpp_concepts
    template<eigen_matrix Arg> requires (column_dimension_of_v<Arg> == columns) and
      (row_dimension_of_v<Arg> == euclidean_dimension_size_of_v<Coefficients>)
#else
    template<typename Arg, std::enable_if_t<eigen_matrix<Arg> and (column_dimension_of<Arg>::value == columns) and
      (row_dimension_of<Arg>::value == euclidean_dimension_size_of_v<Coefficients>), int> = 0>
#endif
    auto& operator+=(const Arg& arg) noexcept
    {
      this->nested_matrix() = from_euclidean<Coefficients>(*this + arg);
      return *this;
    }


    /// Decrement from another \ref to_euclidean_expr.
#ifdef __cpp_concepts
    template<to_euclidean_expr Arg> requires (column_dimension_of_v<Arg> == columns) and
      equivalent_to<row_coefficient_types_of_t<Arg>, Coefficients>
#else
    template<typename Arg, std::enable_if_t<to_euclidean_expr<Arg> and (column_dimension_of<Arg>::value == columns) and
      equivalent_to<row_coefficient_types_of_t<Arg>, Coefficients>, int> = 0>
#endif
    auto& operator-=(const Arg& arg) noexcept
    {
      this->nested_matrix() = from_euclidean<Coefficients>(*this - arg);
      return *this;
    }


    /// Decrement from another \ref eigen_matrix.
#ifdef __cpp_concepts
    template<eigen_matrix Arg> requires (column_dimension_of_v<Arg> == columns) and
      (row_dimension_of_v<Arg> == euclidean_dimension_size_of_v<Coefficients>)
#else
    template<typename Arg, std::enable_if_t<eigen_matrix<Arg> and (column_dimension_of<Arg>::value == columns) and
      (row_dimension_of<Arg>::value == euclidean_dimension_size_of_v<Coefficients>), int> = 0>
#endif
    auto& operator-=(const Arg& arg) noexcept
    {
      this->nested_matrix() = from_euclidean<Coefficients>(*this - arg);
      return *this;
    }


    /// Multiply by a scale factor.
    template<typename S, std::enable_if_t<std::is_convertible_v<S, Scalar>, int> = 0>
    auto& operator*=(const S scale)
    {
      this->nested_matrix() = from_euclidean<Coefficients>(*this * scale);
      return *this;
    }


    /// Divide by a scale factor.
#ifdef __cpp_concepts
    template<std::convertible_to<Scalar> S>
#else
    template<typename S, std::enable_if_t<std::is_convertible_v<S, Scalar>, int> = 0>
#endif
    auto& operator/=(const S scale)
    {
      this->nested_matrix() = from_euclidean<Coefficients>(*this / scale);
      return *this;
    }

  };

} // OpenKalman::Eigen3



#endif //OPENKALMAN_EIGEN3_TOEUCLIDEANEXPR_HPP
