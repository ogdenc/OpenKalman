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
 * \brief FromEuclideanExpr and related definitions.
 */

#ifndef OPENKALMAN_EIGEN3_FROMEUCLIDEANEXPR_HPP
#define OPENKALMAN_EIGEN3_FROMEUCLIDEANEXPR_HPP

namespace OpenKalman::Eigen3
{

#ifdef __cpp_concepts
  template<coefficients Coefficients, typename NestedMatrix>
  requires (eigen_matrix<NestedMatrix> or to_euclidean_expr<NestedMatrix> or eigen_diagonal_expr<NestedMatrix>) and
    (dynamic_coefficients<Coefficients> == dynamic_rows<NestedMatrix>) and
    (not fixed_coefficients<Coefficients> or Coefficients::euclidean_dimensions == row_extent_of_v<NestedMatrix>) and
    (not dynamic_coefficients<Coefficients> or
      std::same_as<typename Coefficients::Scalar, scalar_type_of_t<NestedMatrix>>)
#else
  template<typename Coefficients, typename NestedMatrix>
#endif
  struct FromEuclideanExpr : OpenKalman::internal::TypedMatrixBase<
    FromEuclideanExpr<Coefficients, NestedMatrix>, NestedMatrix, Coefficients>
  {

#ifndef __cpp_concepts
    static_assert(coefficients<Coefficients>);
    static_assert(eigen_matrix<NestedMatrix> or to_euclidean_expr<NestedMatrix> or eigen_diagonal_expr<NestedMatrix>);
    static_assert(dynamic_coefficients<Coefficients> == dynamic_rows<NestedMatrix>);
    static_assert(not fixed_coefficients<Coefficients> or Coefficients::euclidean_dimensions == row_extent_of_v<NestedMatrix>);
#endif

    using Scalar = scalar_type_of_t<NestedMatrix>;

  private:

    static constexpr auto columns = column_extent_of_v<NestedMatrix>; ///< Number of columns.

    using Base = OpenKalman::internal::TypedMatrixBase<FromEuclideanExpr, NestedMatrix, Coefficients>;

  public:

    using Base::Base;

    /**
     * Convert from a compatible from-euclidean expression.
     */
#ifdef __cpp_concepts
    template<from_euclidean_expr Arg> requires (not std::derived_from<std::decay_t<Arg>, FromEuclideanExpr>) and
      equivalent_to<typename MatrixTraits<Arg>::RowCoefficients, Coefficients> and
      std::constructible_from<NestedMatrix, decltype(nested_matrix(std::declval<Arg&&>()))>
      //alt: requires(Arg&& arg) { NestedMatrix {nested_matrix(std::forward<Arg>(arg))}; } -- not accepted in GCC 10
#else
    template<typename Arg, std::enable_if_t<from_euclidean_expr<Arg> and
      (not std::is_base_of_v<FromEuclideanExpr, std::decay_t<Arg>>) and
      equivalent_to<typename MatrixTraits<Arg>::RowCoefficients, Coefficients> and
      std::is_constructible_v<NestedMatrix, decltype(nested_matrix(std::declval<Arg&&>()))>, int> = 0>
#endif
    FromEuclideanExpr(Arg&& arg) noexcept : Base {nested_matrix(std::forward<Arg>(arg))} {}


    /**
     * Construct from a compatible to-euclidean expression.
     */
#ifdef __cpp_concepts
    template<to_euclidean_expr Arg> requires
      equivalent_to<typename MatrixTraits<Arg>::RowCoefficients, Coefficients> and
      std::constructible_from<NestedMatrix, Arg&&>
#else
    template<typename Arg, std::enable_if_t<to_euclidean_expr<Arg> and
      equivalent_to<typename MatrixTraits<Arg>::RowCoefficients, Coefficients> and
      std::is_constructible_v<NestedMatrix, Arg&&>, int> = 0>
#endif
    explicit FromEuclideanExpr(Arg&& other) noexcept : Base {std::forward<Arg>(other)} {}


    /**
     * Construct from compatible matrix object.
     */
#ifdef __cpp_concepts
    template<eigen_matrix Arg> requires std::constructible_from<NestedMatrix, Arg&&>
#else
    template<typename Arg, std::enable_if_t<
      eigen_matrix<Arg> and std::is_constructible_v<NestedMatrix, Arg&&>, int> = 0>
#endif
    explicit FromEuclideanExpr(Arg&& arg) noexcept : Base {std::forward<Arg>(arg)} {}


#ifndef __cpp_concepts
    /**
     * /brief Construct from a list of coefficients.
     * /note If c++ concepts are available, this functionality is inherited from the base class.
     */
    template<typename ... Args, std::enable_if_t<std::conjunction_v<std::is_convertible<Args, const Scalar>...> and
      sizeof...(Args) == columns *
        (to_euclidean_expr<NestedMatrix> ? Coefficients::dimensions : Coefficients::euclidean_dimensions), int> = 0>
    FromEuclideanExpr(Args ... args) : Base {MatrixTraits<NestedMatrix>::make(static_cast<const Scalar>(args)...)} {}
#endif


    /**
     * Assign from a compatible from-Euclidean expression.
     */
#ifdef __cpp_concepts
    template<from_euclidean_expr Arg> requires (not std::derived_from<std::decay_t<Arg>, FromEuclideanExpr>) and
      (equivalent_to<typename MatrixTraits<Arg>::RowCoefficients, Coefficients>) and
      (column_extent_of_v<Arg> == columns) and
      modifiable<NestedMatrix, nested_matrix_of<Arg>>
#else
    template<typename Arg, std::enable_if_t<from_euclidean_expr<Arg> and
      (not std::is_base_of_v<FromEuclideanExpr, std::decay_t<Arg>>) and
      (equivalent_to<typename MatrixTraits<Arg>::RowCoefficients, Coefficients>) and
      (column_extent_of<Arg>::value == columns) and
      modifiable<NestedMatrix, nested_matrix_of<Arg>>, int> = 0>
#endif
    auto& operator=(Arg&& arg) noexcept
    {
      if constexpr (not zero_matrix<NestedMatrix> and not identity_matrix<NestedMatrix>)
      {
        this->nested_matrix() = nested_matrix(std::forward<Arg>(arg));
      }
      return *this;
    }


    /**
     * Assign from a general Eigen matrix.
     */
#ifdef __cpp_concepts
    template<eigen_matrix Arg> requires
      (row_extent_of_v<Arg> == Coefficients::dimensions) and (column_extent_of_v<Arg> == columns) and
      modifiable<NestedMatrix, decltype(to_euclidean<Coefficients>(std::declval<Arg&&>()))>
#else
    template<typename Arg, std::enable_if_t<eigen_matrix<Arg> and
      (row_extent_of<Arg>::value == Coefficients::dimensions) and (column_extent_of<Arg>::value == columns) and
      modifiable<NestedMatrix, decltype(to_euclidean<Coefficients>(std::declval<Arg&&>()))>, int> = 0>
#endif
    auto& operator=(Arg&& arg) noexcept
    {
      if constexpr (not zero_matrix<NestedMatrix> and not identity_matrix<NestedMatrix>)
      {
        this->nested_matrix() = to_euclidean<Coefficients>(std::forward<Arg>(arg));
      }
      return *this;
    }

  private:

    template<typename Arg>
    static auto to_euclidean_noalias(Arg&& arg)
    {
      if constexpr (Coefficients::euclidean_dimensions > Coefficients::dimensions)
        return make_native_matrix(to_euclidean<Coefficients>(std::forward<Arg>(arg))); //< Prevent aliasing
      else
        return to_euclidean<Coefficients>(make_self_contained<Arg>(std::forward<Arg>(arg)));
    }

  public:

    /// Increment from another \ref from_euclidean_expr.
#ifdef __cpp_concepts
    template<from_euclidean_expr Arg> requires (column_extent_of_v<Arg> == columns) and
      equivalent_to<typename MatrixTraits<Arg>::RowCoefficients, Coefficients>
#else
    template<typename Arg, std::enable_if_t<from_euclidean_expr<Arg> and (column_extent_of<Arg>::value == columns) and
      equivalent_to<typename MatrixTraits<Arg>::RowCoefficients, Coefficients>, int> = 0>
#endif
    auto& operator+=(const Arg& arg) noexcept
    {
      this->nested_matrix() = to_euclidean_noalias(*this + arg);
      return *this;
    }


    /// Increment from another \ref eigen_matrix.
#ifdef __cpp_concepts
    template<eigen_matrix Arg> requires (column_extent_of_v<Arg> == columns) and
      (row_extent_of_v<Arg> == Coefficients::dimensions)
#else
    template<typename Arg, std::enable_if_t<eigen_matrix<Arg> and (column_extent_of<Arg>::value == columns) and
      (row_extent_of<Arg>::value == Coefficients::dimensions), int> = 0>
#endif
    auto& operator+=(const Arg& arg) noexcept
    {
      this->nested_matrix() = to_euclidean_noalias(*this + arg);
      return *this;
    }


    /// Decrement from another \ref from_euclidean_expr.
#ifdef __cpp_concepts
    template<from_euclidean_expr Arg> requires (column_extent_of_v<Arg> == columns) and
      equivalent_to<typename MatrixTraits<Arg>::RowCoefficients, Coefficients>
#else
    template<typename Arg, std::enable_if_t<from_euclidean_expr<Arg> and (column_extent_of<Arg>::value == columns) and
      equivalent_to<typename MatrixTraits<Arg>::RowCoefficients, Coefficients>, int> = 0>
#endif
    auto& operator-=(const Arg& arg) noexcept
    {
      this->nested_matrix() = to_euclidean_noalias(*this - arg);
      return *this;
    }


    /// Decrement from another \ref eigen_matrix.
#ifdef __cpp_concepts
    template<eigen_matrix Arg> requires (column_extent_of_v<Arg> == columns) and
      (row_extent_of_v<Arg> == Coefficients::dimensions)
#else
    template<typename Arg, std::enable_if_t<eigen_matrix<Arg> and (column_extent_of<Arg>::value == columns) and
      (row_extent_of<Arg>::value == Coefficients::dimensions), int> = 0>
#endif
    auto& operator-=(const Arg& arg) noexcept
    {
      this->nested_matrix() = to_euclidean_noalias(*this - arg);
      return *this;
    }


    /**
     * Multiply by a scale factor.
     * \param scale The scale factor.
     */
#ifdef __cpp_concepts
    template<std::convertible_to<Scalar> S>
#else
    template<typename S, std::enable_if_t<std::is_convertible_v<S, Scalar>, int> = 0>
#endif
    auto& operator*=(const S scale)
    {
      this->nested_matrix() = to_euclidean_noalias(*this * scale);
      return *this;
    }


    /**
     * Divide by a scale factor.
     * \param scale The scale factor.
     */
#ifdef __cpp_concepts
    template<std::convertible_to<Scalar> S>
#else
    template<typename S, std::enable_if_t<std::is_convertible_v<S, Scalar>, int> = 0>
#endif
    auto& operator/=(const S scale)
    {
      this->nested_matrix() = to_euclidean_noalias(*this / scale);
      return *this;
    }

  };

} // OpenKalman::Eigen3


#endif //OPENKALMAN_EIGEN3_FROMEUCLIDEANEXPR_HPP
