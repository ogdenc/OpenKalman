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
    (not fixed_coefficients<Coefficients> or Coefficients::euclidean_dimensions == MatrixTraits<NestedMatrix>::rows) and
    (not dynamic_coefficients<Coefficients> or
      std::same_as<typename Coefficients::Scalar, typename MatrixTraits<NestedMatrix>::Scalar>)
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
    static_assert(not fixed_coefficients<Coefficients> or Coefficients::euclidean_dimensions == MatrixTraits<NestedMatrix>::rows);
#endif

    using Scalar = typename MatrixTraits<NestedMatrix>::Scalar;

  private:

    static constexpr auto columns = MatrixTraits<NestedMatrix>::columns; ///< Number of columns.

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
      (MatrixTraits<Arg>::columns == columns) and
      modifiable<NestedMatrix, nested_matrix_t<Arg>>
#else
    template<typename Arg, std::enable_if_t<from_euclidean_expr<Arg> and
      (not std::is_base_of_v<FromEuclideanExpr, std::decay_t<Arg>>) and
      (equivalent_to<typename MatrixTraits<Arg>::RowCoefficients, Coefficients>) and
      (MatrixTraits<Arg>::columns == columns) and
      modifiable<NestedMatrix, nested_matrix_t<Arg>>, int> = 0>
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
      (MatrixTraits<Arg>::rows == Coefficients::dimensions) and (MatrixTraits<Arg>::columns == columns) and
      modifiable<NestedMatrix, decltype(to_euclidean<Coefficients>(std::declval<Arg&&>()))>
#else
    template<typename Arg, std::enable_if_t<eigen_matrix<Arg> and
      (MatrixTraits<Arg>::rows == Coefficients::dimensions) and (MatrixTraits<Arg>::columns == columns) and
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
    template<from_euclidean_expr Arg> requires (MatrixTraits<Arg>::columns == columns) and
      equivalent_to<typename MatrixTraits<Arg>::RowCoefficients, Coefficients>
#else
    template<typename Arg, std::enable_if_t<from_euclidean_expr<Arg> and (MatrixTraits<Arg>::columns == columns) and
      equivalent_to<typename MatrixTraits<Arg>::RowCoefficients, Coefficients>, int> = 0>
#endif
    auto& operator+=(const Arg& arg) noexcept
    {
      this->nested_matrix() = to_euclidean_noalias(*this + arg);
      return *this;
    }


    /// Increment from another \ref eigen_matrix.
#ifdef __cpp_concepts
    template<eigen_matrix Arg> requires (MatrixTraits<Arg>::columns == columns) and
      (MatrixTraits<Arg>::rows == Coefficients::dimensions)
#else
    template<typename Arg, std::enable_if_t<eigen_matrix<Arg> and (MatrixTraits<Arg>::columns == columns) and
      (MatrixTraits<Arg>::rows == Coefficients::dimensions), int> = 0>
#endif
    auto& operator+=(const Arg& arg) noexcept
    {
      this->nested_matrix() = to_euclidean_noalias(*this + arg);
      return *this;
    }


    /// Decrement from another \ref from_euclidean_expr.
#ifdef __cpp_concepts
    template<from_euclidean_expr Arg> requires (MatrixTraits<Arg>::columns == columns) and
      equivalent_to<typename MatrixTraits<Arg>::RowCoefficients, Coefficients>
#else
    template<typename Arg, std::enable_if_t<from_euclidean_expr<Arg> and (MatrixTraits<Arg>::columns == columns) and
      equivalent_to<typename MatrixTraits<Arg>::RowCoefficients, Coefficients>, int> = 0>
#endif
    auto& operator-=(const Arg& arg) noexcept
    {
      this->nested_matrix() = to_euclidean_noalias(*this - arg);
      return *this;
    }


    /// Decrement from another \ref eigen_matrix.
#ifdef __cpp_concepts
    template<eigen_matrix Arg> requires (MatrixTraits<Arg>::columns == columns) and
      (MatrixTraits<Arg>::rows == Coefficients::dimensions)
#else
    template<typename Arg, std::enable_if_t<eigen_matrix<Arg> and (MatrixTraits<Arg>::columns == columns) and
      (MatrixTraits<Arg>::rows == Coefficients::dimensions), int> = 0>
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


namespace OpenKalman
{
  // --------------------- //
  //        Traits         //
  // --------------------- //

  namespace internal
  {
    template<typename Coeffs, typename NestedMatrix>
    struct nested_matrix_type<Eigen3::FromEuclideanExpr<Coeffs, NestedMatrix>>
    {
      using type = NestedMatrix;
    };
  }


  template<typename Coeffs, typename NestedMatrix>
  struct MatrixTraits<Eigen3::FromEuclideanExpr<Coeffs, NestedMatrix>>
  {
    using Scalar = typename MatrixTraits<NestedMatrix>::Scalar;


    static constexpr auto rows = Coeffs::dimensions;
    static constexpr auto columns = MatrixTraits<NestedMatrix>::columns;


    using RowCoefficients = Coeffs;
    using ColumnCoefficients = Axes<columns>;
    static_assert(Coeffs::euclidean_dimensions == MatrixTraits<NestedMatrix>::rows);


    template<typename Derived>
    using MatrixBaseFrom = Eigen3::internal::Eigen3MatrixBase<Derived, Eigen3::FromEuclideanExpr<Coeffs, NestedMatrix>>;


    template<std::size_t r = rows, std::size_t c = columns, typename S = Scalar>
    using NativeMatrixFrom = native_matrix_t<NestedMatrix, r, c, S>;


    using SelfContainedFrom = Eigen3::FromEuclideanExpr<Coeffs, self_contained_t<NestedMatrix>>;


    template<TriangleType storage_triangle = TriangleType::lower, std::size_t dim = rows, typename S = Scalar>
    using SelfAdjointMatrixFrom = Eigen3::SelfAdjointMatrix<NativeMatrixFrom<dim, dim, S>, storage_triangle>;


    template<TriangleType triangle_type = TriangleType::lower, std::size_t dim = rows, typename S = Scalar>
    using TriangularMatrixFrom = Eigen3::TriangularMatrix<NativeMatrixFrom<dim, dim, S>, triangle_type>;


    template<std::size_t dim = rows, typename S = Scalar>
    using DiagonalMatrixFrom = Eigen3::DiagonalMatrix<NativeMatrixFrom<dim, 1, S>>;


    // Make from a regular matrix.
#ifdef __cpp_concepts
    template<typename C = Coeffs, typename Arg> requires
      (Eigen3::eigen_matrix<Arg> or Eigen3::to_euclidean_expr<Arg>) and (MatrixTraits<Arg>::rows == C::euclidean_dimensions)
#else
    template<typename C = Coeffs, typename Arg, std::enable_if_t<
      (Eigen3::eigen_matrix<Arg> or Eigen3::to_euclidean_expr<Arg>) and
      (MatrixTraits<Arg>::rows == C::euclidean_dimensions), int> = 0>
#endif
    static auto make(Arg&& arg) noexcept
    {
      using namespace Eigen3;
      return from_euclidean<C>(std::forward<Arg>(arg));
    }


#ifdef __cpp_concepts
    template<std::convertible_to<Scalar> ... Args> requires (sizeof...(Args) == Coeffs::euclidean_dimensions * columns)
#else
    template<typename ... Args, std::enable_if_t<std::conjunction_v<
      std::is_convertible<Args, Scalar>...> and (sizeof...(Args) == Coeffs::euclidean_dimensions * columns), int> = 0>
#endif
    static auto make(const Args ... args)
    {
      return make(MatrixTraits<NestedMatrix>::make(static_cast<const Scalar>(args)...));
    }


#ifdef __cpp_concepts
    template<std::convertible_to<std::size_t> ... Args> requires (sizeof...(Args) == (rows == 0 ? 1 : 0) + (columns == 0 ? 1 : 0))
#else
    template<typename...Args, std::enable_if_t<(std::is_convertible_v<Args, std::size_t> and ...) and
      (sizeof...(Args) == (rows == 0 ? 1 : 0) + (columns == 0 ? 1 : 0)), int> = 0>
#endif
    static auto zero(const Args...args)
    {
      return MatrixTraits<NativeMatrixFrom<>>::zero(static_cast<std::size_t>(args)...);
    }


#ifdef __cpp_concepts
    template<std::convertible_to<Eigen::Index> ... Args> requires (sizeof...(Args) == (rows == 0 ? 1 : 0))
#else
    template<typename...Args, std::enable_if_t<(std::is_convertible_v<Args, Eigen::Index> and ...) and
      (sizeof...(Args) == (rows == 0 ? 1 : 0)), int> = 0>
#endif
    static auto identity(const Args...args)
    {
      return MatrixTraits<NativeMatrixFrom<>>::identity(args...);
    }

  };

} // namespace OpenKalman




#endif //OPENKALMAN_EIGEN3_FROMEUCLIDEANEXPR_HPP
