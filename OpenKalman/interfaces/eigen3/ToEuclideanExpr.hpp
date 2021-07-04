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

#ifdef __cpp_concepts
  template<coefficients Coefficients, typename NestedMatrix> requires
    eigen_matrix<NestedMatrix> or eigen_diagonal_expr<NestedMatrix>
#else
  template<typename Coefficients, typename NestedMatrix>
#endif
  struct ToEuclideanExpr : OpenKalman::internal::MatrixBase<ToEuclideanExpr<Coefficients, NestedMatrix>, NestedMatrix>
  {

#ifndef __cpp_concepts
    static_assert(coefficients<Coefficients>);
    static_assert(eigen_matrix<NestedMatrix> or eigen_diagonal_expr<NestedMatrix>);
#endif

    static_assert([] {
      if constexpr (dynamic_rows<NestedMatrix>)
        return dynamic_coefficients<Coefficients>;
      else
        return fixed_coefficients<Coefficients> and (MatrixTraits<NestedMatrix>::rows == Coefficients::dimensions);
    }());

  private:

    using Base = OpenKalman::internal::MatrixBase<ToEuclideanExpr, NestedMatrix>;

    static constexpr auto columns = MatrixTraits<NestedMatrix>::columns; ///< Number of columns.

  public:

    using Nested = ToEuclideanExpr; ///< Required by Eigen3.

    using typename Base::Scalar;


    /// Default constructor.
#ifdef __cpp_concepts
    ToEuclideanExpr() requires std::default_initializable<NestedMatrix>
#else
    template<typename T = NestedMatrix, std::enable_if_t<std::is_default_constructible_v<T>, int> = 0>
    ToEuclideanExpr()
#endif
      : Base {} {}


    /// Copy constructor.
    ToEuclideanExpr(const ToEuclideanExpr& other) : Base {other} {}


    /// Move constructor.
    ToEuclideanExpr(ToEuclideanExpr&& other) noexcept: Base {std::move(other)} {}


    /// Construct from a compatible to-Euclidean expression.
#ifdef __cpp_concepts
    template<to_euclidean_expr Arg> requires (not std::derived_from<std::decay_t<Arg>, ToEuclideanExpr>) and
      equivalent_to<typename MatrixTraits<Arg>::RowCoefficients, Coefficients> and
      requires(Arg&& arg) { NestedMatrix {nested_matrix(std::forward<Arg>(arg))}; }
#else
    template<typename Arg, std::enable_if_t<to_euclidean_expr<Arg> and
      (not std::is_base_of_v<ToEuclideanExpr, std::decay_t<Arg>>) and
      equivalent_to<typename MatrixTraits<Arg>::RowCoefficients, Coefficients> and
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


    /// Construct from a list of coefficients.
#ifdef __cpp_concepts
    template<std::convertible_to<const Scalar> ... Args> requires (sizeof...(Args) > 0) and
      requires(Args ... args) { NestedMatrix {MatrixTraits<NestedMatrix>::make(static_cast<const Scalar>(args)...)}; }
#else
    template<typename ... Args, std::enable_if_t<std::conjunction_v<std::is_convertible<Args, const Scalar>...> and
      sizeof...(Args) == columns *
        (from_euclidean_expr<NestedMatrix> ? Coefficients::euclidean_dimensions : Coefficients::dimensions), int> = 0>
#endif
    ToEuclideanExpr(Args ... args)
      : Base {MatrixTraits<NestedMatrix>::make(static_cast<const Scalar>(args)...)} {}


    /// Copy assignment operator.
    auto& operator=(const ToEuclideanExpr& other)
    {
      Base::operator=(other);
      return *this;
    }


    /// Move assignment operator.
    auto& operator=(ToEuclideanExpr&& other) noexcept
    {
      Base::operator=(std::move(other));
      return *this;
    }


    /// Assign from a compatible to-Euclidean expression.
#ifdef __cpp_concepts
    template<to_euclidean_expr Arg> requires (not std::derived_from<std::decay_t<Arg>, ToEuclideanExpr>) and
      (equivalent_to<typename MatrixTraits<Arg>::RowCoefficients, Coefficients>) and
      (MatrixTraits<Arg>::columns == columns) and
      modifiable<NestedMatrix, nested_matrix_t<Arg>>
#else
    template<typename Arg, std::enable_if_t<to_euclidean_expr<Arg> and
      (not std::is_base_of_v<ToEuclideanExpr, std::decay_t<Arg>>) and
      (equivalent_to<typename MatrixTraits<Arg>::RowCoefficients, Coefficients>) and
      (MatrixTraits<Arg>::columns == columns) and
      modifiable<NestedMatrix, nested_matrix_t<Arg>>, int> = 0>
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
    template<eigen_matrix Arg> requires (MatrixTraits<Arg>::rows == Coefficients::euclidean_dimensions) and
      (MatrixTraits<Arg>::columns == columns) and
      modifiable<NestedMatrix, decltype(from_euclidean<Coefficients>(std::declval<Arg>()))>
#else
    template<typename Arg, std::enable_if_t<eigen_matrix<Arg> and
      (MatrixTraits<Arg>::rows == Coefficients::euclidean_dimensions) and (MatrixTraits<Arg>::columns == columns) and
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
    template<to_euclidean_expr Arg> requires (MatrixTraits<Arg>::columns == columns) and
      equivalent_to<typename MatrixTraits<Arg>::RowCoefficients, Coefficients>
#else
    template<typename Arg, std::enable_if_t<to_euclidean_expr<Arg> and (MatrixTraits<Arg>::columns == columns) and
      equivalent_to<typename MatrixTraits<Arg>::RowCoefficients, Coefficients>, int> = 0>
#endif
    auto& operator+=(const Arg& arg) noexcept
    {
      this->nested_matrix() = from_euclidean<Coefficients>(*this + arg);
      return *this;
    }


    /// Increment from another \ref eigen_matrix.
#ifdef __cpp_concepts
    template<eigen_matrix Arg> requires (MatrixTraits<Arg>::columns == columns) and
      (MatrixTraits<Arg>::rows == Coefficients::euclidean_dimensions)
#else
    template<typename Arg, std::enable_if_t<eigen_matrix<Arg> and (MatrixTraits<Arg>::columns == columns) and
      (MatrixTraits<Arg>::rows == Coefficients::euclidean_dimensions), int> = 0>
#endif
    auto& operator+=(const Arg& arg) noexcept
    {
      this->nested_matrix() = from_euclidean<Coefficients>(*this + arg);
      return *this;
    }


    /// Decrement from another \ref to_euclidean_expr.
#ifdef __cpp_concepts
    template<to_euclidean_expr Arg> requires (MatrixTraits<Arg>::columns == columns) and
      equivalent_to<typename MatrixTraits<Arg>::RowCoefficients, Coefficients>
#else
    template<typename Arg, std::enable_if_t<to_euclidean_expr<Arg> and (MatrixTraits<Arg>::columns == columns) and
      equivalent_to<typename MatrixTraits<Arg>::RowCoefficients, Coefficients>, int> = 0>
#endif
    auto& operator-=(const Arg& arg) noexcept
    {
      this->nested_matrix() = from_euclidean<Coefficients>(*this - arg);
      return *this;
    }


    /// Decrement from another \ref eigen_matrix.
#ifdef __cpp_concepts
    template<eigen_matrix Arg> requires (MatrixTraits<Arg>::columns == columns) and
      (MatrixTraits<Arg>::rows == Coefficients::euclidean_dimensions)
#else
    template<typename Arg, std::enable_if_t<eigen_matrix<Arg> and (MatrixTraits<Arg>::columns == columns) and
      (MatrixTraits<Arg>::rows == Coefficients::euclidean_dimensions), int> = 0>
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


    auto operator()(std::size_t i, std::size_t j)
    {
      if constexpr (element_settable < ToEuclideanExpr, 2 >)
        return OpenKalman::internal::ElementAccessor(*this, i, j);
      else
        return std::as_const(*this)(i, j);
    }


    auto operator()(std::size_t i, std::size_t j) const noexcept
    {
      return OpenKalman::internal::ElementAccessor(*this, i, j);
    }


    auto operator[](std::size_t i)
    {
      if constexpr (element_settable < ToEuclideanExpr, 1 >)
        return OpenKalman::internal::ElementAccessor(*this, i);
      else
        return std::as_const(*this)[i];
    }


    auto operator[](std::size_t i) const noexcept { return OpenKalman::internal::ElementAccessor(*this, i); }


    auto operator()(std::size_t i) { return operator[](i); }


    auto operator()(std::size_t i) const { return operator[](i); }
  };

} // OpenKalman::Eigen3


namespace OpenKalman
{
  // --------------------- //
  //        Traits         //
  // --------------------- //

  template<typename Coeffs, typename ArgType>
  struct MatrixTraits<Eigen3::ToEuclideanExpr<Coeffs, ArgType>>
  {
    using NestedMatrix = ArgType;
    using Scalar = typename MatrixTraits<NestedMatrix>::Scalar;


    static constexpr auto rows = Coeffs::euclidean_dimensions;
    static constexpr auto columns = MatrixTraits<NestedMatrix>::columns;


    using RowCoefficients = Coeffs;
    using ColumnCoefficients = Axes<columns>;
    static_assert(Coeffs::dimensions == MatrixTraits<NestedMatrix>::rows);


    template<typename Derived>
    using MatrixBaseFrom = Eigen3::internal::Eigen3MatrixBase<Derived, Eigen3::ToEuclideanExpr<Coeffs, ArgType>>;


    template<std::size_t r = rows, std::size_t c = columns, typename S = Scalar>
    using NativeMatrixFrom = native_matrix_t<NestedMatrix, r, c, S>;


    using SelfContainedFrom = Eigen3::ToEuclideanExpr<Coeffs, self_contained_t<NestedMatrix>>;


    template<TriangleType storage_triangle = TriangleType::lower, std::size_t dim = rows, typename S = Scalar>
    using SelfAdjointMatrixFrom = Eigen3::SelfAdjointMatrix<NativeMatrixFrom<dim, dim, S>, storage_triangle>;


    template<TriangleType triangle_type = TriangleType::lower, std::size_t dim = rows, typename S = Scalar>
    using TriangularMatrixFrom = Eigen3::TriangularMatrix<NativeMatrixFrom<dim, dim, S>, triangle_type>;


    template<std::size_t dim = rows, typename S = Scalar>
    using DiagonalMatrixFrom = Eigen3::DiagonalMatrix<NativeMatrixFrom<dim, 1, S>>;


    // Make from a regular matrix.
#ifdef __cpp_concepts
    template<typename C = Coeffs, typename Arg> requires
      (Eigen3::eigen_matrix<Arg> or Eigen3::from_euclidean_expr<Arg>) and (MatrixTraits<Arg>::rows == C::dimensions)
#else
    template<typename C = Coeffs, typename Arg, std::enable_if_t<
      (Eigen3::eigen_matrix<Arg> or Eigen3::from_euclidean_expr<Arg>) and
      (MatrixTraits<Arg>::rows == C::dimensions), int> = 0>
#endif
    static decltype(auto) make(Arg&& arg) noexcept
    {
      using namespace Eigen3;
      return to_euclidean<C>(make_self_contained<Arg>(std::forward<Arg>(arg)));
    }


#ifdef __cpp_concepts
    template<std::convertible_to<Scalar> ... Args> requires (sizeof...(Args) == Coeffs::dimensions * columns)
#else
    template<typename ... Args, std::enable_if_t<std::conjunction_v<std::is_convertible<Args, Scalar>...> and
      (sizeof...(Args) == Coeffs::dimensions * columns), int> = 0>
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

#endif //OPENKALMAN_EIGEN3_TOEUCLIDEANEXPR_HPP
