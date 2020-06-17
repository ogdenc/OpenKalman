/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2018-2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef OPENKALMAN_TESTS_TOEUCLIDEANEXPR_H
#define OPENKALMAN_TESTS_TOEUCLIDEANEXPR_H

namespace OpenKalman
{
  ////////////////////////////////////
  //        ToEuclideanExpr         //
  ////////////////////////////////////

  template<
    typename Coefficients, /// Coefficients.
    typename BaseMatrix> /// A nested, non-Euclidean matrix.
  struct ToEuclideanExpr : internal::MatrixBase<ToEuclideanExpr<Coefficients, BaseMatrix>, BaseMatrix>
  {
    static_assert(MatrixTraits<BaseMatrix>::dimension == Coefficients::size);
    using Base = internal::MatrixBase<ToEuclideanExpr, BaseMatrix>;
    using Scalar = typename MatrixTraits<BaseMatrix>::Scalar; ///< Scalar type for this variable.
    static constexpr auto columns = MatrixTraits<BaseMatrix>::columns; ///< Number of columns.

    using Base::Base;

    /// Construct from a compatible to-Euclidean expression.
    template<typename Arg, std::enable_if_t<is_ToEuclideanExpr_v<Arg>, int> = 0>
    ToEuclideanExpr(Arg&& other) noexcept : Base(std::forward<Arg>(other).base_matrix())
    {
      static_assert(OpenKalman::is_equivalent_v<typename MatrixTraits<Arg>::Coefficients, Coefficients>);
      static_assert(MatrixTraits<Arg>::columns == columns);
    }

    /// Construct from compatible matrix object.
    template<typename Arg, std::enable_if_t<is_Eigen_matrix_v<Arg>, int> = 0>
    ToEuclideanExpr(Arg&& arg) noexcept : Base(std::forward<Arg>(arg))
    {
      static_assert(MatrixTraits<Arg>::dimension == Coefficients::size);
      static_assert(MatrixTraits<Arg>::columns == columns);
    }

    /// Construct from a list of coefficients.
    template<typename ... Args, std::enable_if_t<std::conjunction_v<std::is_convertible<Args, const Scalar>...>, int> = 0>
    ToEuclideanExpr(Args ... args) : Base(MatrixTraits<BaseMatrix>::make(args...)) {}

    using Base::operator=;

    /// Assign from a compatible to-Euclidean expression.
    template<typename Arg, std::enable_if_t<is_ToEuclideanExpr_v<Arg>, int> = 0>
    auto& operator=(Arg&& other) noexcept
    {
      static_assert(OpenKalman::is_equivalent_v<typename MatrixTraits<Arg>::Coefficients, Coefficients>);
      static_assert(MatrixTraits<Arg>::columns == columns);
      if constexpr (std::is_same_v<std::decay_t<Arg>, ToEuclideanExpr>) if (this == &other) return *this;
      this->base_matrix() = std::forward<Arg>(other).base_matrix();
      return *this;
    }

    /// Assign from a general Eigen matrix.
    template<typename Arg, std::enable_if_t<is_Eigen_matrix_v<Arg>, int> = 0>
    auto& operator=(Arg&& arg) noexcept
    {
      static_assert(MatrixTraits<Arg>::dimension == Coefficients::dimension);
      static_assert(MatrixTraits<Arg>::columns == columns);
      this->base_matrix() = from_Euclidean<Coefficients>(std::forward<Arg>(arg));
      return *this;
    }

    /// Increment from another expression.
    template<typename Arg>
    auto& operator+=(Arg&& other) noexcept
    {
      static_assert(is_ToEuclideanExpr_v<Arg> or is_Eigen_matrix_v<Arg>);
      static_assert(MatrixTraits<Arg>::columns == MatrixTraits<BaseMatrix>::columns);
      static_assert(MatrixTraits<Arg>::dimension == Coefficients::dimension);
      if constexpr(is_ToEuclideanExpr_v<Arg>) static_assert(OpenKalman::is_equivalent_v<typename MatrixTraits<Arg>::Coefficients, Coefficients>);
      this->base_matrix() = strict(from_Euclidean<Coefficients>(*this + std::forward<Arg>(other)));
      return *this;
    }

    /// Decrement from another expression.
    template<typename Arg>
    auto& operator-=(Arg&& other) noexcept
    {
      static_assert(is_ToEuclideanExpr_v<Arg> or is_Eigen_matrix_v<Arg>);
      static_assert(MatrixTraits<Arg>::columns == MatrixTraits<BaseMatrix>::columns);
      static_assert(MatrixTraits<Arg>::dimension == Coefficients::dimension);
      if constexpr(is_ToEuclideanExpr_v<Arg>) static_assert(OpenKalman::is_equivalent_v<typename MatrixTraits<Arg>::Coefficients, Coefficients>);
      this->base_matrix() = strict(from_Euclidean<Coefficients>(*this - std::forward<Arg>(other)));
      return *this;
    }

    /// Multiply by a scale factor.
    template<typename S, std::enable_if_t<std::is_convertible_v<S, Scalar>, int> = 0>
    auto& operator*=(const S scale)
    {
      this->base_matrix() = strict(from_Euclidean<Coefficients>(*this * scale));
      return *this;
    }

    /// Divide by a scale factor.
    template<typename S, std::enable_if_t<std::is_convertible_v<S, Scalar>, int> = 0>
    auto& operator/=(const S scale)
    {
      this->base_matrix() = strict(from_Euclidean<Coefficients>(*this / scale));
      return *this;
    }

    /// Zero coefficients.
    static auto zero()
    {
      using ST = typename MatrixTraits<BaseMatrix>::template StrictMatrix<Coefficients::dimension>;
      return MatrixTraits<ST>::zero();
    }

    /// Identity.
    static auto identity()
    {
      static_assert(MatrixTraits<BaseMatrix>::dimension == columns, "Identity requires a square matrix.");
      using ST = typename MatrixTraits<BaseMatrix>::template StrictMatrix<Coefficients::dimension>;
      return MatrixTraits<ST>::identity();
    }
  };


  ///////////////////////////
  //        Traits         //
  ///////////////////////////

  /// Define matrix traits.
  template<typename Coeffs, typename ArgType>
  struct MatrixTraits<ToEuclideanExpr<Coeffs, ArgType>>
  {
    using BaseMatrix = ArgType;
    using Coefficients = Coeffs;
    using Scalar = typename MatrixTraits<BaseMatrix>::Scalar;
    static constexpr auto dimension = Coefficients::dimension;
    static constexpr auto columns = MatrixTraits<BaseMatrix>::columns;
    static_assert(Coefficients::size == MatrixTraits<BaseMatrix>::dimension);

    template<typename Derived>
    using MatrixBaseType = internal::EigenMatrixBase<Derived, ToEuclideanExpr<Coeffs, ArgType>>;

    template<std::size_t rows = dimension, std::size_t cols = columns, typename S = Scalar>
    using StrictMatrix = typename MatrixTraits<BaseMatrix>::template StrictMatrix<rows, cols, S>;

    template<TriangleType storage_triangle = TriangleType::lower, std::size_t dim = dimension, typename S = Scalar>
    using SelfAdjointBaseType = EigenSelfAdjointMatrix<StrictMatrix<dim, dim, S>, storage_triangle>;

    template<TriangleType triangle_type = TriangleType::lower, std::size_t dim = dimension, typename S = Scalar>
    using TriangularBaseType = EigenTriangularMatrix<StrictMatrix<dim, dim, S>, triangle_type>;

    template<std::size_t dim = dimension, typename S = Scalar>
    using DiagonalBaseType = EigenDiagonal<StrictMatrix<dim, 1, S>>;

    /// Make from a regular matrix.
    template<typename C = Coefficients, typename Arg,
      std::enable_if_t<is_Eigen_matrix_v<Arg> or is_FromEuclideanExpr_v<Arg>, int> = 0>
    static auto make(Arg&& arg) noexcept
    {
      static_assert(MatrixTraits<Arg>::dimension == C::size);
      return to_Euclidean<C>(std::forward<Arg>(arg));
    }

    template<typename ... Args, std::enable_if_t<std::conjunction_v<std::is_convertible<Args, const Scalar>...>, int> = 0>
    static auto
    make(Args ... args)
    {
      static_assert(sizeof...(Args) == Coefficients::size * columns);
      return make(MatrixTraits<BaseMatrix>::make(args...));
    }

    static auto zero() { return ToEuclideanExpr<Coefficients, BaseMatrix>::zero(); }

    static auto identity() { return ToEuclideanExpr<Coefficients, BaseMatrix>::identity(); }

  };


} // namespace OpenKalman

#endif //OPENKALMAN_TESTS_TOEUCLIDEANEXPR_H
