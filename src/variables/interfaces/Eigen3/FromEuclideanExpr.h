/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2018-2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef OPENKALMAN_TESTS_FROMEUCLIDEANEXPR_H
#define OPENKALMAN_TESTS_FROMEUCLIDEANEXPR_H

namespace OpenKalman
{
  //////////////////////////////////////
  //        FromEuclideanExpr         //
  //////////////////////////////////////

  template<
    typename Coefficients,
    typename BaseMatrix>
  struct FromEuclideanExpr : internal::MatrixBase<FromEuclideanExpr<Coefficients, BaseMatrix>, BaseMatrix>
  {
    static_assert(MatrixTraits<BaseMatrix>::dimension == Coefficients::dimension);
    using Base = internal::MatrixBase<FromEuclideanExpr, BaseMatrix>;
    using Scalar = typename MatrixTraits<BaseMatrix>::Scalar; ///< Scalar type for this variable.
    static constexpr auto columns = MatrixTraits<BaseMatrix>::columns; ///< Number of columns.

    /// Default constructor.
    FromEuclideanExpr() : Base() {}

    /// Copy constructor.
    FromEuclideanExpr(const FromEuclideanExpr& other) : Base(other.base_matrix()) {}

    /// Move constructor.
    FromEuclideanExpr(FromEuclideanExpr&& other) noexcept : Base(std::move(other).base_matrix()) {}

    /// Convert from a compatible from-euclidean expression.
    template<typename Arg, std::enable_if_t<is_FromEuclideanExpr_v<Arg>, int> = 0>
    FromEuclideanExpr(Arg&& other) noexcept : Base(std::forward<Arg>(other).base_matrix())
    {
      static_assert(OpenKalman::is_equivalent_v<typename MatrixTraits<Arg>::Coefficients, Coefficients>);
      static_assert(MatrixTraits<Arg>::columns == columns);
    }

    /// Construct from a compatible to-euclidean expression.
    template<typename Arg, std::enable_if_t<is_ToEuclideanExpr_v<Arg>, int> = 0>
    FromEuclideanExpr(Arg&& other) noexcept : Base(std::forward<Arg>(other))
    {
      static_assert(OpenKalman::is_equivalent_v<typename MatrixTraits<Arg>::Coefficients, Coefficients>);
      static_assert(MatrixTraits<Arg>::columns == columns);
    }

    /// Construct from compatible matrix object.
    template<typename Arg, std::enable_if_t<is_Eigen_matrix_v<Arg>, int> = 0>
    FromEuclideanExpr(Arg&& arg) noexcept : Base(std::forward<Arg>(arg))
    {
      static_assert(MatrixTraits<Arg>::dimension == Coefficients::dimension);
      static_assert(MatrixTraits<Arg>::columns == columns);
    }

    /// Constructor that fills vector with the values of the arguments.
    template<typename ... Args, std::enable_if_t<std::conjunction_v<std::is_convertible<Args, const Scalar>...> and
      sizeof...(Args) == columns *
        (is_ToEuclideanExpr_v<BaseMatrix> ? Coefficients::size : Coefficients::dimension), int> = 0>
    FromEuclideanExpr(Args ... args) : Base(MatrixTraits<BaseMatrix>::make(args...)) {}

    /// Copy assignment operator.
    auto& operator=(const FromEuclideanExpr& other)
    {
      if constexpr (not is_zero_v<BaseMatrix> and not is_identity_v<BaseMatrix>) if (this != &other)
        this->base_matrix() = other.base_matrix();
      return *this;
    }

    /// Move assignment operator.
    auto& operator=(FromEuclideanExpr&& other) noexcept
    {
      if constexpr (not is_zero_v<BaseMatrix> and not is_identity_v<BaseMatrix>) if (this != &other)
        this->base_matrix() = std::move(other).base_matrix();
      return *this;
    }

    /// Assign from a compatible from-Euclidean expression.
    template<typename Arg, std::enable_if_t<is_FromEuclideanExpr_v<Arg>, int> = 0>
    auto& operator=(Arg&& other) noexcept
    {
      static_assert(is_equivalent_v<typename MatrixTraits<Arg>::Coefficients, Coefficients>);
      static_assert(MatrixTraits<Arg>::columns == columns);
      if constexpr (is_zero_v<BaseMatrix>)
      {
        static_assert(is_zero_v<Arg>);
      }
      else if constexpr (is_identity_v<BaseMatrix>)
      {
        static_assert(is_identity_v<Arg>);
      }
      else
      {
        this->base_matrix() = std::forward<Arg>(other).base_matrix();
      }
      return *this;
    }

    /// Assign from a general Eigen matrix.
    template<typename Arg, std::enable_if_t<is_Eigen_matrix_v<Arg>, int> = 0>
    auto& operator=(Arg&& arg) noexcept
    {
      static_assert(MatrixTraits<Arg>::dimension == Coefficients::size);
      static_assert(MatrixTraits<Arg>::columns == columns);
      if constexpr (is_zero_v<BaseMatrix>)
      {
        static_assert(is_zero_v<Arg>);
      }
      else if constexpr (is_identity_v<BaseMatrix>)
      {
        static_assert(is_identity_v<Arg>);
      }
      else
      {
        this->base_matrix() = to_Euclidean<Coefficients>(std::forward<Arg>(arg));
      }
      return *this;
    }

    /// Increment from another expression.
    template<typename Arg>
    auto& operator+=(Arg&& other) noexcept
    {
      static_assert(is_FromEuclideanExpr_v<Arg> or is_Eigen_matrix_v<Arg>);
      static_assert(MatrixTraits<Arg>::columns == MatrixTraits<BaseMatrix>::columns);
      static_assert(MatrixTraits<Arg>::dimension == Coefficients::size);
      if constexpr(is_FromEuclideanExpr_v<Arg>) static_assert(is_equivalent_v<typename MatrixTraits<Arg>::Coefficients, Coefficients>);
      this->base_matrix() = strict(to_Euclidean<Coefficients>(*this + std::forward<Arg>(other)));
      return *this;
    }

    /// Decrement from another expression.
    template<typename Arg>
    auto& operator-=(Arg&& other) noexcept
    {
      static_assert(is_FromEuclideanExpr_v<Arg> or is_Eigen_matrix_v<Arg>);
      static_assert(MatrixTraits<Arg>::columns == MatrixTraits<BaseMatrix>::columns);
      static_assert(MatrixTraits<Arg>::dimension == Coefficients::size);
      if constexpr(is_FromEuclideanExpr_v<Arg>) static_assert(is_equivalent_v<typename MatrixTraits<Arg>::Coefficients, Coefficients>);
      this->base_matrix() = strict(to_Euclidean<Coefficients>(*this - std::forward<Arg>(other)));
      return *this;
    }

    /// Multiply by a scale factor.
    template<typename S, std::enable_if_t<std::is_convertible_v<S, Scalar>, int> = 0>
    auto& operator*=(const S scale)
    {
      this->base_matrix() = strict(to_Euclidean<Coefficients>(*this * scale));
      return *this;
    }

    /// Divide by a scale factor.
    template<typename S, std::enable_if_t<std::is_convertible_v<S, Scalar>, int> = 0>
    auto& operator/=(const S scale)
    {
      this->base_matrix() = strict(to_Euclidean<Coefficients>(*this / scale));
      return *this;
    }

    /// Zero coefficients.
    static auto zero()
    {
      using ST = strict_matrix_t<BaseMatrix, Coefficients::size>;
      return MatrixTraits<ST>::zero();
    }

    /// Identity.
    static auto identity()
    {
      static_assert(MatrixTraits<BaseMatrix>::dimension == columns, "Identity requires a square matrix.");
      using ST = strict_matrix_t<BaseMatrix, Coefficients::size>;
      return MatrixTraits<ST>::identity();
    }

    auto operator()(std::size_t i, std::size_t j)
    {
      if constexpr (is_element_settable_v<FromEuclideanExpr, 2>)
        return internal::ElementSetter(*this, i, j);
      else
        return const_cast<const FromEuclideanExpr&>(*this)(i, j);
    }

    auto operator()(std::size_t i, std::size_t j) const noexcept { return internal::ElementSetter(*this, i, j); }

    auto operator[](std::size_t i)
    {
      if constexpr (is_element_settable_v<FromEuclideanExpr, 1>)
        return internal::ElementSetter(*this, i);
      else
        return const_cast<const FromEuclideanExpr&>(*this)[i];
    }

    auto operator[](std::size_t i) const noexcept { return internal::ElementSetter(*this, i); }

    auto operator()(std::size_t i) { return operator[](i); }

    auto operator()(std::size_t i) const { return operator[](i); }
  };


  ///////////////////////////
  //        Traits         //
  ///////////////////////////

  /// Define matrix traits.
  template<typename Coeffs, typename ArgType>
  struct MatrixTraits<FromEuclideanExpr<Coeffs, ArgType>>
  {
    using BaseMatrix = ArgType;
    using Coefficients = Coeffs;
    using Scalar = typename MatrixTraits<BaseMatrix>::Scalar;
    static constexpr auto dimension = Coefficients::size;
    static constexpr auto columns = MatrixTraits<BaseMatrix>::columns;
    static_assert(Coefficients::dimension == MatrixTraits<BaseMatrix>::dimension);

    template<typename Derived>
    using MatrixBaseType = internal::EigenMatrixBase<Derived, FromEuclideanExpr<Coeffs, ArgType>>;

    template<std::size_t rows = dimension, std::size_t cols = columns, typename S = Scalar>
    using StrictMatrix = typename MatrixTraits<BaseMatrix>::template StrictMatrix<rows, cols, S>;

    using Strict = FromEuclideanExpr<Coefficients, typename MatrixTraits<BaseMatrix>::Strict>;

    template<TriangleType storage_triangle = TriangleType::lower, std::size_t dim = dimension, typename S = Scalar>
    using SelfAdjointBaseType = EigenSelfAdjointMatrix<StrictMatrix<dim, dim, S>, storage_triangle>;

    template<TriangleType triangle_type = TriangleType::lower, std::size_t dim = dimension, typename S = Scalar>
    using TriangularBaseType = EigenTriangularMatrix<StrictMatrix<dim, dim, S>, triangle_type>;

    template<std::size_t dim = dimension, typename S = Scalar>
    using DiagonalBaseType = EigenDiagonal<StrictMatrix<dim, 1, S>>;

    /// Make from a regular matrix.
    template<typename C = Coefficients, typename Arg,
      std::enable_if_t<is_Eigen_matrix_v<Arg> or is_ToEuclideanExpr_v<Arg>, int> = 0>
    static auto make(Arg&& arg) noexcept
    {
      static_assert(MatrixTraits<Arg>::dimension == C::dimension);
      return from_Euclidean<C>(std::forward<Arg>(arg));
    }

    template<typename ... Args, std::enable_if_t<std::conjunction_v<std::is_convertible<Args, const Scalar>...>, int> = 0>
    static auto
    make(Args ... args)
    {
      static_assert(sizeof...(Args) == Coefficients::dimension * columns);
      return make(MatrixTraits<BaseMatrix>::make(args...));
    }

    static auto zero() { return FromEuclideanExpr<Coefficients, BaseMatrix>::zero(); }

    static auto identity() { return FromEuclideanExpr<Coefficients, BaseMatrix>::identity(); }

  };

} // namespace OpenKalman




#endif //OPENKALMAN_TESTS_FROMEUCLIDEANEXPR_H
