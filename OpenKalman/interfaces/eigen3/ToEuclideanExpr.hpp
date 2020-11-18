/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2018-2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef OPENKALMAN_EIGEN3_TOEUCLIDEANEXPR_HPP
#define OPENKALMAN_EIGEN3_TOEUCLIDEANEXPR_HPP

namespace OpenKalman::Eigen3
{
  // ------------------------------ //
  //        ToEuclideanExpr         //
  // ------------------------------ //

  // Class documentation is in EigenForwardDeclarations.h
#ifdef __cpp_concepts
  template<coefficients Coefficients, typename BaseMatrix> requires
    (MatrixTraits<BaseMatrix>::dimension == Coefficients::size)
#else
  template<typename Coefficients, typename BaseMatrix>
#endif
  struct ToEuclideanExpr : OpenKalman::internal::MatrixBase<ToEuclideanExpr<Coefficients, BaseMatrix>, BaseMatrix>
  {
    static_assert(MatrixTraits<BaseMatrix>::dimension == Coefficients::size);
    using Base = OpenKalman::internal::MatrixBase<ToEuclideanExpr, BaseMatrix>;
    using Scalar = typename MatrixTraits<BaseMatrix>::Scalar; ///< Scalar type for this variable.
    static constexpr auto columns = MatrixTraits<BaseMatrix>::columns; ///< Number of columns.


    /// Default constructor.
    ToEuclideanExpr() : Base() {}


    /// Copy constructor.
    ToEuclideanExpr(const ToEuclideanExpr& other) : Base(other.base_matrix()) {}


    /// Move constructor.
    ToEuclideanExpr(ToEuclideanExpr&& other) noexcept: Base(std::move(other).base_matrix()) {}


    /// Construct from a compatible to-Euclidean expression.
#ifdef __cpp_concepts
    template<to_euclidean_expr Arg>
#else
    template<typename Arg, std::enable_if_t<to_euclidean_expr < Arg>, int> = 0>
#endif
    ToEuclideanExpr(Arg&& other) noexcept: Base(std::forward<Arg>(other).base_matrix())
    {
      static_assert(equivalent_to<typename MatrixTraits<Arg>::Coefficients, Coefficients>);
      static_assert(MatrixTraits<Arg>::columns == columns);
    }

    /// Construct from compatible matrix object.
#ifdef __cpp_concepts
    template<eigen_matrix Arg>
#else
    template<typename Arg, std::enable_if_t<eigen_matrix<Arg>, int> = 0>
#endif
    ToEuclideanExpr(Arg&& arg) noexcept : Base(std::forward<Arg>(arg))
    {
      static_assert(MatrixTraits<Arg>::dimension == Coefficients::size);
      static_assert(MatrixTraits<Arg>::columns == columns);
    }

    /// Construct from a list of coefficients.
#ifdef __cpp_concepts
    template<std::convertible_to<const Scalar> ... Args> requires
      (sizeof...(Args) == columns * (from_euclidean_expr<BaseMatrix> ? Coefficients::dimension : Coefficients::size))
#else
    template<
      typename ... Args, std::enable_if_t<std::conjunction_v<std::is_convertible<Args, const Scalar>...> and
        sizeof...(Args) == columns *
          (from_euclidean_expr < BaseMatrix > ? Coefficients::dimension : Coefficients::size), int> = 0>
#endif
    ToEuclideanExpr(Args ... args) : Base(MatrixTraits<BaseMatrix>::make(args...)) {}

    /// Copy assignment operator.
    auto& operator=(const ToEuclideanExpr& other)
    {
      if constexpr (not zero_matrix < BaseMatrix > and not identity_matrix<BaseMatrix>)
        if (this != &other)
          this->base_matrix() = other.base_matrix();
      return *this;
    }

    /// Move assignment operator.
    auto& operator=(ToEuclideanExpr&& other) noexcept
    {
      if constexpr (not zero_matrix < BaseMatrix > and not identity_matrix<BaseMatrix>)
        if (this != &other)
          this->base_matrix() = std::move(other).base_matrix();
      return *this;
    }

    /// Assign from a compatible to-Euclidean expression.
#ifdef __cpp_concepts
    template<to_euclidean_expr Arg>
#else
    template<typename Arg, std::enable_if_t<to_euclidean_expr < Arg>, int> = 0>
#endif
    auto& operator=(Arg&& other) noexcept
    {
      static_assert(equivalent_to < typename MatrixTraits<Arg>::Coefficients, Coefficients > );
      static_assert(MatrixTraits<Arg>::columns == columns);
      if constexpr (zero_matrix < BaseMatrix >)
      {
        static_assert(zero_matrix < Arg > );
      }
      else if constexpr (identity_matrix<BaseMatrix>)
      {
        static_assert(identity_matrix<Arg>);
      }
      else
      {
        this->base_matrix() = std::forward<Arg>(other).base_matrix();
      }
      return *this;
    }

    /// Assign from a general Eigen matrix.
#ifdef __cpp_concepts
    template<eigen_matrix Arg>
#else
    template<typename Arg, std::enable_if_t<eigen_matrix<Arg>, int> = 0>
#endif
    auto& operator=(Arg&& arg) noexcept
    {
      static_assert(MatrixTraits<Arg>::dimension == Coefficients::dimension);
      static_assert(MatrixTraits<Arg>::columns == columns);
      if constexpr (zero_matrix < BaseMatrix >)
      {
        static_assert(zero_matrix < Arg > );
      }
      else if constexpr (identity_matrix<BaseMatrix>)
      {
        static_assert(identity_matrix<Arg>);
      }
      else
      {
        this->base_matrix() = from_Euclidean<Coefficients>(std::forward<Arg>(arg));
      }
      return *this;
    }

    /// Increment from another expression.
#ifdef __cpp_concepts
    template<typename Arg> requires to_euclidean_expr<Arg> or eigen_matrix<Arg>
#else
    template<typename Arg, std::enable_if_t<to_euclidean_expr < Arg> or eigen_matrix<Arg>, int> = 0>
#endif
    auto& operator+=(Arg&& other) noexcept
    {
      static_assert(MatrixTraits<Arg>::columns == MatrixTraits<BaseMatrix>::columns);
      static_assert(MatrixTraits<Arg>::dimension == Coefficients::dimension);
      if constexpr(to_euclidean_expr < Arg >)
        static_assert(equivalent_to<typename MatrixTraits<Arg>::Coefficients, Coefficients>);
      this->base_matrix() = make_self_contained(from_Euclidean<Coefficients>(*this + std::forward<Arg>(other)));
      return *this;
    }

    /// Decrement from another expression.
#ifdef __cpp_concepts
    template<typename Arg> requires to_euclidean_expr<Arg> or eigen_matrix<Arg>
#else
    template<typename Arg, std::enable_if_t<to_euclidean_expr < Arg> or eigen_matrix<Arg>, int> = 0>
#endif
    auto& operator-=(Arg&& other) noexcept
    {
      static_assert(MatrixTraits<Arg>::columns == MatrixTraits<BaseMatrix>::columns);
      static_assert(MatrixTraits<Arg>::dimension == Coefficients::dimension);
      if constexpr(to_euclidean_expr < Arg >)
        static_assert(equivalent_to<typename MatrixTraits<Arg>::Coefficients, Coefficients>);
      this->base_matrix() = make_self_contained(from_Euclidean<Coefficients>(*this - std::forward<Arg>(other)));
      return *this;
    }

    /// Multiply by a scale factor.
    template<typename S, std::enable_if_t<std::is_convertible_v<S, Scalar>, int> = 0>
    auto& operator*=(const S scale)
    {
      this->base_matrix() = make_self_contained(from_Euclidean<Coefficients>(*this * scale));
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
      this->base_matrix() = make_self_contained(from_Euclidean<Coefficients>(*this / scale));
      return *this;
    }


    /// Zero coefficients.
    static auto zero()
    {
      using ST = native_matrix_t<BaseMatrix, Coefficients::dimension>;
      return MatrixTraits<ST>::zero();
    }


    /// Identity.
    static auto identity()
    {
      static_assert(MatrixTraits<BaseMatrix>::dimension == columns, "Identity requires a square matrix.");
      using ST = native_matrix_t<BaseMatrix, Coefficients::dimension>;
      return MatrixTraits<ST>::identity();
    }


    auto operator()(std::size_t i, std::size_t j)
    {
      if constexpr (element_settable < ToEuclideanExpr, 2 >)
        return OpenKalman::internal::ElementSetter(*this, i, j);
      else
        return const_cast<const ToEuclideanExpr&>(*this)(i, j);
    }


    auto operator()(std::size_t i, std::size_t j) const noexcept
    {
      return OpenKalman::internal::ElementSetter(*this, i, j);
    }


    auto operator[](std::size_t i)
    {
      if constexpr (element_settable < ToEuclideanExpr, 1 >)
        return OpenKalman::internal::ElementSetter(*this, i);
      else
        return const_cast<const ToEuclideanExpr&>(*this)[i];
    }


    auto operator[](std::size_t i) const noexcept { return OpenKalman::internal::ElementSetter(*this, i); }


    auto operator()(std::size_t i) { return operator[](i); }


    auto operator()(std::size_t i) const { return operator[](i); }
  };

} // OpenKalman::Eigen3


namespace OpenKalman
{
  // --------------------- //
  //        Traits         //
  // --------------------- //

  /// Define matrix traits.
  template<typename Coeffs, typename ArgType>
  struct MatrixTraits<Eigen3::ToEuclideanExpr<Coeffs, ArgType>>
  {
    using BaseMatrix = ArgType;
    using Coefficients = Coeffs;
    using Scalar = typename MatrixTraits<BaseMatrix>::Scalar;
    static constexpr auto dimension = Coefficients::dimension;
    static constexpr auto columns = MatrixTraits<BaseMatrix>::columns;
    static_assert(Coefficients::size == MatrixTraits<BaseMatrix>::dimension);

    template<typename Derived>
    using MatrixBaseType = Eigen3::internal::Eigen3MatrixBase<Derived, Eigen3::ToEuclideanExpr<Coeffs, ArgType>>;

    template<std::size_t rows = dimension, std::size_t cols = columns, typename S = Scalar>
    using NativeMatrix = typename MatrixTraits<BaseMatrix>::template NativeMatrix<rows, cols, S>;

    using SelfContained = Eigen3::ToEuclideanExpr<Coefficients, self_contained_t<BaseMatrix>>;

    template<TriangleType storage_triangle = TriangleType::lower, std::size_t dim = dimension, typename S = Scalar>
    using SelfAdjointBaseType = Eigen3::SelfAdjointMatrix<NativeMatrix<dim, dim, S>, storage_triangle>;

    template<TriangleType triangle_type = TriangleType::lower, std::size_t dim = dimension, typename S = Scalar>
    using TriangularBaseType = Eigen3::TriangularMatrix<NativeMatrix<dim, dim, S>, triangle_type>;

    template<std::size_t dim = dimension, typename S = Scalar>
    using DiagonalBaseType = Eigen3::DiagonalMatrix<NativeMatrix<dim, 1, S>>;


    /// Make from a regular matrix.
#ifdef __cpp_concepts
    template<typename C = Coefficients, typename Arg> requires
      Eigen3::eigen_matrix<Arg> or Eigen3::from_euclidean_expr<Arg>
#else
    template<typename C = Coefficients, typename Arg,
      std::enable_if_t<Eigen3::eigen_matrix<Arg> or Eigen3::from_euclidean_expr<Arg>, int> = 0>
#endif
    static auto make(Arg&& arg) noexcept
    {
      static_assert(MatrixTraits<Arg>::dimension == C::size);
      using namespace Eigen3;
      return to_Euclidean<C>(std::forward<Arg>(arg));
    }


#ifdef __cpp_concepts
    template<std::convertible_to<const Scalar> ... Args>
#else
    template<typename ... Args, std::enable_if_t<std::conjunction_v<std::is_convertible<Args, const Scalar>...>, int> = 0>
#endif
    static auto
    make(Args ... args)
    {
      static_assert(sizeof...(Args) == Coefficients::size * columns);
      return make(MatrixTraits<BaseMatrix>::make(args...));
    }


    static auto zero() { return Eigen3::ToEuclideanExpr<Coefficients, BaseMatrix>::zero(); }


    static auto identity() { return Eigen3::ToEuclideanExpr<Coefficients, BaseMatrix>::identity(); }
  };


} // namespace OpenKalman

#endif //OPENKALMAN_EIGEN3_TOEUCLIDEANEXPR_HPP
