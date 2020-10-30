/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef OPENKALMAN_EIGEN3_TRIANGULARMATRIX_HPP
#define OPENKALMAN_EIGEN3_TRIANGULARMATRIX_HPP

namespace OpenKalman::Eigen3
{
  template<typename BaseMatrix, TriangleType triangle_type>
  struct TriangularMatrix
    : OpenKalman::internal::MatrixBase<TriangularMatrix<BaseMatrix, triangle_type>, BaseMatrix>
  {
    using Base = OpenKalman::internal::MatrixBase<TriangularMatrix, BaseMatrix>;
    static constexpr auto uplo = triangle_type == TriangleType::upper ? Eigen::Upper : Eigen::Lower;
    using View = Eigen::TriangularView<std::remove_reference_t<BaseMatrix>, uplo>;
    using Scalar = typename MatrixTraits<BaseMatrix>::Scalar;
    static constexpr auto dimension = MatrixTraits<BaseMatrix>::dimension;
    static_assert(dimension == MatrixTraits<BaseMatrix>::columns);

    /// Default constructor
    TriangularMatrix() : Base(), view(this->base_matrix()) {}

    /// Copy constructor
    TriangularMatrix(const TriangularMatrix& other) : TriangularMatrix(other.base_matrix()) {}

    /// Move constructor
    TriangularMatrix(TriangularMatrix&& other) noexcept
      : TriangularMatrix(std::move(other).base_matrix()) {}

    /// Construct from a compatible triangular matrix object of the same TriangleType.
#ifdef __cpp_concepts
    template<eigen_triangular_expr Arg>
#else
    template<typename Arg, std::enable_if_t<is_eigen_triangular_expr_v<Arg>, int> = 0>
#endif
    TriangularMatrix(Arg&& arg) noexcept: TriangularMatrix(base_matrix(std::forward<Arg>(arg)))
    {
      static_assert(is_lower_triangular_v<Arg> == is_lower_triangular_v<TriangularMatrix>);
    }

    /// Construct from a compatible self-adjoint matrix object
#ifdef __cpp_concepts
    template<eigen_self_adjoint_expr Arg>
#else
    template<typename Arg, std::enable_if_t<is_eigen_self_adjoint_expr_v<Arg>, int> = 0>
#endif
    TriangularMatrix(Arg&& arg) noexcept
      : TriangularMatrix(Cholesky_factor<triangle_type>(std::forward<Arg>(arg))) {}

    /// Construct from a reference to a regular or diagonal matrix object
#ifdef __cpp_concepts
    template<typename Arg> requires eigen_matrix<Arg> or eigen_diagonal_expr<Arg>
#else
    template<typename Arg, std::enable_if_t<is_eigen_matrix_v<Arg> or is_eigen_diagonal_expr_v<Arg>, int> = 0>
#endif
    TriangularMatrix(Arg&& arg) noexcept: Base(std::forward<Arg>(arg)), view(this->base_matrix()) {}

    /// Construct from a list of scalar coefficients, in row-major order.
#ifdef __cpp_concepts
    template<std::convertible_to<const Scalar> ... Args> requires
      (triangle_type != TriangleType::diagonal and (not eigen_diagonal_expr<BaseMatrix> and
      sizeof...(Args) == dimension * dimension)) or
      (eigen_diagonal_expr<BaseMatrix> and sizeof...(Args) == dimension)
#else
    template<
      typename ... Args, std::enable_if_t<
        std::conjunction_v<std::is_convertible<Args, const Scalar>...> and
          ((not is_eigen_diagonal_expr_v < BaseMatrix > and sizeof...(Args) == dimension * dimension and
              triangle_type != TriangleType::diagonal) or
            (is_eigen_diagonal_expr_v < BaseMatrix > and sizeof...(Args) == dimension)), int> = 0>
#endif
    TriangularMatrix(Args ... args) : TriangularMatrix(MatrixTraits<BaseMatrix>::make(args...)) {}

    /// Construct from a list of scalar coefficients, in row-major order.
#if defined (__cpp_concepts) && defined (__clang__) // Because of compiler issue in at least GCC version 10.1.0
    template<std::convertible_to<const Scalar> ... Args> requires (triangle_type == TriangleType::diagonal) and
    (not eigen_diagonal_expr<BaseMatrix>) and (sizeof...(Args) == dimension)
#else
    template<
      typename ... Args, std::enable_if_t<
        std::conjunction_v<std::is_convertible<Args, const Scalar>...> and not eigen_diagonal_expr < BaseMatrix>and
    sizeof...(Args) == dimension and triangle_type == TriangleType::diagonal, int> = 0>
#endif
    TriangularMatrix(Args ... args)
      : TriangularMatrix(strict_matrix(DiagonalMatrix {static_cast<const Scalar>(args)...})) {}

    /// Copy assignment operator
    auto& operator=(const TriangularMatrix& other)
    {
      if constexpr (not is_zero_v < BaseMatrix > and not is_identity_v<BaseMatrix>)
        if (this != &other)
        {
          this->base_matrix().template triangularView<uplo>() = other.base_matrix();
        }
      return *this;
    }

    /// Move assignment operator
    auto& operator=(TriangularMatrix&& other) noexcept
    {
      if constexpr (not is_zero_v < BaseMatrix > and not is_identity_v<BaseMatrix>)
        if (this != &other)
        {
          this->base_matrix() = std::move(other).base_matrix();
        }
      return *this;
    }


    /// Assign from another triangular matrix (must be the same triangle)
#ifdef __cpp_concepts
    template<eigen_triangular_expr Arg>
#else
    template<typename Arg, std::enable_if_t<is_eigen_triangular_expr_v<Arg>, int> = 0>
#endif

    auto& operator=(Arg&& arg)
    {
      static_assert(is_upper_triangular_v<Arg> == is_upper_triangular_v<TriangularMatrix>);
      if constexpr (is_zero_v<BaseMatrix>)
      {
        static_assert(is_zero_v<Arg>);
      }
      else if constexpr (is_identity_v<BaseMatrix>)
      {
        static_assert(is_identity_v<Arg>);
      }
      else if constexpr(std::is_lvalue_reference_v<Arg>)
      {
        this->base_matrix().template triangularView<uplo>() = base_matrix(arg);
      }
      else
      {
        this->base_matrix() = std::forward<Arg>(arg);
      }
      return *this;
    }

    /// Assign by converting from a self-adjoint special matrix
#ifdef __cpp_concepts
    template<eigen_self_adjoint_expr Arg>
#else
    template<typename Arg, std::enable_if_t<is_eigen_self_adjoint_expr_v<Arg>, int> = 0>
#endif

    auto& operator=(Arg&& arg)
    {
      if constexpr (is_zero_v < BaseMatrix >)
      {
        static_assert(is_zero_v < Arg > );
      }
      else if constexpr (is_identity_v<BaseMatrix>)
      {
        static_assert(is_identity_v<Arg>);
      }
      else
      {
        this->base_matrix().template triangularView<uplo>() =
          base_matrix(Cholesky_factor<triangle_type>(std::forward<Arg>(arg)));
      }
      return *this;
    }

    /// Assign from an Eigen::MatrixBase derived object. (Only uses the triangular part.)
#ifdef __cpp_concepts
    template<eigen_matrix Arg>
#else

    template<typename Arg, std::enable_if_t<is_eigen_matrix_v<Arg>, int> = 0>
#endif
    auto& operator=(Arg&& arg)
    {
      if constexpr (is_zero_v<BaseMatrix>)
      {
        static_assert(is_zero_v<Arg>);
      }
      else if constexpr (is_identity_v<BaseMatrix>)
      {
        static_assert(is_identity_v<Arg>);
      }
      else if constexpr(std::is_lvalue_reference_v<Arg>)
      {
        this->base_matrix().template triangularView<uplo>() = arg;
      }
      else
      {
        this->base_matrix() = std::forward<Arg>(arg);
      }
      return *this;
    }

    /// Assign by copying from an Eigen::TriangularBase derived object.
    template<typename Arg, unsigned int UpLo>
    auto& operator=(const Eigen::TriangularView<Arg, UpLo>& arg)
    {
      if constexpr (is_zero_v<BaseMatrix>)
      {
        static_assert(is_zero_v<Arg>);
      }
      else if constexpr (is_identity_v<BaseMatrix>)
      {
        static_assert(is_identity_v<Arg>);
      }
      else if constexpr(((UpLo & Eigen::Lower) != 0) == (triangle_type == TriangleType::lower))
      {
        this->base_matrix().template triangularView<UpLo>() = arg.nestedExpression();
      }
      else
      {
        this->base_matrix().template triangularView<UpLo>() = arg.nestedExpression().adjoint();
      }
      return *this;
    }

    /// Assign by moving from an Eigen::TriangularBase derived object.
    template<typename Arg>
    auto& operator=(Eigen::TriangularView<Arg, uplo>&& arg) noexcept
    {
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
        this->base_matrix() = std::move(arg.nestedExpression());
      }
      return *this;
    }

    template<typename Arg>
    auto& operator+=(const TriangularMatrix<Arg, triangle_type>& arg)
    {
      static_assert(MatrixTraits<Arg>::dimension == dimension);
      this->base_view() += arg.base_matrix();
      return *this;
    }

    template<typename Arg>
    auto& operator-=(const TriangularMatrix<Arg, triangle_type>& arg)
    {
      static_assert(MatrixTraits<Arg>::dimension == dimension);
      this->base_view() -= arg.base_matrix();
      return *this;
    }

#ifdef __cpp_concepts
    template<std::convertible_to<Scalar> S>
#else
    template<typename S, std::enable_if_t<std::is_convertible_v<S, Scalar>, int> = 0>
#endif
    auto& operator*=(const S s)
    {
      this->base_view() *= s;
      return *this;
    }

#ifdef __cpp_concepts
    template<std::convertible_to<Scalar> S>
#else
    template<typename S, std::enable_if_t<std::is_convertible_v<S, Scalar>, int> = 0>
#endif
    auto& operator/=(const S s)
    {
      this->base_view() /= s;
      return *this;
    }

    template<typename Arg>
    auto& operator*=(const TriangularMatrix<Arg, triangle_type>& arg)
    {
      static_assert(MatrixTraits<Arg>::dimension == dimension);
      this->base_view() = this->base_view() * strict_matrix(arg);
      return *this;
    }

    constexpr auto& base_view()& { return view; }

    constexpr const auto& base_view() const& { return view; }

    constexpr auto&& base_view()&& { return std::move(view); }

    constexpr const auto&& base_view() const&& { return std::move(view); }


    auto operator()(std::size_t i, std::size_t j)
    {
      if constexpr (is_element_settable_v < TriangularMatrix, 2 >)
        return OpenKalman::internal::ElementSetter(*this, i, j);
      else
        return const_cast<const TriangularMatrix&>(*this)(i, j);
    }

    auto operator()(std::size_t i, std::size_t j) const noexcept
    {
      return OpenKalman::internal::ElementSetter(*this, i, j);
    }

    auto operator[](std::size_t i)
    {
      if constexpr(is_element_gettable_v < TriangularMatrix, 1 >)
        return OpenKalman::internal::ElementSetter(*this, i);
      else if constexpr(is_element_gettable_v < TriangularMatrix, 2 >)
        return OpenKalman::internal::ElementSetter(*this, i, i);
      else
        return const_cast<const TriangularMatrix&>(*this)[i];
    }

    auto operator[](std::size_t i) const
    {
      if constexpr(is_element_gettable_v < TriangularMatrix, 1 >)
        return OpenKalman::internal::ElementSetter(*this, i);
      else
        return OpenKalman::internal::ElementSetter(*this, i, i);
    }

    auto operator()(std::size_t i) { return operator[](i); }

    auto operator()(std::size_t i) const { return operator[](i); }


#ifdef __cpp_concepts
    template<eigen_matrix B>
#else
    template<typename B, std::enable_if_t<is_eigen_matrix_v<B>, int> = 0>
#endif
    auto solve(const B& b) const
    {
      return this->base_view().solve(b);
    }

    static auto zero() { return MatrixTraits<BaseMatrix>::zero(); }

    static auto identity()
    {
      auto b = MatrixTraits<BaseMatrix>::identity();
      return TriangularMatrix<std::decay_t<decltype(b)>, TriangleType::diagonal>(std::move(b));
    }

  private:
    View view;
  };


  /////////////////////////////////////
  //        Deduction Guides         //
  /////////////////////////////////////

#ifdef __cpp_concepts
  template<typename M> requires eigen_matrix<M> or eigen_diagonal_expr<M>
#else
  template<typename M, std::enable_if_t<is_eigen_matrix_v<M> or is_eigen_diagonal_expr_v<M>, int> = 0>
#endif
  TriangularMatrix(M&&) -> TriangularMatrix<lvalue_or_strict_t < M>, TriangleType::lower>;

#ifdef __cpp_concepts
  template<typename M> requires (eigen_triangular_expr<M> and is_lower_triangular_v<M>) or
    (eigen_self_adjoint_expr<M> and is_lower_storage_triangle_v<M>)
#else
  template<
    typename M, std::enable_if_t<(is_eigen_triangular_expr_v<M> and is_lower_triangular_v<M>) or
      (is_eigen_self_adjoint_expr_v<M> and is_lower_storage_triangle_v<M>), int> = 0>
#endif
  TriangularMatrix(M&&) -> TriangularMatrix<typename MatrixTraits<M>::BaseMatrix, TriangleType::lower>;

#ifdef __cpp_concepts
  template<typename M> requires (eigen_triangular_expr<M> and is_upper_triangular_v<M>) or
    (eigen_self_adjoint_expr<M> and is_upper_storage_triangle_v<M>)
#else
  template<
    typename M, std::enable_if_t<(is_eigen_triangular_expr_v<M> and is_upper_triangular_v<M>) or
      (is_eigen_self_adjoint_expr_v<M> and is_upper_storage_triangle_v<M>), int> = 0>
#endif
  TriangularMatrix(M&&) -> TriangularMatrix<typename MatrixTraits<M>::BaseMatrix, TriangleType::upper>;

  template<typename Arg, unsigned int UpLo>
  TriangularMatrix(const Eigen::TriangularView<Arg, UpLo>&)
  -> TriangularMatrix<Arg, UpLo & Eigen::Upper ? TriangleType::upper : TriangleType::lower>;

  /// If the arguments are a sequence of scalars, deduce a square, lower triangular matrix.
#ifdef __cpp_concepts
  template<typename ... Args> requires ((sizeof...(Args) >= 1) and ... and std::is_arithmetic_v<Args>)
#else
  template<
    typename ... Args,
    std::enable_if_t<sizeof...(Args) >= 1 and std::conjunction_v<std::is_arithmetic<Args>...>, int> = 0>
#endif
  TriangularMatrix(Args ...) -> TriangularMatrix<
    Eigen::Matrix<std::decay_t<std::common_type_t<Args...>>, OpenKalman::internal::constexpr_sqrt(sizeof...(Args)),
      OpenKalman::internal::constexpr_sqrt(sizeof...(Args))>, TriangleType::lower>;


  ///////////////////////////////////
  //        Make functions         //
  ///////////////////////////////////

#ifdef __cpp_concepts
  template<TriangleType t = TriangleType::lower, typename M> requires eigen_matrix<M> or eigen_diagonal_expr<M>
#else
  template<
    TriangleType t = TriangleType::lower, typename M,
    std::enable_if_t<is_eigen_matrix_v<M> or is_eigen_diagonal_expr_v<M>, int> = 0>
#endif
  auto make_EigenTriangularMatrix(M&& m)
  {
    return TriangularMatrix<lvalue_or_strict_t<M>, t > (std::forward<M>(m));
  }

#ifdef __cpp_concepts
  template<TriangleType t, eigen_triangular_expr M>
#else
  template<TriangleType t, typename M, std::enable_if_t<is_eigen_triangular_expr_v<M>, int> = 0>
#endif
  auto make_EigenTriangularMatrix(M&& m)
  {
    static_assert(t == MatrixTraits<M>::triangle_type);
    return TriangularMatrix<lvalue_or_strict_t<M>, t > (std::forward<M>(m));
  }

#ifdef __cpp_concepts
  template<eigen_triangular_expr M>
#else
  template<typename M, std::enable_if_t<is_eigen_triangular_expr_v<M>, int> = 0>
#endif
  auto make_EigenTriangularMatrix(M&& m)
  {
    return make_EigenTriangularMatrix<MatrixTraits<M>::triangle_type>(std::forward<M>(m));
  }

} // namespace OpenKalman::Eigen3


namespace OpenKalman
{
  ///////////////////////////
  //        Traits         //
  ///////////////////////////

  template<typename ArgType, TriangleType triangle>
  struct MatrixTraits<Eigen3::TriangularMatrix<ArgType, triangle>>
  {
    static constexpr TriangleType triangle_type = triangle;
    using BaseMatrix = ArgType;
    using Scalar = typename MatrixTraits<BaseMatrix>::Scalar;
    static constexpr auto dimension = MatrixTraits<BaseMatrix>::dimension;
    static constexpr auto columns = MatrixTraits<BaseMatrix>::columns;
    static_assert(dimension == columns, "A triangular matrix must be square.");

    template<typename Derived>
    using MatrixBaseType =
    Eigen3::internal::Eigen3MatrixBase<Derived, Eigen3::TriangularMatrix<std::decay_t<BaseMatrix>, triangle_type>>;

    template<typename Derived>
    using CovarianceBaseType = Eigen3::internal::Eigen3CovarianceBase<
      Derived, Eigen3::TriangularMatrix<std::decay_t<BaseMatrix>, triangle_type>>;

    template<std::size_t rows = dimension, std::size_t cols = dimension, typename S = Scalar>
    using StrictMatrix = typename MatrixTraits<BaseMatrix>::template StrictMatrix<rows, cols, S>;

    using Strict = Eigen3::TriangularMatrix<strict_t<BaseMatrix>, triangle_type>;

    template<TriangleType t = triangle_type, std::size_t dim = dimension, typename S = Scalar>
    using SelfAdjointBaseType = Eigen3::SelfAdjointMatrix<StrictMatrix<dim, dim, S>, t>;

    template<TriangleType t = triangle_type, std::size_t dim = dimension, typename S = Scalar>
    using TriangularBaseType = Eigen3::TriangularMatrix<StrictMatrix<dim, dim, S>, t>;

    template<std::size_t dim = dimension, typename S = Scalar>
    using DiagonalBaseType = Eigen3::DiagonalMatrix<StrictMatrix<dim, 1, S>>;

#ifdef __cpp_concepts

    template<TriangleType t = triangle_type, typename Arg>
    requires (not std::convertible_to<Arg, const Scalar>)
#else
    template<TriangleType t = triangle_type, typename Arg,
      std::enable_if_t<not std::is_convertible_v<Arg, const Scalar>, int> = 0>
#endif
    static auto make(Arg&& arg) noexcept
    {
      return Eigen3::TriangularMatrix<std::decay_t<Arg>, t>(std::forward<Arg>(arg));
    }

    /// Make triangular matrix using a list of coefficients in row-major order.
    /// Only the coefficients in the lower-left corner are significant.
#ifdef __cpp_concepts

    template<TriangleType t = triangle_type, std::convertible_to<const Scalar> ... Args>
#else
    template<TriangleType t = triangle_type, typename ... Args,
      std::enable_if_t<std::conjunction_v<std::is_convertible<Args, const Scalar>...>, int> = 0>
#endif
    static auto make(Args ... args)
    {
      static_assert(sizeof...(Args) == dimension * dimension);
      return make<t>(MatrixTraits<BaseMatrix>::make(args...));
    }

    static auto zero() { return Eigen3::TriangularMatrix<BaseMatrix, triangle_type>::zero(); }

    static auto identity() { return Eigen3::TriangularMatrix<BaseMatrix, triangle_type>::identity(); }

  };

} // namespace OpenKalman


namespace OpenKalman::Eigen3
{
  //////////////////////////////
  //        Overloads         //
  //////////////////////////////

#ifdef __cpp_concepts
  template<Eigen3::eigen_triangular_expr Arg>
#else
  template<typename Arg, std::enable_if_t<Eigen3::is_eigen_triangular_expr_v<Arg>, int> = 0>
#endif
  constexpr auto
  transpose(Arg&& arg)
  {
    constexpr auto t = triangle_type_of_v<Arg> == TriangleType::lower ? TriangleType::upper :
                       triangle_type_of_v<Arg> == TriangleType::upper ? TriangleType::lower : TriangleType::diagonal;
    auto b = base_matrix(std::forward<Arg>(arg)).transpose();
    return Eigen3::TriangularMatrix<decltype(b), t>(std::move(b));
  }


#ifdef __cpp_concepts
  template<Eigen3::eigen_triangular_expr Arg>
#else
  template<typename Arg, std::enable_if_t<Eigen3::is_eigen_triangular_expr_v<Arg>, int> = 0>
#endif
  constexpr auto
  adjoint(Arg&& arg)
  {
    constexpr auto t = triangle_type_of_v<Arg> == TriangleType::lower ? TriangleType::upper :
                       triangle_type_of_v<Arg> == TriangleType::upper ? TriangleType::lower : TriangleType::diagonal;
    auto b = base_matrix(std::forward<Arg>(arg)).adjoint();
    return Eigen3::TriangularMatrix<decltype(b), t>(std::move(b));
  }


#ifdef __cpp_concepts
  template<Eigen3::eigen_triangular_expr Arg, typename U> requires
      (Eigen3::eigen_matrix<U> or Eigen3::eigen_triangular_expr<U> or Eigen3::eigen_self_adjoint_expr<U> or
        Eigen3::eigen_diagonal_expr<U> or Eigen3::euclidean_expr<U>) and
    (not std::is_const_v<std::remove_reference_t<Arg>>)
#else
  template<typename Arg, typename U,
    std::enable_if_t<Eigen3::is_eigen_triangular_expr_v<Arg> and
      (Eigen3::is_eigen_matrix_v<U> or Eigen3::is_eigen_triangular_expr_v<U> or Eigen3::is_eigen_self_adjoint_expr_v<U> or
        Eigen3::is_eigen_diagonal_expr_v<U> or Eigen3::is_from_euclidean_expr_v<U> or Eigen3::is_to_euclidean_expr_v<U>) and
      not std::is_const_v<std::remove_reference_t<Arg>>, int> = 0>
#endif
  inline Arg&
  rank_update(Arg& arg, const U& u, const typename MatrixTraits<Arg>::Scalar alpha = 1)
  {
    static_assert(MatrixTraits<U>::dimension == MatrixTraits<Arg>::dimension);
    using Scalar = typename MatrixTraits<Arg>::Scalar;
    constexpr auto t = is_lower_triangular_v<Arg> ? Eigen::Lower : Eigen::Upper;
    for (Eigen::Index i = 0; i < Eigen::Index(MatrixTraits<U>::columns); ++i)
    {
      if (Eigen::internal::llt_inplace<Scalar, t>::rankUpdate(arg.base_matrix(), u.col(i), alpha) >= 0)
      {
        throw (std::runtime_error("TriangularMatrix rank_update: product is not positive definite"));
      }
    }
    return arg;
  }


#ifdef __cpp_concepts
  template<Eigen3::eigen_triangular_expr Arg, typename U> requires
      (Eigen3::eigen_matrix<U> or Eigen3::eigen_triangular_expr<U> or Eigen3::eigen_self_adjoint_expr<U> or
        Eigen3::eigen_diagonal_expr<U> or Eigen3::euclidean_expr<U>)
#else
  template<typename Arg, typename U,
    std::enable_if_t<Eigen3::is_eigen_triangular_expr_v<Arg> and
      (Eigen3::is_eigen_matrix_v<U> or Eigen3::is_eigen_triangular_expr_v<U> or Eigen3::is_eigen_self_adjoint_expr_v<U> or
        Eigen3::is_eigen_diagonal_expr_v<U> or Eigen3::is_from_euclidean_expr_v<U> or Eigen3::is_to_euclidean_expr_v<U>), int> = 0>
#endif
  inline auto
  rank_update(Arg&& arg, const U& u, const typename MatrixTraits<Arg>::Scalar alpha = 1)
  {
    static_assert(MatrixTraits<U>::dimension == MatrixTraits<Arg>::dimension);
    auto sa = std::forward<Arg>(arg);
    rank_update(sa, u, alpha);
    return sa;
  }


  /// Perform an LQ decomposition of matrix A=[L,0]Q, where L is a lower-triangular matrix, and Q is orthogonal.
  /// Returns L as a lower-triangular matrix.
#ifdef __cpp_concepts
  template<Eigen3::eigen_triangular_expr A>
#else
  template<typename A, std::enable_if_t<Eigen3::is_eigen_triangular_expr_v<A>, int> = 0>
#endif
  constexpr decltype(auto)
  LQ_decomposition(A&& a)
  {
    if constexpr(is_lower_triangular_v<A>)
      return std::forward<A>(a);
    else
      return LQ_decomposition(strict_matrix(std::forward<A>(a)));
  }


  /// Perform a QR decomposition of matrix A=Q[U,0], where U is an upper-triangular matrix, and Q is orthogonal.
  /// Returns U as an upper-triangular matrix.
#ifdef __cpp_concepts
  template<Eigen3::eigen_triangular_expr A>
#else
  template<typename A, std::enable_if_t<Eigen3::is_eigen_triangular_expr_v<A>, int> = 0>
#endif
  constexpr decltype(auto)
  QR_decomposition(A&& a)
  {
    if constexpr(is_upper_triangular_v<A>)
      return std::forward<A>(a);
    else
      return QR_decomposition(strict_matrix(std::forward<A>(a)));
  }


  ///////////////////////////////
  //        Arithmetic         //
  ///////////////////////////////

#ifdef __cpp_concepts
  template<Eigen3::eigen_triangular_expr Arg1, Eigen3::eigen_triangular_expr Arg2> requires
    (not is_diagonal_v<Arg1>) and (not is_diagonal_v<Arg2>)
#else
  template<typename Arg1, typename Arg2, std::enable_if_t<
    Eigen3::is_eigen_triangular_expr_v<Arg1> and Eigen3::is_eigen_triangular_expr_v<Arg2> and
      not is_diagonal_v<Arg1> and not is_diagonal_v<Arg2>, int> = 0>
#endif
  inline auto operator+(Arg1&& arg1, Arg2&& arg2)
  {
    static_assert(MatrixTraits<Arg1>::dimension == MatrixTraits<Arg2>::dimension);
    if constexpr(is_lower_triangular_v<Arg1> == is_lower_triangular_v<Arg2>)
    {
      auto ret = MatrixTraits<Arg1>::make(base_matrix(std::forward<Arg1>(arg1)) + base_matrix(std::forward<Arg2>(arg2)));
      if constexpr (not std::is_lvalue_reference_v<Arg1&&> or not std::is_lvalue_reference_v<Arg2&&>)
        return strict(std::move(ret));
      else
        return ret;
    }
    else
    {
      auto ret = strict_matrix(std::forward<Arg1>(arg1));
      constexpr auto mode = is_upper_triangular_v<Arg2> ? Eigen::Upper : Eigen::Lower;
      ret.template triangularView<mode>() += base_matrix(std::forward<Arg2>(arg2));
      return ret;
    }
  }


#ifdef __cpp_concepts
  template<typename Arg1, typename Arg2> requires
    ((Eigen3::eigen_triangular_expr<Arg1> and is_diagonal_v<Arg2>) or
      (is_diagonal_v<Arg1> and Eigen3::eigen_triangular_expr<Arg2>)) and
      (not is_zero_v<Arg1>) and (not is_zero_v<Arg2>) and (not is_identity_v<Arg1>) and (not is_identity_v<Arg2>)
#else
  template<typename Arg1, typename Arg2,
    std::enable_if_t<((Eigen3::is_eigen_triangular_expr_v<Arg1> and is_diagonal_v<Arg2>) or
      (is_diagonal_v<Arg1> and Eigen3::is_eigen_triangular_expr_v<Arg2>)) and
      not is_zero_v<Arg1> and not is_zero_v<Arg2> and not is_identity_v<Arg1> and not is_identity_v<Arg2>, int> = 0>
#endif
  inline auto operator+(Arg1&& arg1, Arg2&& arg2)
  {
    static_assert(MatrixTraits<Arg1>::dimension == MatrixTraits<Arg2>::dimension);
    if constexpr(Eigen3::eigen_triangular_expr<Arg1>)
    {
      auto ret = MatrixTraits<Arg1>::make(base_matrix(std::forward<Arg1>(arg1)) + std::forward<Arg2>(arg2));
      if constexpr (not std::is_lvalue_reference_v<Arg1&&> or not std::is_lvalue_reference_v<Arg2&&>)
        return strict(std::move(ret));
      else
        return ret;
    }
    else
    {
      auto ret = MatrixTraits<Arg2>::make(std::forward<Arg1>(arg1) + base_matrix(std::forward<Arg2>(arg2)));
      if constexpr (not std::is_lvalue_reference_v<Arg1&&> or not std::is_lvalue_reference_v<Arg2&&>)
        return strict(std::move(ret));
      else
        return ret;
    }
  }


#ifdef __cpp_concepts
  template<typename Arg1, typename Arg2> requires (Eigen3::eigen_triangular_expr<Arg1> and is_identity_v<Arg2>) or
    (is_identity_v<Arg1> and Eigen3::eigen_triangular_expr<Arg2>)
#else
  template<typename Arg1, typename Arg2,
    std::enable_if_t<(Eigen3::is_eigen_triangular_expr_v<Arg1> and is_identity_v<Arg2>) or
      (is_identity_v<Arg1> and Eigen3::is_eigen_triangular_expr_v<Arg2>), int> = 0>
#endif
  inline auto operator+(Arg1&& arg1, Arg2&& arg2)
  {
    static_assert(MatrixTraits<Arg1>::dimension == MatrixTraits<Arg2>::dimension);
    static_assert(MatrixTraits<Arg1>::columns == MatrixTraits<Arg2>::columns);
    if constexpr(Eigen3::eigen_triangular_expr<Arg1>)
    {
      auto ret = MatrixTraits<Arg1>::make(base_matrix(std::forward<Arg1>(arg1)) + std::forward<Arg2>(arg2));
      if constexpr (not std::is_lvalue_reference_v<Arg1&&> or not std::is_lvalue_reference_v<Arg2&&>) return strict(std::move(ret)); else return ret;
    }
    else
    {
      auto ret = MatrixTraits<Arg2>::make(std::forward<Arg1>(arg1) + base_matrix(std::forward<Arg2>(arg2)));
      if constexpr (not std::is_lvalue_reference_v<Arg1&&> or not std::is_lvalue_reference_v<Arg2&&>) return strict(std::move(ret)); else return ret;
    }
  }


#ifdef __cpp_concepts
  template<typename Arg1, typename Arg2> requires (Eigen3::eigen_triangular_expr<Arg1> and is_zero_v<Arg2>) or
    (is_zero_v<Arg1> and Eigen3::eigen_triangular_expr<Arg2>)
#else
  template<typename Arg1, typename Arg2,
    std::enable_if_t<(Eigen3::is_eigen_triangular_expr_v<Arg1> and is_zero_v<Arg2>) or
      (is_zero_v<Arg1> and Eigen3::is_eigen_triangular_expr_v<Arg2>), int> = 0>
#endif
  constexpr decltype(auto) operator+(Arg1&& arg1, Arg2&& arg2)
  {
    static_assert(MatrixTraits<Arg1>::dimension == MatrixTraits<Arg2>::dimension);
    static_assert(MatrixTraits<Arg1>::columns == MatrixTraits<Arg2>::columns);
    if constexpr(Eigen3::eigen_triangular_expr<Arg1>)
    {
      return std::forward<Arg1>(arg1);
    }
    else
    {
      return std::forward<Arg2>(arg2);
    }
  }


  ////

#ifdef __cpp_concepts
  template<Eigen3::eigen_triangular_expr Arg1, Eigen3::eigen_triangular_expr Arg2>
#else
  template<typename Arg1, typename Arg2,
    std::enable_if_t<Eigen3::is_eigen_triangular_expr_v<Arg1> and Eigen3::is_eigen_triangular_expr_v<Arg2>, int> = 0>
#endif
  inline auto operator-(Arg1&& arg1, Arg2&& arg2)
  {
    static_assert(MatrixTraits<Arg1>::dimension == MatrixTraits<Arg2>::dimension);
    if constexpr(is_lower_triangular_v<Arg1> == is_lower_triangular_v<Arg2>)
    {
      auto ret = MatrixTraits<Arg1>::make(base_matrix(std::forward<Arg1>(arg1)) - base_matrix(std::forward<Arg2>(arg2)));
      if constexpr (not std::is_lvalue_reference_v<Arg1&&> or not std::is_lvalue_reference_v<Arg2&&>) return strict(std::move(ret)); else return ret;
    }
    else
    {
      auto ret = strict_matrix(std::forward<Arg1>(arg1));
      constexpr auto mode = is_upper_triangular_v<Arg2> ? Eigen::Upper : Eigen::Lower;
      ret.template triangularView<mode>() -= base_matrix(std::forward<Arg2>(arg2));
      return ret;
    }
  }


#ifdef __cpp_concepts
  template<typename Arg1, typename Arg2> requires
    ((Eigen3::eigen_triangular_expr<Arg1> and is_diagonal_v<Arg2>) or
      (is_diagonal_v<Arg1> and Eigen3::eigen_triangular_expr<Arg2>)) and
    (not is_zero_v<Arg1>) and (not is_zero_v<Arg2>) and (not is_identity_v<Arg1>) and (not is_identity_v<Arg2>)
#else
  template<typename Arg1, typename Arg2,
    std::enable_if_t<((Eigen3::is_eigen_triangular_expr_v<Arg1> and is_diagonal_v<Arg2>) or
      (is_diagonal_v<Arg1> and Eigen3::is_eigen_triangular_expr_v<Arg2>)) and
      not is_zero_v<Arg1> and not is_zero_v<Arg2> and not is_identity_v<Arg1> and not is_identity_v<Arg2>, int> = 0>
#endif
  inline auto operator-(Arg1&& arg1, Arg2&& arg2)
  {
    static_assert(MatrixTraits<Arg1>::dimension == MatrixTraits<Arg2>::dimension);
    if constexpr(Eigen3::eigen_triangular_expr<Arg1>)
    {
      auto ret = MatrixTraits<Arg1>::make(base_matrix(std::forward<Arg1>(arg1)) - std::forward<Arg2>(arg2));
      if constexpr (not std::is_lvalue_reference_v<Arg1&&> or not std::is_lvalue_reference_v<Arg2&&>) return strict(std::move(ret)); else return ret;
    }
    else
    {
      auto ret = MatrixTraits<Arg2>::make(std::forward<Arg1>(arg1) - base_matrix(std::forward<Arg2>(arg2)));
      if constexpr (not std::is_lvalue_reference_v<Arg1&&> or not std::is_lvalue_reference_v<Arg2&&>) return strict(std::move(ret)); else return ret;
    }
  }


#ifdef __cpp_concepts
  template<typename Arg1, typename Arg2> requires (Eigen3::eigen_triangular_expr<Arg1> and is_identity_v<Arg2>) or
    (is_identity_v<Arg1> and Eigen3::eigen_triangular_expr<Arg2>)
#else
  template<typename Arg1, typename Arg2,
    std::enable_if_t<(Eigen3::is_eigen_triangular_expr_v<Arg1> and is_identity_v<Arg2>) or
      (is_identity_v<Arg1> and Eigen3::is_eigen_triangular_expr_v<Arg2>), int> = 0>
#endif
  inline auto operator-(Arg1&& arg1, Arg2&& arg2)
  {
    static_assert(MatrixTraits<Arg1>::dimension == MatrixTraits<Arg2>::dimension);
    static_assert(MatrixTraits<Arg1>::columns == MatrixTraits<Arg2>::columns);
    if constexpr(Eigen3::eigen_triangular_expr<Arg1>)
    {
      auto ret = MatrixTraits<Arg1>::make(base_matrix(std::forward<Arg1>(arg1)) - std::forward<Arg2>(arg2));
      if constexpr (not std::is_lvalue_reference_v<Arg1&&> or not std::is_lvalue_reference_v<Arg2&&>) return strict(std::move(ret)); else return ret;
    }
    else
    {
      auto ret = MatrixTraits<Arg2>::make(std::forward<Arg1>(arg1) - base_matrix(std::forward<Arg2>(arg2)));
      if constexpr (not std::is_lvalue_reference_v<Arg1&&> or not std::is_lvalue_reference_v<Arg2&>) return strict(std::move(ret)); else return ret;
    }
  }


#ifdef __cpp_concepts
  template<typename Arg1, typename Arg2> requires (Eigen3::eigen_triangular_expr<Arg1> and is_zero_v<Arg2>) or
      (is_zero_v<Arg1> and Eigen3::eigen_triangular_expr<Arg2>)
#else
  template<typename Arg1, typename Arg2,
    std::enable_if_t<(Eigen3::is_eigen_triangular_expr_v<Arg1> and is_zero_v<Arg2>) or
      (is_zero_v<Arg1> and Eigen3::is_eigen_triangular_expr_v<Arg2>), int> = 0>
#endif
  constexpr decltype(auto) operator-(Arg1&& arg1, Arg2&& arg2)
  {
    static_assert(MatrixTraits<Arg1>::dimension == MatrixTraits<Arg2>::dimension);
    static_assert(MatrixTraits<Arg1>::columns == MatrixTraits<Arg2>::columns);
    if constexpr(Eigen3::eigen_triangular_expr<Arg1>)
    {
      return std::forward<Arg1>(arg1);
    }
    else
    {
      return -std::forward<Arg2>(arg2);
    }
  }


  ////

#ifdef __cpp_concepts
  template<Eigen3::eigen_triangular_expr Arg, std::convertible_to<typename MatrixTraits<Arg>::Scalar> S>
#else
template<typename Arg, typename S, std::enable_if_t<Eigen3::is_eigen_triangular_expr_v<Arg> and
    std::is_convertible_v<S, typename MatrixTraits<Arg>::Scalar>, int> = 0>
#endif
  inline auto operator*(Arg&& arg, const S scale)
  {
    auto ret = MatrixTraits<Arg>::make(base_matrix(std::forward<Arg>(arg)) * scale);
    if constexpr (not std::is_lvalue_reference_v<Arg&&>) return strict(std::move(ret)); else return ret;
  }


#ifdef __cpp_concepts
  template<Eigen3::eigen_triangular_expr Arg, std::convertible_to<typename MatrixTraits<Arg>::Scalar> S>
#else
  template<typename Arg, typename S, std::enable_if_t<Eigen3::is_eigen_triangular_expr_v<Arg> and
    std::is_convertible_v<S, typename MatrixTraits<Arg>::Scalar>, int> = 0>
#endif
  inline auto operator*(const S scale, Arg&& arg)
  {
    auto ret = MatrixTraits<Arg>::make(scale * base_matrix(std::forward<Arg>(arg)));
    if constexpr (not std::is_lvalue_reference_v<Arg&&>) return strict(std::move(ret)); else return ret;
  }


#ifdef __cpp_concepts
  template<Eigen3::eigen_triangular_expr Arg, std::convertible_to<typename MatrixTraits<Arg>::Scalar> S>
#else
  template<typename Arg, typename S, std::enable_if_t<Eigen3::is_eigen_triangular_expr_v<Arg> and
    std::is_convertible_v<S, typename MatrixTraits<Arg>::Scalar>, int> = 0>
#endif
  inline auto operator/(Arg&& arg, const S scale)
  {
    auto ret = MatrixTraits<Arg>::make(base_matrix(std::forward<Arg>(arg)) / scale);
    if constexpr (not std::is_lvalue_reference_v<Arg&&>) return strict(std::move(ret)); else return ret;
  }


  ////

#ifdef __cpp_concepts
  template<Eigen3::eigen_triangular_expr Arg1, Eigen3::eigen_triangular_expr Arg2> requires
    (not is_identity_v<Arg1>) and (not is_identity_v<Arg2>) and (not is_zero_v<Arg1>) and (not is_zero_v<Arg2>)
#else
  template<typename Arg1, typename Arg2, std::enable_if_t<
    Eigen3::is_eigen_triangular_expr_v<Arg1> and Eigen3::is_eigen_triangular_expr_v<Arg2> and
      not is_identity_v<Arg1> and not is_identity_v<Arg2> and not is_zero_v<Arg1> and not is_zero_v<Arg2>, int> = 0>
#endif
  inline auto operator*(Arg1&& arg1, Arg2&& arg2)
  {
    static_assert(MatrixTraits<Arg1>::dimension == MatrixTraits<Arg2>::dimension);
    if constexpr(is_lower_triangular_v<Arg1> == is_lower_triangular_v<Arg2>)
    {
      auto ret = MatrixTraits<Arg1>::make(std::forward<Arg1>(arg1).base_view() * std::forward<Arg2>(arg2));
      if constexpr (not std::is_lvalue_reference_v<Arg1&&> or not std::is_lvalue_reference_v<Arg2&&>) return strict(std::move(ret)); else return ret;
    }
    else
    {
      auto ret = strict(std::forward<Arg1>(arg1).base_view() * strict_matrix(std::forward<Arg2>(arg2)));
      if constexpr (not std::is_lvalue_reference_v<Arg1&&> or not std::is_lvalue_reference_v<Arg2&&>) return strict(std::move(ret)); else return ret;
    }
  }


#ifdef __cpp_concepts
  template<typename Arg1, typename Arg2> requires ((Eigen3::eigen_triangular_expr<Arg1> and is_diagonal_v<Arg2>) or
      (is_diagonal_v<Arg1> and Eigen3::eigen_triangular_expr<Arg2>)) and
    (not is_identity_v<Arg1>) and (not is_identity_v<Arg2>) and (not is_zero_v<Arg1>) and (not is_zero_v<Arg2>)
#else
  template<typename Arg1, typename Arg2, std::enable_if_t<
    ((Eigen3::is_eigen_triangular_expr_v<Arg1> and is_diagonal_v<Arg2>) or
      (is_diagonal_v<Arg1> and Eigen3::is_eigen_triangular_expr_v<Arg2>)) and
    not is_identity_v<Arg1> and not is_identity_v<Arg2> and not is_zero_v<Arg1> and not is_zero_v<Arg2>, int> = 0>
#endif
  inline auto operator*(Arg1&& arg1, Arg2&& arg2)
  {
    static_assert(MatrixTraits<Arg1>::dimension == MatrixTraits<Arg2>::dimension);
    if constexpr(Eigen3::eigen_triangular_expr<Arg1>)
    {
      auto ret = MatrixTraits<Arg1>::make(std::forward<Arg1>(arg1).base_view() * std::forward<Arg2>(arg2));
      if constexpr (not std::is_lvalue_reference_v<Arg1&&> or not std::is_lvalue_reference_v<Arg2&&>) return strict(std::move(ret)); else return ret;
    }
    else
    {
      auto ret = MatrixTraits<Arg2>::make(std::forward<Arg1>(arg1) * std::forward<Arg2>(arg2).base_view());
      if constexpr (not std::is_lvalue_reference_v<Arg1&&> or not std::is_lvalue_reference_v<Arg2&&>) return strict(std::move(ret)); else return ret;
    }
  }


#ifdef __cpp_concepts
  template<typename Arg1, typename Arg2> requires (Eigen3::eigen_triangular_expr<Arg1> and is_identity_v<Arg2>) or
    (is_identity_v<Arg1> and Eigen3::eigen_triangular_expr<Arg2>)
#else
  template<typename Arg1, typename Arg2, std::enable_if_t<
    (Eigen3::is_eigen_triangular_expr_v<Arg1> and is_identity_v<Arg2>) or
      (is_identity_v<Arg1> and Eigen3::is_eigen_triangular_expr_v<Arg2>), int> = 0>
#endif
  inline auto operator*(Arg1&& arg1, Arg2&& arg2)
  {
    static_assert(MatrixTraits<Arg1>::dimension == MatrixTraits<Arg2>::dimension);
    static_assert(MatrixTraits<Arg1>::columns == MatrixTraits<Arg2>::columns);
    if constexpr(Eigen3::eigen_triangular_expr<Arg1>)
    {
      return std::forward<Arg1>(arg1);
    }
    else
    {
      return std::forward<Arg2>(arg2);
    }
  }


#ifdef __cpp_concepts
  template<typename Arg1, typename Arg2> requires (Eigen3::eigen_triangular_expr<Arg1> and is_zero_v<Arg2>) or
      (is_zero_v<Arg1> and Eigen3::eigen_triangular_expr<Arg2>)
#else
  template<typename Arg1, typename Arg2,
    std::enable_if_t<(Eigen3::is_eigen_triangular_expr_v<Arg1> and is_zero_v<Arg2>) or
      (is_zero_v<Arg1> and Eigen3::is_eigen_triangular_expr_v<Arg2>), int> = 0>
#endif
  constexpr decltype(auto) operator*(Arg1&& arg1, Arg2&& arg2)
  {
    static_assert(MatrixTraits<Arg1>::columns == MatrixTraits<Arg2>::dimension);
    constexpr auto rows = MatrixTraits<Arg1>::dimension;
    constexpr auto cols = MatrixTraits<Arg2>::columns;
    using B = strict_matrix_t<Arg1, rows, cols>;
    return Eigen3::ZeroMatrix<B>();
  }


  ////

#ifdef __cpp_concepts
  template<typename Arg1, typename Arg2>
  requires ((Eigen3::eigen_triangular_expr<Arg1> and Eigen3::eigen_matrix<Arg2>) and
    (Eigen3::eigen_matrix<Arg1> and Eigen3::eigen_triangular_expr<Arg2>)) and
    (not is_identity_v<Arg1>) and (not is_identity_v<Arg2>) and (not is_zero_v<Arg1>) and (not is_zero_v<Arg2>)
#else
  template<typename Arg1, typename Arg2,
    std::enable_if_t<((Eigen3::is_eigen_triangular_expr_v<Arg1> and Eigen3::is_eigen_matrix_v<Arg2>) and
      (Eigen3::is_eigen_matrix_v<Arg1> and Eigen3::is_eigen_triangular_expr_v<Arg2>)) and
      not is_identity_v<Arg1> and not is_identity_v<Arg2> and not is_zero_v<Arg1> and not is_zero_v<Arg2>, int> = 0>
#endif
  inline auto operator*(Arg1&& arg1, Arg2&& arg2)
  {
    static_assert(MatrixTraits<Arg1>::dimension == MatrixTraits<Arg2>::dimension);
    if constexpr(Eigen3::eigen_triangular_expr<Arg1>)
    {
      auto ret = std::forward<Arg1>(arg1).base_view() * std::forward<Arg2>(arg2);
      if constexpr (not std::is_lvalue_reference_v<Arg1&&> or not std::is_lvalue_reference_v<Arg2&&>) return strict(std::move(ret)); else return ret;
    }
    else
    {
      auto ret = std::forward<Arg1>(arg1) * std::forward<Arg2>(arg2).base_view();
      if constexpr (not std::is_lvalue_reference_v<Arg1&&> or not std::is_lvalue_reference_v<Arg2&&>) return strict(std::move(ret)); else return ret;
    }
  }


  ////

#ifdef __cpp_concepts
  template<Eigen3::eigen_triangular_expr Arg>
#else
  template<typename Arg, std::enable_if_t<Eigen3::is_eigen_triangular_expr_v<Arg>, int> = 0>
#endif
  inline auto operator-(Arg&& arg)
  {
    auto ret = MatrixTraits<Arg>::make(-base_matrix(std::forward<Arg>(arg)));
    if constexpr (not std::is_lvalue_reference_v<Arg&&>) return strict(std::move(ret)); else return ret;
  }

} // namespace OpenKalman::Eigen3

#endif //OPENKALMAN_EIGEN3_TRIANGULARMATRIX_HPP
