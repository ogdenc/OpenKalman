/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef OPENKALMAN_EIGENTRIANGULARMATRIX_H
#define OPENKALMAN_EIGENTRIANGULARMATRIX_H

namespace OpenKalman
{
  template<typename BaseMatrix, TriangleType triangle_type>
  struct EigenTriangularMatrix
    : internal::MatrixBase<EigenTriangularMatrix<BaseMatrix, triangle_type>, BaseMatrix>
  {
    using Base = internal::MatrixBase<EigenTriangularMatrix, BaseMatrix>;
    static constexpr auto uplo = triangle_type == TriangleType::upper ? Eigen::Upper : Eigen::Lower;
    using View = Eigen::TriangularView<std::remove_reference_t<BaseMatrix>, uplo>;
    using Scalar = typename MatrixTraits<BaseMatrix>::Scalar;
    static constexpr auto dimension = MatrixTraits<BaseMatrix>::dimension;
    static_assert(dimension == MatrixTraits<BaseMatrix>::columns);

    /// Default constructor
    EigenTriangularMatrix() : Base(), view(this->base_matrix()) {}

    /// Copy constructor
    EigenTriangularMatrix(const EigenTriangularMatrix& other) : EigenTriangularMatrix(other.base_matrix()) {}

    /// Move constructor
    EigenTriangularMatrix(EigenTriangularMatrix&& other) noexcept
      : EigenTriangularMatrix(std::move(other).base_matrix()) {}

    /// Construct from a compatible triangular matrix object of the same TriangleType.
    template<typename Arg, std::enable_if_t<is_EigenTriangularMatrix_v<Arg>, int> = 0>
    EigenTriangularMatrix(Arg&& arg) noexcept : EigenTriangularMatrix(base_matrix(std::forward<Arg>(arg)))
    {
      static_assert(is_lower_triangular_v<Arg> == is_lower_triangular_v<EigenTriangularMatrix>);
    }

    /// Construct from a compatible self-adjoint matrix object
    template<typename Arg, std::enable_if_t<is_EigenSelfAdjointMatrix_v<Arg>, int> = 0>
    EigenTriangularMatrix(Arg&& arg) noexcept
      : EigenTriangularMatrix(Cholesky_factor<triangle_type>(std::forward<Arg>(arg))) {}

    /// Construct from a reference to a regular or diagonal matrix object
    template<typename Arg, std::enable_if_t<is_Eigen_matrix_v<Arg> or is_EigenDiagonal_v<Arg>, int> = 0>
    EigenTriangularMatrix(Arg&& arg) noexcept : Base(std::forward<Arg>(arg)), view(this->base_matrix()) {}

    /// Construct from a list of scalar coefficients, in row-major order. Only reads the triangular part.
    template<typename ... Args, std::enable_if_t<std::conjunction_v<std::is_convertible<Args, const Scalar>...>, int> = 0>
    EigenTriangularMatrix(Args ... args) : EigenTriangularMatrix(MatrixTraits<BaseMatrix>::make(args...)) {}

    /// Copy assignment operator
    auto& operator=(const EigenTriangularMatrix& other)
    {
      if (this != &other) this->base_matrix().template triangularView<uplo>() = other.base_matrix();
      return *this;
    }

    /// Move assignment operator
    auto& operator=(EigenTriangularMatrix&& other) noexcept
    {
      if (this != &other) this->base_matrix() = std::move(other).base_matrix();
      return *this;
    }


    /// Assign from another triangular matrix (must be the same triangle)
    template<typename Arg, std::enable_if_t<is_EigenTriangularMatrix_v<Arg>, int> = 0>
    auto& operator=(Arg&& arg)
    {
      static_assert(is_upper_triangular_v<Arg> == is_upper_triangular_v<EigenTriangularMatrix>);
      if constexpr (std::is_same_v<std::decay_t<Arg>, EigenTriangularMatrix>) if (this == &arg) return *this;
      if constexpr(std::is_lvalue_reference_v<Arg>)
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
    template<typename Arg, std::enable_if_t<is_EigenSelfAdjointMatrix_v<Arg>, int> = 0>
    auto& operator=(Arg&& arg)
    {
      this->base_matrix().template triangularView<uplo>() =
        base_matrix(Cholesky_factor<triangle_type>(std::forward<Arg>(arg)));
      return *this;
    }

    /// Assign from an Eigen::MatrixBase derived object. (Only uses the triangular part.)
    template<typename Arg, std::enable_if_t<is_Eigen_matrix_v<Arg>, int> = 0>
    auto& operator=(Arg&& arg)
    {
      if constexpr(std::is_lvalue_reference_v<Arg>)
      {
        this->base_matrix().template triangularView<uplo>() = arg;
      }
      else
        this->base_matrix() = std::forward<Arg>(arg);
      return *this;
    }

    /// Assign by copying from an Eigen::TriangularBase derived object.
    template<typename Arg, unsigned int UpLo>
    auto& operator=(const Eigen::TriangularView<Arg, UpLo>& arg)
    {
      if constexpr(((UpLo & Eigen::Lower) != 0) == (triangle_type == TriangleType::lower))
        this->base_matrix().template triangularView<UpLo>() = arg.nestedExpression();
      else
        this->base_matrix().template triangularView<UpLo>() = arg.nestedExpression().adjoint();
      return *this;
    }

    /// Assign by moving from an Eigen::TriangularBase derived object.
    template<typename Arg>
    auto& operator=(Eigen::TriangularView<Arg, uplo>&& arg) noexcept
    {
      this->base_matrix() = std::move(arg.nestedExpression());
      return *this;
    }

    template<typename Arg>
    auto& operator+=(const EigenTriangularMatrix<Arg, triangle_type>& arg)
    {
      static_assert(MatrixTraits<Arg>::dimension == dimension);
      this->base_view() += arg.base_matrix();
      return *this;
    }

    template<typename Arg>
    auto& operator-=(const EigenTriangularMatrix<Arg, triangle_type>& arg)
    {
      static_assert(MatrixTraits<Arg>::dimension == dimension);
      this->base_view() -= arg.base_matrix();
      return *this;
    }

    template<typename S, std::enable_if_t<std::is_convertible_v<S, Scalar>, int> = 0>
    auto& operator*=(const S s)
    {
      this->base_view() *= s;
      return *this;
    }

    template<typename S, std::enable_if_t<std::is_convertible_v<S, Scalar>, int> = 0>
    auto& operator/=(const S s)
    {
      this->base_view() /= s;
      return *this;
    }

    template<typename Arg>
    auto& operator*=(const EigenTriangularMatrix<Arg, triangle_type>& arg)
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
      if constexpr (is_element_settable_v<EigenTriangularMatrix, 2>)
        return internal::ElementSetter(*this, i, j);
      else
        return const_cast<const EigenTriangularMatrix&>(*this)(i, j);
    }

    auto operator()(std::size_t i, std::size_t j) const noexcept { return internal::ElementSetter(*this, i, j); }

    auto operator[](std::size_t i)
    {
      if constexpr(is_element_gettable_v<EigenTriangularMatrix, 1>)
        return internal::ElementSetter(*this, i);
      else if constexpr(is_element_gettable_v<EigenTriangularMatrix, 2>)
        return internal::ElementSetter(*this, i, i);
      else
        return const_cast<const EigenTriangularMatrix&>(*this)[i];
    }

    auto operator[](std::size_t i) const
    {
      if constexpr(is_element_gettable_v<EigenTriangularMatrix, 1>)
        return internal::ElementSetter(*this, i);
      else
        return internal::ElementSetter(*this, i, i);
    }

    auto operator()(std::size_t i) { return operator[](i); }

    auto operator()(std::size_t i) const { return operator[](i); }


    template<typename B, std::enable_if_t<is_Eigen_matrix_v<B>, int> = 0>
    auto solve(const B& b) const
    {
      return this->base_view().solve(b);
    }

  private:
    View view;
  };


  /////////////////////////////////////
  //        Deduction Guides         //
  /////////////////////////////////////

  template<typename M, std::enable_if_t<is_Eigen_matrix_v<M> or is_EigenDiagonal_v<M>, int> = 0>
  EigenTriangularMatrix(M&&) -> EigenTriangularMatrix<std::decay_t<M>, TriangleType::lower>;

  template<typename M, std::enable_if_t<(is_EigenTriangularMatrix_v<M> and is_lower_triangular_v<M>) or
    (is_EigenSelfAdjointMatrix_v<M> and is_Eigen_lower_storage_triangle_v<M>), int> = 0>
  EigenTriangularMatrix(M&&)
    -> EigenTriangularMatrix<typename MatrixTraits<M>::BaseMatrix, TriangleType::lower>;

  template<typename M, std::enable_if_t<(is_EigenTriangularMatrix_v<M> and is_upper_triangular_v<M>) or
    (is_EigenSelfAdjointMatrix_v<M> and is_Eigen_upper_storage_triangle_v<M>), int> = 0>
  EigenTriangularMatrix(M&&)
    -> EigenTriangularMatrix<typename MatrixTraits<M>::BaseMatrix, TriangleType::upper>;

  template<typename Arg, unsigned int UpLo>
  EigenTriangularMatrix(const Eigen::TriangularView<Arg, UpLo>&)
    -> EigenTriangularMatrix<Arg, UpLo & Eigen::Upper ? TriangleType::upper : TriangleType::lower>;

  /// If the arguments are a sequence of scalars, deduce a square, lower triangular matrix.
  template<typename ... Args, std::enable_if_t<std::conjunction_v<std::is_arithmetic<Args>...>, int> = 0>
  EigenTriangularMatrix(Args ...)
    -> EigenTriangularMatrix<
    Eigen::Matrix<std::decay_t<std::common_type_t<Args...>>, internal::constexpr_sqrt(sizeof...(Args)),
      internal::constexpr_sqrt(sizeof...(Args))>, TriangleType::lower>;


  ///////////////////////////
  //        Traits         //
  ///////////////////////////

  template<typename ArgType, TriangleType triangle>
  struct MatrixTraits<OpenKalman::EigenTriangularMatrix<ArgType, triangle>>
  {
    static constexpr TriangleType triangle_type = triangle;
    using BaseMatrix = ArgType;
    using Scalar = typename MatrixTraits<BaseMatrix>::Scalar;
    static constexpr auto dimension = MatrixTraits<BaseMatrix>::dimension;
    static constexpr auto columns = MatrixTraits<BaseMatrix>::columns;
    static_assert(dimension == columns, "A triangular matrix must be square.");

    template<typename Derived>
    using MatrixBaseType = internal::EigenMatrixBase<Derived, OpenKalman::EigenTriangularMatrix<std::decay_t<BaseMatrix>, triangle_type>>;

    template<typename Derived>
    using CovarianceBaseType = internal::EigenCovarianceBase<Derived, EigenTriangularMatrix<std::decay_t<BaseMatrix>, triangle_type>>;

    template<std::size_t rows = dimension, std::size_t cols = dimension, typename S = Scalar>
    using StrictMatrix = typename MatrixTraits<BaseMatrix>::template StrictMatrix<rows, cols, S>;

    template<TriangleType t = triangle_type, std::size_t dim = dimension, typename S = Scalar>
    using SelfAdjointBaseType = EigenSelfAdjointMatrix<StrictMatrix<dim, dim, S>, t>;

    template<TriangleType t = triangle_type, std::size_t dim = dimension, typename S = Scalar>
    using TriangularBaseType = EigenTriangularMatrix<StrictMatrix<dim, dim, S>, t>;

    template<std::size_t dim = dimension, typename S = Scalar>
    using DiagonalBaseType = EigenDiagonal<StrictMatrix<dim, 1, S>>;

    template<typename Arg, std::enable_if_t<is_Eigen_matrix_v<Arg>, int> = 0>
    static auto make(Arg&& arg) noexcept
    {
      return EigenTriangularMatrix<std::decay_t<Arg>, triangle_type>(std::forward<Arg>(arg));
    }

    /// Make triangular matrix using a list of coefficients in row-major order.
    /// Only the coefficients in the lower-left corner are significant.
    template<typename ... Args,
      std::enable_if_t<std::conjunction_v<std::is_convertible<Args, const Scalar>...>, int> = 0>
    static auto make(Args ... args)
    {
      static_assert(sizeof...(Args) == dimension * dimension);
      return make(MatrixTraits<BaseMatrix>::make(args...));
    }

    static auto zero() { return MatrixTraits<BaseMatrix>::zero(); }

    static auto identity() { return MatrixTraits<BaseMatrix>::identity(); }
  };


  //////////////////////////////
  //        Overloads         //
  //////////////////////////////

  template<typename Arg, std::enable_if_t<is_EigenTriangularMatrix_v<Arg>, int> = 0>
  constexpr auto
  transpose(Arg&& arg)
  {
    constexpr auto t = triangle_type_of_v<Arg> == TriangleType::lower ? TriangleType::upper :
                       triangle_type_of_v<Arg> == TriangleType::upper ? TriangleType::lower : TriangleType::diagonal;
    auto b = base_matrix(std::forward<Arg>(arg)).transpose();
    return EigenTriangularMatrix<decltype(b), t>(std::move(b));
  }


  template<typename Arg, std::enable_if_t<is_EigenTriangularMatrix_v<Arg>, int> = 0>
  constexpr auto
  adjoint(Arg&& arg)
  {
    constexpr auto t = triangle_type_of_v<Arg> == TriangleType::lower ? TriangleType::upper :
                       triangle_type_of_v<Arg> == TriangleType::upper ? TriangleType::lower : TriangleType::diagonal;
    auto b = base_matrix(std::forward<Arg>(arg)).adjoint();
    return EigenTriangularMatrix<decltype(b), t>(std::move(b));
  }


  template<typename Arg, typename U,
    std::enable_if_t<is_EigenTriangularMatrix_v<Arg> and
      (is_Eigen_matrix_v<U> or is_EigenTriangularMatrix_v<U> or is_EigenSelfAdjointMatrix_v<U> or is_EigenDiagonal_v<U>) and
      not std::is_const_v<std::remove_reference_t<Arg>>, int> = 0>
  inline Arg&
  rank_update(Arg& arg, const U& u, const typename MatrixTraits<Arg>::Scalar alpha = 1)
  {
    static_assert(MatrixTraits<U>::dimension == MatrixTraits<Arg>::dimension);
    using Scalar = typename MatrixTraits<Arg>::Scalar;
    constexpr auto t = is_lower_triangular_v<Arg> ? Eigen::Lower : Eigen::Upper;
    for (Eigen::Index i = 0; i < MatrixTraits<U>::columns; ++i)
    {
      if (Eigen::internal::llt_inplace<Scalar, t>::rankUpdate(arg.base_matrix(), u.col(i), alpha) >= 0)
      {
        throw (std::runtime_error("EigenTriangularMatrix rank_update: product is not positive definite"));
      }
    }
    return arg;
  }


  template<typename Arg, typename U,
    std::enable_if_t<is_EigenTriangularMatrix_v<Arg> and
      (is_Eigen_matrix_v<U> or is_EigenTriangularMatrix_v<U> or is_EigenSelfAdjointMatrix_v<U> or is_EigenDiagonal_v<U>), int> = 0>
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
  template<typename A, std::enable_if_t<is_EigenTriangularMatrix_v<A>, int> = 0>
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
  template<typename A, std::enable_if_t<is_EigenTriangularMatrix_v<A>, int> = 0>
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

  template<
    typename Arg1, typename Arg2,
    std::enable_if_t<is_EigenTriangularMatrix_v<Arg1> and is_EigenTriangularMatrix_v<Arg2> and
      not is_diagonal_v<Arg1> and not is_diagonal_v<Arg2>, int> = 0>
  inline auto operator+(Arg1&& arg1, Arg2&& arg2)
  {
    static_assert(MatrixTraits<Arg1>::dimension == MatrixTraits<Arg2>::dimension);
    if constexpr(is_lower_triangular_v<Arg1> == is_lower_triangular_v<Arg2>)
      return MatrixTraits<Arg1>::make(base_matrix(std::forward<Arg1>(arg1)) + base_matrix(std::forward<Arg2>(arg2)));
    else
    {
      auto ret = strict_matrix(std::forward<Arg1>(arg1));
      constexpr auto mode = is_upper_triangular_v<Arg2> ? Eigen::Upper : Eigen::Lower;
      ret.template triangularView<mode>() += base_matrix(std::forward<Arg2>(arg2));
      return ret;
    }
  }


  template<typename Arg1, typename Arg2,
    std::enable_if_t<((is_EigenTriangularMatrix_v<Arg1> and is_diagonal_v<Arg2>) or
      (is_diagonal_v<Arg1> and is_EigenTriangularMatrix_v<Arg2>)) and
      not is_zero_v<Arg1> and not is_zero_v<Arg2> and not is_identity_v<Arg1> and not is_identity_v<Arg2>, int> = 0>
  inline auto operator+(Arg1&& arg1, Arg2&& arg2)
  {
    static_assert(MatrixTraits<Arg1>::dimension == MatrixTraits<Arg2>::dimension);
    if constexpr(is_EigenTriangularMatrix_v<Arg1>)
    {
      return MatrixTraits<Arg1>::make(base_matrix(std::forward<Arg1>(arg1)) + std::forward<Arg2>(arg2));
    }
    else
    {
      return MatrixTraits<Arg2>::make(std::forward<Arg1>(arg1) + base_matrix(std::forward<Arg2>(arg2)));
    }
  }


  template<typename Arg1, typename Arg2,
    std::enable_if_t<(is_EigenTriangularMatrix_v<Arg1> and is_identity_v<Arg2>) or
      (is_identity_v<Arg1> and is_EigenTriangularMatrix_v<Arg2>), int> = 0>
  inline auto operator+(Arg1&& arg1, Arg2&& arg2)
  {
    static_assert(MatrixTraits<Arg1>::dimension == MatrixTraits<Arg2>::dimension);
    static_assert(MatrixTraits<Arg1>::columns == MatrixTraits<Arg2>::columns);
    if constexpr(OpenKalman::is_EigenTriangularMatrix_v<Arg1>)
    {
      using B = typename MatrixTraits<Arg1>::BaseMatrix;
      return MatrixTraits<Arg1>::make(base_matrix(std::forward<Arg1>(arg1)) + std::forward<Arg2>(arg2));
    }
    else
    {
      using B = typename MatrixTraits<Arg2>::BaseMatrix;
      return MatrixTraits<Arg2>::make(std::forward<Arg1>(arg1) + base_matrix(std::forward<Arg2>(arg2)));
    }
  }


  template<typename Arg1, typename Arg2,
    std::enable_if_t<(is_EigenTriangularMatrix_v<Arg1> and is_zero_v<Arg2>) or
      (is_zero_v<Arg1> and is_EigenTriangularMatrix_v<Arg2>), int> = 0>
  constexpr decltype(auto) operator+(Arg1&& arg1, Arg2&& arg2)
  {
    static_assert(MatrixTraits<Arg1>::dimension == MatrixTraits<Arg2>::dimension);
    static_assert(MatrixTraits<Arg1>::columns == MatrixTraits<Arg2>::columns);
    if constexpr(OpenKalman::is_EigenTriangularMatrix_v<Arg1>)
    {
      return std::forward<Arg1>(arg1);
    }
    else
    {
      return std::forward<Arg2>(arg2);
    }
  }


  ////

  template<
    typename Arg1, typename Arg2,
    std::enable_if_t<is_EigenTriangularMatrix_v<Arg1> and is_EigenTriangularMatrix_v<Arg2>, int> = 0>
  inline auto operator-(Arg1&& arg1, Arg2&& arg2)
  {
    static_assert(MatrixTraits<Arg1>::dimension == MatrixTraits<Arg2>::dimension);
    if constexpr(is_lower_triangular_v<Arg1> == is_lower_triangular_v<Arg2>)
      return MatrixTraits<Arg1>::make(base_matrix(std::forward<Arg1>(arg1)) - base_matrix(std::forward<Arg2>(arg2)));
    else
    {
      auto ret = strict_matrix(std::forward<Arg1>(arg1));
      constexpr auto mode = is_upper_triangular_v<Arg2> ? Eigen::Upper : Eigen::Lower;
      ret.template triangularView<mode>() -= base_matrix(std::forward<Arg2>(arg2));
      return ret;
    }
  }


  template<typename Arg1, typename Arg2,
    std::enable_if_t<((is_EigenTriangularMatrix_v<Arg1> and is_diagonal_v<Arg2>) or
      (is_diagonal_v<Arg1> and is_EigenTriangularMatrix_v<Arg2>)) and
      not is_zero_v<Arg1> and not is_zero_v<Arg2> and not is_identity_v<Arg1> and not is_identity_v<Arg2>, int> = 0>
  inline auto operator-(Arg1&& arg1, Arg2&& arg2)
  {
    static_assert(MatrixTraits<Arg1>::dimension == MatrixTraits<Arg2>::dimension);
    if constexpr(is_EigenTriangularMatrix_v<Arg1>)
    {
      return MatrixTraits<Arg1>::make(base_matrix(std::forward<Arg1>(arg1)) - std::forward<Arg2>(arg2));
    }
    else
    {
      return MatrixTraits<Arg2>::make(std::forward<Arg1>(arg1) - base_matrix(std::forward<Arg2>(arg2)));
    }
  }


  template<typename Arg1, typename Arg2,
    std::enable_if_t<(is_EigenTriangularMatrix_v<Arg1> and is_identity_v<Arg2>) or
      (is_identity_v<Arg1> and is_EigenTriangularMatrix_v<Arg2>), int> = 0>
  inline auto operator-(Arg1&& arg1, Arg2&& arg2)
  {
    static_assert(MatrixTraits<Arg1>::dimension == MatrixTraits<Arg2>::dimension);
    static_assert(MatrixTraits<Arg1>::columns == MatrixTraits<Arg2>::columns);
    if constexpr(OpenKalman::is_EigenTriangularMatrix_v<Arg1>)
    {
      using B = typename MatrixTraits<Arg1>::BaseMatrix;
      return MatrixTraits<Arg1>::make(base_matrix(std::forward<Arg1>(arg1)) - std::forward<Arg2>(arg2));
    }
    else
    {
      using B = typename MatrixTraits<Arg2>::BaseMatrix;
      return MatrixTraits<Arg2>::make(std::forward<Arg1>(arg1) - base_matrix(std::forward<Arg2>(arg2)));
    }
  }


  template<typename Arg1, typename Arg2,
    std::enable_if_t<(is_EigenTriangularMatrix_v<Arg1> and is_zero_v<Arg2>) or
      (is_zero_v<Arg1> and is_EigenTriangularMatrix_v<Arg2>), int> = 0>
  constexpr decltype(auto) operator-(Arg1&& arg1, Arg2&& arg2)
  {
    static_assert(MatrixTraits<Arg1>::dimension == MatrixTraits<Arg2>::dimension);
    static_assert(MatrixTraits<Arg1>::columns == MatrixTraits<Arg2>::columns);
    if constexpr(OpenKalman::is_EigenTriangularMatrix_v<Arg1>)
    {
      return std::forward<Arg1>(arg1);
    }
    else
    {
      return -std::forward<Arg2>(arg2);
    }
  }


  ////

  template<
    typename Arg, typename S,
    std::enable_if_t<is_EigenTriangularMatrix_v<Arg> and
      std::is_convertible_v<S, typename MatrixTraits<Arg>::Scalar>, int> = 0>
  inline auto operator*(Arg&& arg, const S scale)
  {
    return MatrixTraits<Arg>::make(base_matrix(std::forward<Arg>(arg)) * scale);
  }


  template<
    typename Arg, typename S,
    std::enable_if_t<is_EigenTriangularMatrix_v<Arg> and
      std::is_convertible_v<S, typename MatrixTraits<Arg>::Scalar>, int> = 0>
  inline auto operator*(const S scale, Arg&& arg)
  {
    return MatrixTraits<Arg>::make(scale * base_matrix(std::forward<Arg>(arg)));
  }


  template<
    typename Arg, typename S,
    std::enable_if_t<is_EigenTriangularMatrix_v<Arg> and
      std::is_convertible_v<S, typename MatrixTraits<Arg>::Scalar>, int> = 0>
  inline auto operator/(Arg&& arg, const S scale)
  {
    return MatrixTraits<Arg>::make(base_matrix(std::forward<Arg>(arg)) / scale);
  }


  ////

  template<
    typename Arg1, typename Arg2,
    std::enable_if_t<is_EigenTriangularMatrix_v<Arg1> and is_EigenTriangularMatrix_v<Arg2> and
      not is_identity_v<Arg1> and not is_identity_v<Arg2> and not is_zero_v<Arg1> and not is_zero_v<Arg2>, int> = 0>
  inline auto operator*(Arg1&& arg1, Arg2&& arg2)
  {
    static_assert(MatrixTraits<Arg1>::dimension == MatrixTraits<Arg2>::dimension);
    if constexpr(is_lower_triangular_v<Arg1> == is_lower_triangular_v<Arg2>)
    {
      return MatrixTraits<Arg1>::make(std::forward<Arg1>(arg1).base_view() * std::forward<Arg2>(arg2));
    }
    else
    {
      return strict(std::forward<Arg1>(arg1).base_view() * strict_matrix(std::forward<Arg2>(arg2)));
    }
  }


  template<
    typename Arg1, typename Arg2,
    std::enable_if_t<((is_EigenTriangularMatrix_v<Arg1> and is_diagonal_v<Arg2>) or
      (is_diagonal_v<Arg1> and is_EigenTriangularMatrix_v<Arg2>)) and
      not is_identity_v<Arg1> and not is_identity_v<Arg2> and not is_zero_v<Arg1> and not is_zero_v<Arg2>, int> = 0>
  inline auto operator*(Arg1&& arg1, Arg2&& arg2)
  {
    static_assert(MatrixTraits<Arg1>::dimension == MatrixTraits<Arg2>::dimension);
    if constexpr(is_EigenTriangularMatrix_v<Arg1>)
    {
      return MatrixTraits<Arg1>::make(std::forward<Arg1>(arg1).base_view() * std::forward<Arg2>(arg2));
    }
    else
    {
      return MatrixTraits<Arg2>::make(std::forward<Arg1>(arg1) * std::forward<Arg2>(arg2).base_view());
    }
  }


  template<typename Arg1, typename Arg2,
    std::enable_if_t<(is_EigenTriangularMatrix_v<Arg1> and is_identity_v<Arg2>) or
      (is_identity_v<Arg1> and is_EigenTriangularMatrix_v<Arg2>), int> = 0>
  inline auto operator*(Arg1&& arg1, Arg2&& arg2)
  {
    static_assert(MatrixTraits<Arg1>::dimension == MatrixTraits<Arg2>::dimension);
    static_assert(MatrixTraits<Arg1>::columns == MatrixTraits<Arg2>::columns);
    if constexpr(OpenKalman::is_EigenTriangularMatrix_v<Arg1>)
    {
      return std::forward<Arg1>(arg1);
    }
    else
    {
      return std::forward<Arg2>(arg2);
    }
  }


  template<typename Arg1, typename Arg2,
    std::enable_if_t<(is_EigenTriangularMatrix_v<Arg1> and is_zero_v<Arg2>) or
      (is_zero_v<Arg1> and is_EigenTriangularMatrix_v<Arg2>), int> = 0>
  constexpr decltype(auto) operator*(Arg1&& arg1, Arg2&& arg2)
  {
    static_assert(MatrixTraits<Arg1>::columns == MatrixTraits<Arg2>::dimension);
    constexpr auto rows = MatrixTraits<Arg1>::dimension;
    constexpr auto cols = MatrixTraits<Arg2>::columns;
    using B = typename MatrixTraits<Arg1>::template StrictMatrix<rows, cols>;
    return EigenZero<B>();
  }


  ////

  template<
    typename Arg1, typename Arg2,
    std::enable_if_t<((is_EigenTriangularMatrix_v<Arg1> and is_Eigen_matrix_v<Arg2>) and
      (is_Eigen_matrix_v<Arg1> and is_EigenTriangularMatrix_v<Arg2>)) and
      not is_identity_v<Arg1> and not is_identity_v<Arg2> and not is_zero_v<Arg1> and not is_zero_v<Arg2>, int> = 0>
  inline auto operator*(Arg1&& arg1, Arg2&& arg2)
  {
    static_assert(MatrixTraits<Arg1>::dimension == MatrixTraits<Arg2>::dimension);
    if constexpr(is_EigenTriangularMatrix_v<Arg1>)
    {
      return std::forward<Arg1>(arg1).base_view() * std::forward<Arg2>(arg2);
    }
    else
    {
      return std::forward<Arg1>(arg1) * std::forward<Arg2>(arg2).base_view();
    }
  }


  ////

  template<
    typename Arg,
    std::enable_if_t<is_EigenTriangularMatrix_v<Arg>, int> = 0>
  inline auto operator-(Arg&& arg)
  {
    return MatrixTraits<Arg>::make(-base_matrix(std::forward<Arg>(arg)));
  }


}

#endif //OPENKALMAN_EIGENTRIANGULARMATRIX_H
