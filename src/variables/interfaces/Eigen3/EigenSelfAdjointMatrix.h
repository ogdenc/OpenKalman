/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef OPENKALMAN_EIGENSELFADJOINTMATRIX_H
#define OPENKALMAN_EIGENSELFADJOINTMATRIX_H

namespace OpenKalman
{
  template<typename BaseMatrix, TriangleType storage_triangle>
  struct EigenSelfAdjointMatrix
    : internal::MatrixBase<EigenSelfAdjointMatrix<BaseMatrix, storage_triangle>, BaseMatrix>
  {
    using Base = internal::MatrixBase<EigenSelfAdjointMatrix, BaseMatrix>;
    static constexpr auto uplo = storage_triangle == TriangleType::upper ? Eigen::Upper : Eigen::Lower;
    using View = Eigen::SelfAdjointView<std::remove_reference_t<BaseMatrix>, uplo>;
    using Scalar = typename MatrixTraits<BaseMatrix>::Scalar;
    static constexpr auto dimension = MatrixTraits<BaseMatrix>::dimension;
    static_assert(dimension == MatrixTraits<BaseMatrix>::columns);

    /// Default constructor
    EigenSelfAdjointMatrix() : Base(), view(this->base_matrix()) {}

    /// Copy constructor.
    EigenSelfAdjointMatrix(const EigenSelfAdjointMatrix& other) : EigenSelfAdjointMatrix(other.base_matrix()) {}

    /// Move constructor.
    EigenSelfAdjointMatrix(EigenSelfAdjointMatrix&& other) noexcept : EigenSelfAdjointMatrix(std::move(other.base_matrix())) {}

    /// Construct from a compatible self-joint matrix object of the same storage type
    template<typename Arg, std::enable_if_t<is_EigenSelfAdjointMatrix_v<Arg>, int> = 0,
    std::enable_if_t<is_Eigen_lower_storage_triangle_v<Arg> == is_Eigen_lower_storage_triangle_v<EigenSelfAdjointMatrix>, int> = 0>
    EigenSelfAdjointMatrix(Arg&& arg) noexcept : EigenSelfAdjointMatrix(base_matrix(std::forward<Arg>(arg))) {}

    /// Construct from a compatible self-joint matrix object of the opposite storage type
    template<typename Arg, std::enable_if_t<is_EigenSelfAdjointMatrix_v<Arg>, int> = 0,
    std::enable_if_t<is_Eigen_lower_storage_triangle_v<Arg> != is_Eigen_lower_storage_triangle_v<EigenSelfAdjointMatrix>, int> = 0>
    EigenSelfAdjointMatrix(Arg&& arg) noexcept : EigenSelfAdjointMatrix(adjoint(base_matrix(std::forward<Arg>(arg)))) {}

    /// Construct from a compatible triangular matrix object
    template<typename Arg, std::enable_if_t<is_EigenTriangularMatrix_v<Arg>, int> = 0>
    EigenSelfAdjointMatrix(Arg&& arg) noexcept
      : EigenSelfAdjointMatrix(Cholesky_square(std::forward<Arg>(arg))) {}

    /// Construct from a regular or diagonal matrix object.
    template<typename Arg, std::enable_if_t<is_Eigen_matrix_v<Arg> or is_EigenDiagonal_v<Arg>, int> = 0>
    EigenSelfAdjointMatrix(Arg&& arg) noexcept : Base(std::forward<Arg>(arg)), view(this->base_matrix()) {}

    /// Construct from a list of scalar coefficients, in row-major order.
    template<typename ... Args, std::enable_if_t<std::conjunction_v<std::is_convertible<Args, const Scalar>...> and
      ((not is_EigenDiagonal_v<BaseMatrix> and sizeof...(Args) == dimension * dimension and storage_triangle != TriangleType::diagonal) or
      (is_EigenDiagonal_v<BaseMatrix> and sizeof...(Args) == dimension)), int> = 0>
    EigenSelfAdjointMatrix(Args ... args) : EigenSelfAdjointMatrix(MatrixTraits<BaseMatrix>::make(args...)) {}

    /// Construct from a list of scalar coefficients, in row-major order.
    template<typename ... Args, std::enable_if_t<std::conjunction_v<std::is_convertible<Args, const Scalar>...> and
      not is_EigenDiagonal_v<BaseMatrix> and sizeof...(Args) == dimension and storage_triangle == TriangleType::diagonal, int> = 0>
    EigenSelfAdjointMatrix(Args ... args)
      : EigenSelfAdjointMatrix(strict_matrix(EigenDiagonal {static_cast<const Scalar>(args)...})) {}

    /// Copy assignment operator
    auto& operator=(const EigenSelfAdjointMatrix& other)
    {
      if constexpr (not is_zero_v<BaseMatrix> and not is_identity_v<BaseMatrix>) if (this != &other)
      {
        this->base_matrix().template triangularView<uplo>() = other.base_matrix();
      }
      return *this;
    }

    /// Move assignment operator
    auto& operator=(EigenSelfAdjointMatrix&& other) noexcept
    {
      if constexpr (not is_zero_v<BaseMatrix> and not is_identity_v<BaseMatrix>) if (this != &other)
      {
        this->base_matrix() = std::move(other).base_matrix();
      }
      return *this;
    }

    /// Assign from another self-adjoint special matrix
    template<typename Arg, std::enable_if_t<is_EigenSelfAdjointMatrix_v<Arg>, int> = 0>
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
        if constexpr(is_Eigen_upper_storage_triangle_v<Arg> == is_Eigen_upper_storage_triangle_v<EigenSelfAdjointMatrix>)
          this->base_matrix().template triangularView<uplo>() = base_matrix(arg);
        else
          this->base_matrix().template triangularView<uplo>() = adjoint(base_matrix(arg));
      }
      else
      {
        this->base_matrix() = std::forward<Arg>(arg);
      }
      return *this;
    }

    /// Assign by converting from a triangular special matrix
    template<typename Arg, std::enable_if_t<is_EigenTriangularMatrix_v<Arg>, int> = 0>
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
      else
      {
        return operator=(Cholesky_square(std::forward<Arg>(arg)));
      }
    }

    /// Assign from a regular Eigen matrix. (Uses only the storage triangular part.)
    template<typename Arg, std::enable_if_t<is_Eigen_matrix_v<Arg>, int> = 0>
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

    /// Assign by copying from an Eigen::SelfAdjointView object.
    template<typename Arg, unsigned int UpLo>
    auto& operator=(const Eigen::SelfAdjointView<Arg, UpLo>& arg)
    {
      if constexpr (is_zero_v<BaseMatrix>)
      {
        static_assert(is_zero_v<Arg>);
      }
      else if constexpr (is_identity_v<BaseMatrix>)
      {
        static_assert(is_identity_v<Arg>);
      }
      else if constexpr(((UpLo & Eigen::Upper) != 0) != (storage_triangle == TriangleType::upper))
      {
        this->base_matrix().template triangularView<UpLo>() = arg.nestedExpression().adjoint();
      }
      else
      {
        this->base_matrix().template triangularView<UpLo>() = arg.nestedExpression();
      }
      return *this;
    }

    /// Assign by moving from an Eigen::SelfAdjointView object.
    template<typename Arg>
    auto& operator=(Eigen::SelfAdjointView<Arg, uplo>&& arg) noexcept
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

    template<typename Arg, TriangleType t>
    auto& operator+=(const EigenSelfAdjointMatrix<Arg, t>& arg)
    {
      static_assert(MatrixTraits<Arg>::dimension == dimension);
      if constexpr(t == storage_triangle)
        this->base_matrix().template triangularView<uplo>() += arg.base_matrix();
      else
        this->base_matrix().template triangularView<uplo>() += arg.base_matrix().adjoint();
      return *this;
    }

    template<typename Arg, TriangleType t>
    auto& operator-=(const EigenSelfAdjointMatrix<Arg, t>& arg)
    {
      static_assert(MatrixTraits<Arg>::dimension == dimension);
      if constexpr(t == storage_triangle)
        this->base_matrix().template triangularView<uplo>() -= arg.base_matrix();
      else
        this->base_matrix().template triangularView<uplo>() -= arg.base_matrix().adjoint();
      return *this;
    }

    template<typename S, std::enable_if_t<std::is_convertible_v<S, Scalar>, int> = 0>
    auto& operator*=(const S s)
    {
      this->base_matrix().template triangularView<uplo>() *= s;
      return *this;
    }

    template<typename S, std::enable_if_t<std::is_convertible_v<S, Scalar>, int> = 0>
    auto& operator/=(const S s)
    {
      this->base_matrix().template triangularView<uplo>() /= s;
      return *this;
    }

    constexpr auto& base_view() & { return view; }

    constexpr const auto& base_view() const & { return view; }

    constexpr auto&& base_view() && { return std::move(view); }

    constexpr const auto&& base_view() const && { return std::move(view); }


    auto operator()(std::size_t i, std::size_t j)
    {
      if constexpr (is_element_settable_v<EigenSelfAdjointMatrix, 2>)
        return internal::ElementSetter(*this, i, j);
      else
        return const_cast<const EigenSelfAdjointMatrix&>(*this)(i, j);
    }

    auto operator()(std::size_t i, std::size_t j) const noexcept { return internal::ElementSetter(*this, i, j); }

    auto operator[](std::size_t i)
    {
      if constexpr(is_element_settable_v<EigenSelfAdjointMatrix, 1>)
        return internal::ElementSetter(*this, i);
      else if constexpr(is_element_settable_v<EigenSelfAdjointMatrix, 2>)
        return internal::ElementSetter(*this, i, i);
      else
        return const_cast<const EigenSelfAdjointMatrix&>(*this)[i];
    }

    auto operator[](std::size_t i) const
    {
      if constexpr(is_element_gettable_v<EigenSelfAdjointMatrix, 1>)
        return internal::ElementSetter(*this, i);
      else
        return internal::ElementSetter(*this, i, i);
    }

    auto operator()(std::size_t i) { return operator[](i); }

    auto operator()(std::size_t i) const { return operator[](i); }


    template<typename B, std::enable_if_t<is_Eigen_matrix_v<B>, int> = 0>
    auto solve(const B& b) const
    {
      using M = Eigen::Matrix<Scalar, dimension, MatrixTraits<B>::columns>;
      auto llt = this->base_view().llt();
      M ret = llt.solve(b);
      if (llt.info() != Eigen::Success) //[[unlikely]] // C++20
      {
        // A is semidefinite. Use LDLT decomposition instead.
        auto ldlt = this->base_view().ldlt();
        M ret2 = ldlt.solve(b);
        if ((not ldlt.isPositive() and not ldlt.isNegative()) or ldlt.info() != Eigen::Success)
        {
          throw (std::runtime_error("EigenSelfAdjointMatrix solve: A is indefinite"));
        }
        return ret2;
      }
      return ret;
    }

    static auto zero() { return MatrixTraits<BaseMatrix>::zero(); }

    static auto identity()
    {
      auto b = MatrixTraits<BaseMatrix>::identity();
      return EigenSelfAdjointMatrix<std::decay_t<decltype(b)>, TriangleType::diagonal>(std::move(b));
    }

  private:
    View view;
  };

  /////////////////////////////////////
  //        Deduction Guides         //
  /////////////////////////////////////

  template<typename M, std::enable_if_t<is_Eigen_matrix_v<M> or is_EigenDiagonal_v<M>, int> = 0>
  EigenSelfAdjointMatrix(M&&)
    -> EigenSelfAdjointMatrix<lvalue_or_strict_t<M>, TriangleType::lower>;

  template<typename M, std::enable_if_t<(is_EigenTriangularMatrix_v<M> and is_lower_triangular_v<M>) or
    (is_EigenSelfAdjointMatrix_v<M> and is_Eigen_lower_storage_triangle_v<M>), int> = 0>
  EigenSelfAdjointMatrix(M&&)
    -> EigenSelfAdjointMatrix<typename MatrixTraits<M>::BaseMatrix, TriangleType::lower>;

  template<typename M, std::enable_if_t<(is_EigenTriangularMatrix_v<M> and is_upper_triangular_v<M>) or
  (is_EigenSelfAdjointMatrix_v<M> and is_Eigen_upper_storage_triangle_v<M>), int> = 0>
  EigenSelfAdjointMatrix(M&&)
    -> EigenSelfAdjointMatrix<typename MatrixTraits<M>::BaseMatrix, TriangleType::upper>;

  template<typename Arg, unsigned int UpLo>
  EigenSelfAdjointMatrix(Eigen::SelfAdjointView<Arg, UpLo>&&)
    -> EigenSelfAdjointMatrix<Arg, UpLo & Eigen::Upper ? TriangleType::upper : TriangleType::lower>;

  /// If the arguments are a sequence of scalars, deduce a square, self-adjoint matrix.
  template<typename ... Args, std::enable_if_t<std::conjunction_v<std::is_arithmetic<Args>...>, int> = 0>
  EigenSelfAdjointMatrix(Args ...)
    -> EigenSelfAdjointMatrix<
    Eigen::Matrix<std::decay_t<std::common_type_t<Args...>>, internal::constexpr_sqrt(sizeof...(Args)),
      internal::constexpr_sqrt(sizeof...(Args))>, TriangleType::lower>;


  ///////////////////////////////////
  //        Make functions         //
  ///////////////////////////////////

  template<TriangleType t = TriangleType::lower, typename M,
    std::enable_if_t<is_Eigen_matrix_v<M> or is_EigenDiagonal_v<M>, int> = 0>
  auto
  make_EigenSelfAdjointMatrix(M&& m)
  {
    return EigenSelfAdjointMatrix<lvalue_or_strict_t<M>, t>(std::forward<M>(m));
  }

  template<TriangleType t, typename M, std::enable_if_t<is_EigenSelfAdjointMatrix_v<M>, int> = 0>
  auto
  make_EigenSelfAdjointMatrix(M&& m)
  {
    return EigenSelfAdjointMatrix<lvalue_or_strict_t<M>, t>(std::forward<M>(m));
  }

  template<typename M, std::enable_if_t<is_EigenSelfAdjointMatrix_v<M>, int> = 0>
  auto
  make_EigenSelfAdjointMatrix(M&& m)
  {
    return make_EigenSelfAdjointMatrix<MatrixTraits<M>::storage_type>(std::forward<M>(m));
  }


  ///////////////////////////
  //        Traits         //
  ///////////////////////////

  template<typename ArgType, TriangleType storage_triangle>
  struct MatrixTraits<EigenSelfAdjointMatrix<ArgType, storage_triangle>>
  {
    static constexpr TriangleType storage_type = storage_triangle;
    using BaseMatrix = ArgType;
    using Scalar = typename MatrixTraits<BaseMatrix>::Scalar;
    static constexpr auto dimension = MatrixTraits<BaseMatrix>::dimension;
    static constexpr auto columns = MatrixTraits<BaseMatrix>::columns;
    static_assert(dimension == columns, "A self-adjoint matrix must be square.");

    template<typename Derived>
    using MatrixBaseType = internal::EigenMatrixBase<Derived, OpenKalman::EigenSelfAdjointMatrix<std::decay_t<BaseMatrix>, storage_type>>;

    template<typename Derived>
    using CovarianceBaseType = internal::EigenCovarianceBase<Derived, EigenSelfAdjointMatrix<std::decay_t<BaseMatrix>, storage_type>>;

    template<std::size_t rows = dimension, std::size_t cols = dimension, typename S = Scalar>
    using StrictMatrix = typename MatrixTraits<BaseMatrix>::template StrictMatrix<rows, cols, S>;

    using Strict = EigenSelfAdjointMatrix<typename MatrixTraits<BaseMatrix>::Strict, storage_type>;

    template<TriangleType t = storage_type, std::size_t dim = dimension, typename S = Scalar>
    using SelfAdjointBaseType = EigenSelfAdjointMatrix<StrictMatrix<dim, dim, S>, t>;

    template<TriangleType t = storage_type, std::size_t dim = dimension, typename S = Scalar>
    using TriangularBaseType = EigenTriangularMatrix<StrictMatrix<dim, dim, S>, t>;

    template<std::size_t dim = dimension, typename S = Scalar>
    using DiagonalBaseType = EigenDiagonal<StrictMatrix<dim, 1, S>>;

    template<TriangleType t = storage_type, typename Arg,
      std::enable_if_t<not std::is_convertible_v<Arg, const Scalar>, int> = 0>
    static auto make(Arg&& arg) noexcept
    {
      return EigenSelfAdjointMatrix<std::decay_t<Arg>, t>(std::forward<Arg>(arg));
    }

    /// Make self-adjoint matrix using a list of coefficients in row-major order.
    /// Only the coefficients in the lower-left corner are significant.
    template<TriangleType t = storage_type, typename ... Args,
      std::enable_if_t<std::conjunction_v<std::is_convertible<Args, const Scalar>...>, int> = 0>
    static auto make(Args ... args)
    {
      static_assert(sizeof...(Args) == dimension * dimension);
      return make<t>(MatrixTraits<BaseMatrix>::make(args...));
    }

    static auto zero() { return EigenSelfAdjointMatrix<BaseMatrix, storage_type>::zero(); }

    static auto identity() { return EigenSelfAdjointMatrix<BaseMatrix, storage_type>::identity(); }

  };


  //////////////////////////////
  //        Overloads         //
  //////////////////////////////

  template<typename Arg, std::enable_if_t<is_EigenSelfAdjointMatrix_v<Arg>, int> = 0>
  inline auto
  transpose(Arg&& arg)
  {
    constexpr auto t = is_Eigen_lower_storage_triangle_v<Arg> ? TriangleType::upper : TriangleType::lower;
    auto b = transpose(base_matrix(std::forward<Arg>(arg)));
    return EigenSelfAdjointMatrix<decltype(b), t>(std::move(b));
  }


  template<typename Arg, std::enable_if_t<is_EigenSelfAdjointMatrix_v<Arg>, int> = 0>
  inline auto
  adjoint(Arg&& arg)
  {
    constexpr auto t = is_Eigen_lower_storage_triangle_v<Arg> ? TriangleType::upper : TriangleType::lower;
    auto b = adjoint(base_matrix(std::forward<Arg>(arg)));
    return EigenSelfAdjointMatrix<decltype(b), t>(std::move(b));
  }


  template<typename Arg, typename U,
    std::enable_if_t<is_EigenSelfAdjointMatrix_v<Arg> and
      (is_Eigen_matrix_v<U> or is_EigenTriangularMatrix_v<U> or is_EigenSelfAdjointMatrix_v<U> or is_EigenDiagonal_v<U>) and
      not std::is_const_v<std::remove_reference_t<Arg>>, int> = 0>
  inline Arg&
  rank_update(Arg& arg, const U& u, const typename MatrixTraits<Arg>::Scalar alpha = 1)
  {
    static_assert(MatrixTraits<U>::dimension == MatrixTraits<Arg>::dimension);
    arg.base_view().rankUpdate(u, alpha);
    return arg;
  }


  template<typename Arg, typename U,
    std::enable_if_t<is_EigenSelfAdjointMatrix_v<Arg> and
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
  template<typename A, std::enable_if_t<is_EigenSelfAdjointMatrix_v<A>, int> = 0>
  inline auto
  LQ_decomposition(A&& a)
  {
    return LQ_decomposition(strict_matrix(std::forward<A>(a)));
  }


  /// Perform a QR decomposition of matrix A=Q[U,0], where U is an upper-triangular matrix, and Q is orthogonal.
  /// Returns L as an upper-triangular matrix.
  template<typename A, std::enable_if_t<is_EigenSelfAdjointMatrix_v<A>, int> = 0>
  inline auto
  QR_decomposition(A&& a)
  {
    return QR_decomposition(strict_matrix(std::forward<A>(a)));
  }


  ///////////////////////////////
  //        Arithmetic         //
  ///////////////////////////////

  template<
    typename Arg1, typename Arg2,
    std::enable_if_t<is_EigenSelfAdjointMatrix_v<Arg1> and is_EigenSelfAdjointMatrix_v<Arg2> and
      not is_diagonal_v<Arg1> and not is_diagonal_v<Arg2>, int> = 0>
  inline auto operator+(Arg1&& arg1, Arg2&& arg2)
  {
    static_assert(MatrixTraits<Arg1>::dimension == MatrixTraits<Arg2>::dimension);
    if constexpr(is_Eigen_lower_storage_triangle_v<Arg1> != is_Eigen_lower_storage_triangle_v<Arg2>)
    {
      auto ret = MatrixTraits<Arg1>::make(base_matrix(std::forward<Arg1>(arg1)) + adjoint(base_matrix(std::forward<Arg2>(arg2))));
      if constexpr (not std::is_lvalue_reference_v<Arg1&&> or not std::is_lvalue_reference_v<Arg2&&>) return strict(std::move(ret)); else return ret;
    }
    else
    {
      auto ret = MatrixTraits<Arg1>::make(base_matrix(std::forward<Arg1&&>(arg1)) + base_matrix(std::forward<Arg2&&>(arg2)));
      if constexpr (not std::is_lvalue_reference_v<Arg1> or not std::is_lvalue_reference_v<Arg2>) return strict(std::move(ret)); else return ret;
    }
  }


  template<typename Arg1, typename Arg2,
    std::enable_if_t<((is_EigenSelfAdjointMatrix_v<Arg1> and is_diagonal_v<Arg2>) or
    (is_diagonal_v<Arg1> and is_EigenSelfAdjointMatrix_v<Arg2>)) and
    not is_zero_v<Arg1> and not is_zero_v<Arg2> and not is_identity_v<Arg1> and not is_identity_v<Arg2>, int> = 0>
  inline auto operator+(Arg1&& arg1, Arg2&& arg2)
  {
    static_assert(MatrixTraits<Arg1>::dimension == MatrixTraits<Arg2>::dimension);
    if constexpr(is_EigenSelfAdjointMatrix_v<Arg1>)
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


  template<typename Arg1, typename Arg2,
    std::enable_if_t<(is_EigenSelfAdjointMatrix_v<Arg1> and is_identity_v<Arg2>) or
      (is_identity_v<Arg1> and is_EigenSelfAdjointMatrix_v<Arg2>), int> = 0>
  inline auto operator+(Arg1&& arg1, Arg2&& arg2)
  {
    static_assert(MatrixTraits<Arg1>::dimension == MatrixTraits<Arg2>::dimension);
    static_assert(MatrixTraits<Arg1>::columns == MatrixTraits<Arg2>::columns);
    if constexpr(is_EigenSelfAdjointMatrix_v<Arg1>)
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


  template<typename Arg1, typename Arg2,
    std::enable_if_t<(is_EigenSelfAdjointMatrix_v<Arg1> and is_zero_v<Arg2>) or
      (is_zero_v<Arg1> and is_EigenSelfAdjointMatrix_v<Arg2>), int> = 0>
  constexpr decltype(auto) operator+(Arg1&& arg1, Arg2&& arg2)
  {
    static_assert(MatrixTraits<Arg1>::dimension == MatrixTraits<Arg2>::dimension);
    static_assert(MatrixTraits<Arg1>::columns == MatrixTraits<Arg2>::columns);
    if constexpr(is_EigenSelfAdjointMatrix_v<Arg1>)
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
    std::enable_if_t<is_EigenSelfAdjointMatrix_v<Arg1>, int> = 0,
    std::enable_if_t<is_EigenSelfAdjointMatrix_v<Arg2>, int> = 0>
  inline auto operator-(Arg1&& arg1, Arg2&& arg2)
  {
    static_assert(MatrixTraits<Arg1>::dimension == MatrixTraits<Arg2>::dimension);
    if constexpr(is_Eigen_lower_storage_triangle_v<Arg1> != is_Eigen_lower_storage_triangle_v<Arg2>)
    {
      auto ret = MatrixTraits<Arg1>::make(base_matrix(std::forward<Arg1>(arg1)) - adjoint(base_matrix(std::forward<Arg2>(arg2))));
      if constexpr (not std::is_lvalue_reference_v<Arg1&&> or not std::is_lvalue_reference_v<Arg2&&>) return strict(std::move(ret)); else return ret;
    }
    else
    {
      auto ret = MatrixTraits<Arg1>::make(base_matrix(std::forward<Arg1>(arg1)) - base_matrix(std::forward<Arg2>(arg2)));
      if constexpr (not std::is_lvalue_reference_v<Arg1&&> or not std::is_lvalue_reference_v<Arg2&&>) return strict(std::move(ret)); else return ret;
    }
  }


  template<typename Arg1, typename Arg2,
    std::enable_if_t<((is_EigenSelfAdjointMatrix_v<Arg1> and is_diagonal_v<Arg2>) or
      (is_diagonal_v<Arg1> and is_EigenSelfAdjointMatrix_v<Arg2>)) and
      not is_zero_v<Arg1> and not is_zero_v<Arg2> and not is_identity_v<Arg1> and not is_identity_v<Arg2>, int> = 0>
  inline auto operator-(Arg1&& arg1, Arg2&& arg2)
  {
    static_assert(MatrixTraits<Arg1>::dimension == MatrixTraits<Arg2>::dimension);
    if constexpr(is_EigenSelfAdjointMatrix_v<Arg1>)
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


  template<typename Arg1, typename Arg2,
    std::enable_if_t<(is_EigenSelfAdjointMatrix_v<Arg1> and is_identity_v<Arg2>) or
      (is_identity_v<Arg1> and is_EigenSelfAdjointMatrix_v<Arg2>), int> = 0>
  inline auto operator-(Arg1&& arg1, Arg2&& arg2)
  {
    static_assert(MatrixTraits<Arg1>::dimension == MatrixTraits<Arg2>::dimension);
    static_assert(MatrixTraits<Arg1>::columns == MatrixTraits<Arg2>::columns);
    if constexpr(is_EigenSelfAdjointMatrix_v<Arg1>)
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


  template<typename Arg1, typename Arg2,
    std::enable_if_t<(is_EigenSelfAdjointMatrix_v<Arg1> and is_zero_v<Arg2>) or
      (is_zero_v<Arg1> and is_EigenSelfAdjointMatrix_v<Arg2>), int> = 0>
  constexpr decltype(auto) operator-(Arg1&& arg1, Arg2&& arg2)
  {
    static_assert(MatrixTraits<Arg1>::dimension == MatrixTraits<Arg2>::dimension);
    static_assert(MatrixTraits<Arg1>::columns == MatrixTraits<Arg2>::columns);
    if constexpr(is_EigenSelfAdjointMatrix_v<Arg1>)
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
    std::enable_if_t<is_EigenSelfAdjointMatrix_v<Arg>, int> = 0,
    std::enable_if_t<std::is_convertible_v<S, typename MatrixTraits<Arg>::Scalar>, int> = 0>
  inline auto operator*(Arg&& arg, const S scale)
  {
    auto ret = MatrixTraits<Arg>::make(std::forward<Arg>(arg).base_matrix() * scale);
    if constexpr (not std::is_lvalue_reference_v<Arg&&>) return strict(std::move(ret)); else return ret;
  }


  template<
    typename Arg, typename S,
    std::enable_if_t<is_EigenSelfAdjointMatrix_v<Arg>, int> = 0,
    std::enable_if_t<std::is_convertible_v<S, typename MatrixTraits<Arg>::Scalar>, int> = 0>
  inline auto operator*(const S scale, Arg&& arg)
  {
    auto ret = MatrixTraits<Arg>::make(scale * std::forward<Arg>(arg).base_matrix());
    if constexpr (not std::is_lvalue_reference_v<Arg&&>) return strict(std::move(ret)); else return ret;
  }


  template<
    typename Arg, typename S,
    std::enable_if_t<is_EigenSelfAdjointMatrix_v<Arg>, int> = 0,
    std::enable_if_t<std::is_convertible_v<S, typename MatrixTraits<Arg>::Scalar>, int> = 0>
  inline auto operator/(Arg&& arg, const S scale)
  {
    auto ret = MatrixTraits<Arg>::make(std::forward<Arg>(arg).base_matrix() / scale);
    if constexpr (not std::is_lvalue_reference_v<Arg&&>) return strict(std::move(ret)); else return ret;
  }


  ////

  template<
    typename Arg1, typename Arg2,
    std::enable_if_t<((is_EigenSelfAdjointMatrix_v<Arg1> and is_EigenSelfAdjointMatrix_v<Arg2>) or
      (is_EigenTriangularMatrix_v<Arg1> and is_EigenSelfAdjointMatrix_v<Arg2>) or
      (is_EigenSelfAdjointMatrix_v<Arg1> and is_EigenTriangularMatrix_v<Arg2>)) and
      not is_diagonal_v<Arg1> and not is_diagonal_v<Arg2>, int> = 0>
  inline auto operator*(Arg1&& arg1, Arg2&& arg2)
  {
    static_assert(MatrixTraits<Arg1>::dimension == MatrixTraits<Arg2>::dimension);
    auto ret = std::forward<Arg1>(arg1).base_view() * std::forward<Arg2>(arg2);
    if constexpr (not std::is_lvalue_reference_v<Arg1&&> or not std::is_lvalue_reference_v<Arg2&&>) return strict(std::move(ret)); else return ret;
  }


  template<
    typename Arg1, typename Arg2,
    std::enable_if_t<((is_EigenSelfAdjointMatrix_v<Arg1> and is_diagonal_v<Arg2>) or
      (is_diagonal_v<Arg1> and is_EigenSelfAdjointMatrix_v<Arg2>)) and
      not is_identity_v<Arg1> and not is_identity_v<Arg2> and not is_zero_v<Arg1> and not is_zero_v<Arg2>, int> = 0>
  inline auto operator*(Arg1&& arg1, Arg2&& arg2)
  {
    static_assert(MatrixTraits<Arg1>::dimension == MatrixTraits<Arg2>::dimension);
    if constexpr(is_EigenSelfAdjointMatrix_v<Arg1>)
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


  template<typename Arg1, typename Arg2,
    std::enable_if_t<(is_EigenSelfAdjointMatrix_v<Arg1> and is_identity_v<Arg2>) or
      (is_identity_v<Arg1> and is_EigenSelfAdjointMatrix_v<Arg2>), int> = 0>
  inline auto operator*(Arg1&& arg1, Arg2&& arg2)
  {
    static_assert(MatrixTraits<Arg1>::dimension == MatrixTraits<Arg2>::dimension);
    static_assert(MatrixTraits<Arg1>::columns == MatrixTraits<Arg2>::columns);
    if constexpr(is_EigenSelfAdjointMatrix_v<Arg1>)
    {
      return std::forward<Arg1>(arg1);
    }
    else
    {
      return std::forward<Arg2>(arg2);
    }
  }


  template<typename Arg1, typename Arg2,
    std::enable_if_t<(is_EigenSelfAdjointMatrix_v<Arg1> and is_zero_v<Arg2>) or
      (is_zero_v<Arg1> and is_EigenSelfAdjointMatrix_v<Arg2>), int> = 0>
  constexpr decltype(auto) operator*(Arg1&& arg1, Arg2&& arg2)
  {
    static_assert(MatrixTraits<Arg1>::columns == MatrixTraits<Arg2>::dimension);
    constexpr auto rows = MatrixTraits<Arg1>::dimension;
    constexpr auto cols = MatrixTraits<Arg2>::columns;
    using B = strict_matrix_t<Arg1, rows, cols>;
    return EigenZero<B>();
  }


  ////

  template<
    typename Arg1, typename Arg2,
    std::enable_if_t<((is_EigenSelfAdjointMatrix_v<Arg1> and is_Eigen_matrix_v<Arg2>)
      or (is_Eigen_matrix_v<Arg1> and is_EigenSelfAdjointMatrix_v<Arg2>)) and
      not is_identity_v<Arg1> and not is_identity_v<Arg2> and not is_zero_v<Arg1> and not is_zero_v<Arg2>, int> = 0>
  inline auto operator*(Arg1&& arg1, Arg2&& arg2)
  {
    static_assert(MatrixTraits<Arg1>::columns == MatrixTraits<Arg2>::dimension);
    if constexpr(is_EigenSelfAdjointMatrix_v<Arg1>)
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

  template<
    typename Arg,
    std::enable_if_t<is_EigenSelfAdjointMatrix_v<Arg>, int> = 0>
  inline auto operator-(Arg&& arg)
  {
    auto ret = MatrixTraits<Arg>::make(-std::forward<Arg>(arg).base_matrix());
    if constexpr (not std::is_lvalue_reference_v<Arg&&>) return strict(std::move(ret)); else return ret;
  }

}


#endif //OPENKALMAN_EIGENSELFADJOINTMATRIX_H
