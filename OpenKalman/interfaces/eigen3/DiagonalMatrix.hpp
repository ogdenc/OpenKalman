/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef OPENKALMAN_EIGEN3_DIAGONALMATRIX_HPP
#define OPENKALMAN_EIGEN3_DIAGONALMATRIX_HPP

namespace OpenKalman::Eigen3
{
  template<typename BaseMatrix>
  struct DiagonalMatrix : OpenKalman::internal::MatrixBase<DiagonalMatrix<BaseMatrix>, BaseMatrix>
  {
    using Base = OpenKalman::internal::MatrixBase<DiagonalMatrix, BaseMatrix>;
    using Scalar = typename MatrixTraits<BaseMatrix>::Scalar;
    static constexpr auto dimension = MatrixTraits<BaseMatrix>::dimension;

    /// Default constructor.
    DiagonalMatrix() : Base() {}

    /// Copy constructor.
    DiagonalMatrix(const DiagonalMatrix& other) : DiagonalMatrix(other.base_matrix()) {}

    /// Move constructor.
    DiagonalMatrix(DiagonalMatrix&& other) noexcept: DiagonalMatrix(std::move(other).base_matrix()) {}

    /// Construct from a compatible DiagonalMatrix.
#ifdef __cpp_concepts
    template<eigen_diagonal_expr Arg>
#else
    template<typename Arg, std::enable_if_t<eigen_diagonal_expr < Arg>, int> = 0>
#endif
    DiagonalMatrix(Arg&& other) noexcept: Base(std::forward<Arg>(other).base_matrix())
    {
      static_assert(MatrixTraits<Arg>::dimension == dimension);
    }

    /// Construct from a column vector matrix.
#ifdef __cpp_concepts
    template<eigen_matrix Arg>
    requires (MatrixTraits<Arg>::columns == 1)
#else
    template<typename Arg, std::enable_if_t<eigen_matrix<Arg> and MatrixTraits<Arg>::columns == 1, int> = 0>
#endif
    DiagonalMatrix(Arg&& arg) noexcept : Base(std::forward<Arg>(arg)) {}

    /// Construct from a compatible diagonal native matrix that is not DiagonalMatrix.
#ifdef __cpp_concepts
    template<eigen_native Arg>
    requires is_diagonal_v<Arg> and
      (not eigen_diagonal_expr < Arg > ) and (MatrixTraits<Arg>::columns > 1)
#else
    template<
      typename Arg,
      std::enable_if_t<eigen_native < Arg>and is_diagonal_v<Arg>and
    not
    eigen_diagonal_expr <Arg>and (MatrixTraits<Arg>::columns
    > 1), int> = 0>
#endif
    DiagonalMatrix(Arg&& other) noexcept: Base(std::forward<Arg>(other).diagonal())
    {
      static_assert(MatrixTraits<Arg>::dimension == dimension);
    }

    //@TODO: Add constructor from Eigen::DiagonalMatrix

    /// Construct from a square zero matrix.
#ifdef __cpp_concepts
    template<typename Arg>
    requires is_zero_v<Arg> and (MatrixTraits<Arg>::columns > 1)
#else
    template<typename Arg, std::enable_if_t<is_zero_v < Arg>and (MatrixTraits<Arg>::columns> 1), int> = 0>
#endif
    DiagonalMatrix(const Arg&) : Base(MatrixTraits<Eigen::Matrix<Scalar, dimension, 1>>::zero())
    {
      static_assert(MatrixTraits<Arg>::dimension == dimension);
      static_assert(MatrixTraits<Arg>::columns == dimension);
    }

    /// Construct from an identity matrix.
#ifdef __cpp_concepts
    template<typename Arg>
    requires is_identity_v<Arg> and (MatrixTraits<Arg>::columns > 1)
#else
    template<typename Arg, std::enable_if_t<is_identity_v<Arg> and (MatrixTraits<Arg>::columns > 1), int> = 0>
#endif
    DiagonalMatrix(const Arg&) : Base(Eigen::Matrix<Scalar, dimension, 1>::Constant(1))
    {
      static_assert(MatrixTraits<Arg>::dimension == dimension);
      static_assert(MatrixTraits<Arg>::columns == dimension);
    }

    /// Construct from a list of scalar coefficients defining the diagonal.
#ifdef __cpp_concepts
    template<std::convertible_to<const Scalar> ... Args>
    requires (sizeof...(Args) == dimension)
#else
    template<
      typename ... Args, std::enable_if_t<std::conjunction_v<std::is_convertible<Args, const Scalar>...> and
        sizeof...(Args) == dimension, int> = 0>
#endif
    DiagonalMatrix(Args ... args) : DiagonalMatrix(MatrixTraits<BaseMatrix>::make(args...)) {}

    /// Copy assignment operator.
    auto& operator=(const DiagonalMatrix& other)
    {
      if (this != &other) this->base_matrix() = other.base_matrix();
      return *this;
    }

    /// Move assignment operator.
    auto& operator=(DiagonalMatrix&& other) noexcept
    {
      if (this != &other) this->base_matrix() = std::move(other).base_matrix();
      return *this;
    }

    /// Assign from a compatible DiagonalMatrix.
#ifdef __cpp_concepts
    template<eigen_diagonal_expr Arg>
    requires (not is_zero_v<Arg>) and (not is_identity_v<Arg>)
#else
    template<typename Arg, std::enable_if_t<
      eigen_diagonal_expr < Arg>and not is_zero_v <Arg> and not is_identity_v<Arg>, int> = 0>
#endif
    auto& operator=(Arg&& other)
    {
      if constexpr (std::is_same_v<std::decay_t<Arg>, DiagonalMatrix>) if (this == &other) return *this;
      this->base_matrix() = std::forward<Arg>(other).base_matrix();
      return *this;
    }

    /// Assign from a square zero matrix.
#ifdef __cpp_concepts
    template<typename Arg> requires is_zero_v<Arg>
#else
    template<typename Arg, std::enable_if_t<is_zero_v<Arg>, int> = 0>
#endif
    auto& operator=(const Arg&)
    {
      static_assert(MatrixTraits<Arg>::dimension == dimension);
      static_assert(MatrixTraits<Arg>::columns == dimension);
      this->base_matrix() = MatrixTraits<Eigen::Matrix<Scalar, dimension, 1>>::zero();
      return *this;
    }

    /// Assign from an identity matrix.
#ifdef __cpp_concepts
    template<typename Arg> requires is_identity_v<Arg>
#else
    template<typename Arg, std::enable_if_t<is_identity_v<Arg>, int> = 0>
#endif
    auto& operator=(const Arg&)
    {
      static_assert(MatrixTraits<Arg>::dimension == dimension);
      static_assert(MatrixTraits<Arg>::columns == dimension);
      this->base_matrix() = Eigen::Matrix<Scalar, dimension, 1>::Constant(1);
      return *this;
    }

    /// Assign by copying from an Eigen::DiagonalBase derived object.
    template<typename Arg>
    auto& operator=(const Eigen::DiagonalBase<Arg>& arg)
    {
      this->base_matrix() = arg.diagonal();
      return *this;
    }

    /// Assign by moving from an Eigen::DiagonalBase derived object.
    template<typename Arg>
    auto& operator=(Eigen::DiagonalBase<Arg>&& arg) noexcept
    {
      this->base_matrix() = std::move(arg).diagonal();
      return *this;
    }

    template<typename Arg>
    auto& operator+=(const DiagonalMatrix<Arg>& arg)
    {
      static_assert(MatrixTraits<Arg>::dimension == dimension);
      this->base_matrix() += arg.base_matrix();
      return *this;
    }

    template<typename Arg>
    auto& operator-=(const DiagonalMatrix<Arg>& arg)
    {
      static_assert(MatrixTraits<Arg>::dimension == dimension);
      this->base_matrix() -= arg.base_matrix();
      return *this;
    }

#ifdef __cpp_concepts
    template<std::convertible_to<Scalar> S>
#else
    template<typename S, std::enable_if_t<std::is_convertible_v<S, Scalar>, int> = 0>
#endif
    auto& operator*=(const S s)
    {
      this->base_matrix() *= s;
      return *this;
    }

#ifdef __cpp_concepts
    template<std::convertible_to<Scalar> S>
#else
    template<typename S, std::enable_if_t<std::is_convertible_v<S, Scalar>, int> = 0>
#endif
    auto& operator/=(const S s)
    {
      this->base_matrix() /= s;
      return *this;
    }

    template<typename Arg>
    auto& operator*=(const DiagonalMatrix<Arg>& arg)
    {
      static_assert(MatrixTraits<Arg>::dimension == dimension);
      this->base_matrix() = this->base_matrix().array() * arg.base_matrix().array();
      return *this;
    }

    auto square() const
    {
      auto b = this->base_matrix().array().square().matrix();
      return DiagonalMatrix<decltype(b)>(std::move(b));
    }

    auto square_root() const
    {
      auto b = this->base_matrix().cwiseSqrt();
      return DiagonalMatrix<decltype(b)>(std::move(b));
    }


    auto operator()(std::size_t i, std::size_t j)
    {
      if constexpr (is_element_settable_v < DiagonalMatrix, 2 >)
        return OpenKalman::internal::ElementSetter(*this, i, j);
      else
        return const_cast<const DiagonalMatrix&>(*this)(i, j);
    }

    auto operator()(std::size_t i, std::size_t j) const noexcept
    {
      return OpenKalman::internal::ElementSetter(*this, i, j);
    }


    auto operator[](std::size_t i)
    {
      if constexpr (is_element_settable_v < DiagonalMatrix, 1 >)
        return OpenKalman::internal::ElementSetter(*this, i);
      else
        return const_cast<const DiagonalMatrix&>(*this)[i];
    }

    auto operator[](std::size_t i) const { return OpenKalman::internal::ElementSetter(*this, i); }

    auto operator()(std::size_t i) { return operator[](i); }

    auto operator()(std::size_t i) const { return operator[](i); }

    static auto zero() { return MatrixTraits<Eigen::Matrix<Scalar, dimension, dimension>>::zero(); }

    static auto identity() { return MatrixTraits<Eigen::Matrix<Scalar, dimension, dimension>>::identity(); }

  };


  /////////////////////////////////////
  //        Deduction guides         //
  /////////////////////////////////////

#if defined(__cpp_concepts) and false
  // @TODO Unlike SFINAE version, this incorrectly matches M==double in both GCC 10.1.0 and clang 10.0.0:
  template<eigen_matrix M> requires (MatrixTraits<M>::columns == 1)
#else
  template<typename M, std::enable_if_t<eigen_matrix<M> and MatrixTraits<M>::columns == 1, int> = 0>
#endif
  DiagonalMatrix(M&&) -> DiagonalMatrix<lvalue_or_strict_t < M>>;


#ifdef __cpp_concepts
  template<eigen_native Arg>
  requires is_diagonal_v<Arg> and
    (not eigen_diagonal_expr < Arg > ) and (MatrixTraits<Arg>::columns > 1)
#else
  template<
    typename Arg,
    std::enable_if_t<eigen_native < Arg>and is_diagonal_v<Arg>and
  not
  eigen_diagonal_expr <Arg>and (MatrixTraits<Arg>::columns
  > 1), int> = 0>
#endif
  DiagonalMatrix(Arg && )
  -> DiagonalMatrix<strict_t < decltype(std::forward<Arg>(std::declval<Arg>()).diagonal())>>;


#ifdef __cpp_concepts
  template<typename Arg>
  requires is_zero_v<Arg> and (MatrixTraits<Arg>::columns > 1) and
    (MatrixTraits<Arg>::dimension == MatrixTraits<Arg>::columns)
#else
  template<typename Arg, std::enable_if_t<is_zero_v < Arg>and (MatrixTraits<Arg>::columns> 1) and
  MatrixTraits<Arg>::dimension == MatrixTraits<Arg>::columns, int> = 0>
#endif
  DiagonalMatrix(const Arg&)
  -> DiagonalMatrix<ZeroMatrix < Eigen::Matrix<typename MatrixTraits<Arg>::Scalar, MatrixTraits<Arg>::dimension, 1>>>;


#ifdef __cpp_concepts
  template<typename Arg>
  requires is_identity_v<Arg> and (MatrixTraits<Arg>::columns > 1) and
    (MatrixTraits<Arg>::dimension == MatrixTraits<Arg>::columns)
#else
  template<
    typename Arg, std::enable_if_t<is_identity_v<Arg> and (MatrixTraits<Arg>::columns > 1) and
      MatrixTraits<Arg>::dimension == MatrixTraits<Arg>::columns, int> = 0>
#endif
  DiagonalMatrix(const Arg&)
  -> DiagonalMatrix<typename Eigen::Matrix<typename MatrixTraits<Arg>::Scalar, MatrixTraits<Arg>::dimension, 1>::ConstantReturnType>;


#ifdef __cpp_concepts
  template<typename Arg, typename ... Args>
  requires (std::is_arithmetic_v<Arg> and ... and std::is_arithmetic_v<Args>) and (std::common_with<Arg, Args>and ...)
#else
    template<
      typename Arg, typename ... Args,
      std::enable_if_t<std::conjunction_v<std::is_arithmetic<Arg>, std::is_arithmetic<Args>...>, int> = 0>
#endif
    DiagonalMatrix(Arg, Args ...)
      ->DiagonalMatrix<Eigen::Matrix<std::decay_t<std::common_type_t<Arg, Args...>>, 1 + sizeof...(Args), 1>>;

} // OpenKalman::Eigen3


namespace OpenKalman
{
  /////////////////////////////////
  //        MatrixTraits         //
  /////////////////////////////////

  template<typename ArgType>
  struct MatrixTraits<Eigen3::DiagonalMatrix<ArgType>>
  {
    using BaseMatrix = ArgType;
    using Scalar = typename MatrixTraits<BaseMatrix>::Scalar;
    static constexpr auto dimension = MatrixTraits<BaseMatrix>::dimension;
    static constexpr auto columns = dimension;

    template<typename Derived>
    using MatrixBaseType = Eigen3::internal::Eigen3MatrixBase<Derived, Eigen3::DiagonalMatrix<std::decay_t<BaseMatrix>>>;

    template<typename Derived>
    using CovarianceBaseType = Eigen3::internal::Eigen3CovarianceBase<Derived, Eigen3::DiagonalMatrix<std::decay_t<ArgType>>>;

    template<std::size_t rows = dimension, std::size_t cols = columns, typename S = Scalar>
    using StrictMatrix = typename MatrixTraits<std::decay_t<BaseMatrix>>::template StrictMatrix<rows, cols, S>;

    using Strict = Eigen3::DiagonalMatrix<strict_t<BaseMatrix>>;

    template<TriangleType storage_triangle = TriangleType::diagonal, std::size_t dim = dimension, typename S = Scalar>
    using SelfAdjointBaseType = Eigen3::SelfAdjointMatrix<StrictMatrix<dim, dim, S>, storage_triangle>;

    template<TriangleType triangle_type = TriangleType::diagonal, std::size_t dim = dimension, typename S = Scalar>
    using TriangularBaseType = Eigen3::TriangularMatrix<StrictMatrix<dim, dim, S>, triangle_type>;

    template<std::size_t dim = dimension, typename S = Scalar>
    using DiagonalBaseType = Eigen3::DiagonalMatrix<StrictMatrix<dim, 1, S>>;

#ifdef __cpp_concepts

    template<typename Arg>
    requires (not std::convertible_to<Arg, const Scalar>)
#else
    template<typename Arg, std::enable_if_t<not std::is_convertible_v<Arg, const Scalar>, int> = 0>
#endif
    static auto make(Arg&& arg) noexcept
    {
      static_assert(MatrixTraits<Arg>::columns == 1);
      return Eigen3::DiagonalMatrix < std::decay_t<Arg>>
      (std::forward<Arg>(arg));
    }

    /** Make diagonal matrix using a list of coefficients defining the diagonal.
     * The size of the list must match the number of diagonal coefficients.
     */
#ifdef __cpp_concepts

    template<std::convertible_to<const Scalar> ... Args>
    requires (sizeof...(Args) == dimension)
#else
    template<typename ... Args, std::enable_if_t<std::conjunction_v<std::is_convertible<Args, const Scalar>...> and
      sizeof...(Args) == dimension, int> = 0>
#endif
    static auto make(Args ... args)
    {
      static_assert(sizeof...(Args) == dimension);
      return make(MatrixTraits<BaseMatrix>::make(args...));
    }

    /** Make diagonal matrix using a list of coefficients in row-major order (ignoring non-diagonal coefficients).
     * The size of the list must match the number of coefficients in the matrix (diagonal and non-diagonal).
     */
#ifdef __cpp_concepts

    template<std::convertible_to<const Scalar> ... Args>
    requires
    (sizeof...(Args) == dimension * dimension) and (dimension > 1)
#else
    template<typename ... Args, std::enable_if_t<std::conjunction_v<std::is_convertible<Args, const Scalar>...> and
      (sizeof...(Args) == dimension * dimension) and (dimension > 1), int> = 0>
#endif
    static auto
    make(Args ... args)
    {
      return make(Eigen3::strict(MatrixTraits<StrictMatrix<>>::make(args...).diagonal()));
    }

    static auto zero() { return Eigen3::DiagonalMatrix < BaseMatrix > ::zero(); }

    static auto identity() { return Eigen3::DiagonalMatrix < BaseMatrix > ::identity(); }

  };

} // namespace OpenKalman


namespace OpenKalman::Eigen3
{
  //////////////////////////////
  //        Overloads         //
  //////////////////////////////

#ifdef __cpp_concepts
  template<Eigen3::eigen_diagonal_expr Arg>
#else
  template<typename Arg, std::enable_if_t<Eigen3::eigen_diagonal_expr<Arg>, int> = 0>
#endif
  constexpr decltype(auto)
  base_matrix(Arg&& arg) { return std::forward<Arg>(arg).base_matrix(); }


  /// Convert to strict version
#ifdef __cpp_concepts
  template<Eigen3::eigen_diagonal_expr Arg>
#else
  template<typename Arg, std::enable_if_t<Eigen3::eigen_diagonal_expr<Arg>, int> = 0>
#endif
  inline decltype(auto)
  strict(Arg&& arg)
  {
    if constexpr(is_strict_v<Arg>)
    {
      return std::forward<Arg>(arg);
    }
    else
    {
      return Eigen3::DiagonalMatrix(strict(base_matrix(std::forward<Arg>(arg))));
    }
  }


#ifdef __cpp_concepts
  template<Eigen3::eigen_diagonal_expr Arg>
#else
  template<typename Arg, std::enable_if_t<Eigen3::eigen_diagonal_expr<Arg>, int> = 0>
#endif
  constexpr decltype(auto)
  transpose(Arg&& arg) noexcept
  {
    return std::forward<Arg>(arg);
  }


#ifdef __cpp_concepts
  template<Eigen3::eigen_diagonal_expr Arg>
#else
  template<typename Arg, std::enable_if_t<Eigen3::eigen_diagonal_expr<Arg>, int> = 0>
#endif
  constexpr decltype(auto)
  adjoint(Arg&& arg) noexcept
  {
    return Eigen3::DiagonalMatrix(base_matrix(std::forward<Arg>(arg)).conjugate());
  }


#ifdef __cpp_concepts
  template<Eigen3::eigen_diagonal_expr Arg>
#else
  template<typename Arg, std::enable_if_t<Eigen3::eigen_diagonal_expr<Arg>, int> = 0>
#endif
  inline auto
  determinant(Arg&& arg) noexcept
  {
    static_assert(MatrixTraits<Arg>::dimension == MatrixTraits<Arg>::columns);
    return base_matrix(std::forward<Arg>(arg)).prod();
  }


#ifdef __cpp_concepts
  template<Eigen3::eigen_diagonal_expr Arg>
#else
  template<typename Arg, std::enable_if_t<Eigen3::eigen_diagonal_expr<Arg>, int> = 0>
#endif
  inline auto
  trace(Arg&& arg) noexcept
  {
    static_assert(MatrixTraits<Arg>::dimension == MatrixTraits<Arg>::columns);
    return base_matrix(std::forward<Arg>(arg)).sum();
  }


#ifdef __cpp_concepts
  template<Eigen3::eigen_diagonal_expr Arg, Eigen3::eigen_diagonal_expr U>
    requires (not std::is_const_v<std::remove_reference_t<Arg>>)
#else
  template<typename Arg, typename U, std::enable_if_t<Eigen3::eigen_diagonal_expr<Arg> and Eigen3::eigen_diagonal_expr<U> and
    not std::is_const_v<std::remove_reference_t<Arg>>, int> = 0>
#endif
  inline Arg&
  rank_update(Arg& arg, const U& u, const typename MatrixTraits<Arg>::Scalar alpha = 1)
  {
    static_assert(MatrixTraits<U>::dimension == MatrixTraits<Arg>::dimension);
    arg = (base_matrix(arg).array().square() + alpha * base_matrix(u).array().square()).sqrt().matrix();
    return arg;
  }


#ifdef __cpp_concepts
  template<Eigen3::eigen_diagonal_expr Arg, typename U> requires is_diagonal_v<U> and (not Eigen3::eigen_diagonal_expr<U>) and
    (not std::is_const_v<std::remove_reference_t<Arg>>)
#else
  template<typename Arg, typename U, std::enable_if_t<Eigen3::eigen_diagonal_expr<Arg> and is_diagonal_v<U> and
    not Eigen3::eigen_diagonal_expr<U> and not std::is_const_v<std::remove_reference_t<Arg>>, int> = 0>
#endif
  inline Arg&
  rank_update(Arg& arg, const U& u, const typename MatrixTraits<Arg>::Scalar alpha = 1)
  {
    static_assert(MatrixTraits<U>::dimension == MatrixTraits<Arg>::dimension);
    auto sa = TriangularMatrix(strict_matrix(arg));
    rank_update(sa, u, alpha);
    arg = Eigen3::DiagonalMatrix(strict_matrix(base_matrix(sa).diagonal()));
    return arg;
  }


#ifdef __cpp_concepts
  template<Eigen3::eigen_diagonal_expr Arg, Eigen3::eigen_diagonal_expr U>
#else
  template<typename Arg, typename U, std::enable_if_t<
    Eigen3::eigen_diagonal_expr<Arg> and Eigen3::eigen_diagonal_expr<U>, int> = 0>
#endif
  inline auto
  rank_update(const Arg& arg, const U& u, const typename MatrixTraits<Arg>::Scalar alpha = 1)
  {
    static_assert(MatrixTraits<U>::dimension == MatrixTraits<Arg>::dimension);
    auto sa = (base_matrix(arg).array().square() + alpha * base_matrix(u).array().square()).sqrt().matrix();
    return Eigen3::DiagonalMatrix(sa);
  }


#ifdef __cpp_concepts
  template<Eigen3::eigen_diagonal_expr Arg, typename U> requires is_diagonal_v<U> and (not Eigen3::eigen_diagonal_expr<U>)
#else
  template<typename Arg, typename U, std::enable_if_t<
    Eigen3::eigen_diagonal_expr<Arg> and is_diagonal_v<U> and not Eigen3::eigen_diagonal_expr<U>, int> = 0>
#endif
  inline auto
  rank_update(const Arg& arg, const U& u, const typename MatrixTraits<Arg>::Scalar alpha = 1)
  {
    static_assert(MatrixTraits<U>::dimension == MatrixTraits<Arg>::dimension);
    auto sa = TriangularMatrix(strict_matrix(arg));
    rank_update(sa, u, alpha);
    return Eigen3::TriangularMatrix<std::decay_t<decltype(sa)>, TriangleType::diagonal>(sa);
  }


#ifdef __cpp_concepts
  template<Eigen3::eigen_diagonal_expr Arg, typename U> requires (not is_diagonal_v<U>)
#else
  template<typename Arg, typename U, std::enable_if_t<Eigen3::eigen_diagonal_expr<Arg> and not is_diagonal_v<U>, int> = 0>
#endif
  inline auto
  rank_update(const Arg& arg, const U& u, const typename MatrixTraits<Arg>::Scalar alpha = 1)
  {
    static_assert(MatrixTraits<U>::dimension == MatrixTraits<Arg>::dimension);
    auto sa = TriangularMatrix(strict_matrix(arg));
    return rank_update(sa, u, alpha);
  }


  /// Solve the equation AX = B for X. A is a diagonal matrix.
#ifdef __cpp_concepts
  template<Eigen3::eigen_diagonal_expr A, Eigen3::eigen_matrix B>
#else
  template<typename A, typename B, std::enable_if_t<
    Eigen3::eigen_diagonal_expr<A> and Eigen3::eigen_matrix<B>, int> = 0>
#endif
  inline auto
  solve(const A& a, const B& b)
  {
    static_assert(MatrixTraits<A>::dimension == MatrixTraits<B>::dimension);
    return (b.array().colwise() / base_matrix(a).array()).matrix();
  }


  /// Create a column vector from a diagnoal matrix. (Same as base_matrix()).
#ifdef __cpp_concepts
  template<Eigen3::eigen_diagonal_expr Arg>
#else
  template<typename Arg, std::enable_if_t<Eigen3::eigen_diagonal_expr<Arg>, int> = 0>
#endif
  constexpr decltype(auto)
  reduce_columns(Arg&& arg) noexcept
  {
    return base_matrix(std::forward<Arg>(arg));
  }


  /// Perform an LQ decomposition. Since it is diagonal, it returns the matrix unchanged.
#ifdef __cpp_concepts
  template<Eigen3::eigen_diagonal_expr Arg>
#else
  template<typename Arg, std::enable_if_t<Eigen3::eigen_diagonal_expr<Arg>, int> = 0>
#endif
  constexpr decltype(auto)
  LQ_decomposition(Arg&& arg)
  {
    return std::forward<Arg>(arg);
  }


  /// Perform a QR decomposition. Since it is diagonal, it returns the matrix unchanged.
#ifdef __cpp_concepts
  template<Eigen3::eigen_diagonal_expr Arg>
#else
  template<typename Arg, std::enable_if_t<Eigen3::eigen_diagonal_expr<Arg>, int> = 0>
#endif
  constexpr decltype(auto)
  QR_decomposition(Arg&& arg)
  {
    return std::forward<Arg>(arg);
  }


  /// Concatenate diagonally.
#ifdef __cpp_concepts
  template<Eigen3::eigen_diagonal_expr V, Eigen3::eigen_diagonal_expr ... Vs>
#else
  template<typename V, typename ... Vs, std::enable_if_t<
    (Eigen3::eigen_diagonal_expr<V> and ... and Eigen3::eigen_diagonal_expr<Vs>), int> = 0>
#endif
  constexpr decltype(auto)
  concatenate_diagonal(V&& v, Vs&& ... vs)
  {
    if constexpr(sizeof...(Vs) > 0)
    {
      return MatrixTraits<V>::make(concatenate_vertical(base_matrix(std::forward<V>(v)), base_matrix(std::forward<Vs>(vs))...));
    }
    else
    {
      return std::forward<V>(v);
    }
  };


  // split functions for DiagonalMatrix are found in EigenSpecialMatrixOverloads


  /// Get element (i) of diagonal matrix arg.
#ifdef __cpp_concepts
  template<Eigen3::eigen_diagonal_expr Arg> requires (is_element_gettable_v<typename MatrixTraits<Arg>::BaseMatrix, 1> or
    is_element_gettable_v<typename MatrixTraits<Arg>::BaseMatrix, 2>)
#else
  template<typename Arg, std::enable_if_t<Eigen3::eigen_diagonal_expr<Arg> and
    (is_element_gettable_v<typename MatrixTraits<Arg>::BaseMatrix, 1> or
    is_element_gettable_v<typename MatrixTraits<Arg>::BaseMatrix, 2>), int> = 0>
#endif
  inline auto
  get_element(Arg&& arg, const std::size_t i)
  {
    if constexpr (is_element_gettable_v<typename MatrixTraits<Arg>::BaseMatrix, 1>)
      return get_element(base_matrix(std::forward<Arg>(arg)), i);
    else
      return get_element(base_matrix(std::forward<Arg>(arg)), i, 1);
  }


  /// Get element (i, j) of diagonal matrix arg.
#ifdef __cpp_concepts
  template<Eigen3::eigen_diagonal_expr Arg> requires (is_element_gettable_v<typename MatrixTraits<Arg>::BaseMatrix, 1> or
    is_element_gettable_v<typename MatrixTraits<Arg>::BaseMatrix, 2>)
#else
  template<typename Arg, std::enable_if_t<Eigen3::eigen_diagonal_expr<Arg> and
    (is_element_gettable_v<typename MatrixTraits<Arg>::BaseMatrix, 1> or
    is_element_gettable_v<typename MatrixTraits<Arg>::BaseMatrix, 2>), int> = 0>
#endif
  inline auto
  get_element(Arg&& arg, const std::size_t i, const std::size_t j)
  {
    if (i == j)
    {
      if constexpr (is_element_gettable_v<typename MatrixTraits<Arg>::BaseMatrix, 1>)
        return get_element(base_matrix(std::forward<Arg>(arg)), i);
      else
        return get_element(base_matrix(std::forward<Arg>(arg)), i, 1);
    }
    else
      return typename MatrixTraits<Arg>::Scalar(0);
  }


  /// Set element (i) of matrix arg to s.
#ifdef __cpp_concepts
  template<Eigen3::eigen_diagonal_expr Arg, typename Scalar> requires
    (not std::is_const_v<std::remove_reference_t<Arg>>) and
    (is_element_settable_v<typename MatrixTraits<Arg>::BaseMatrix, 1> or
      is_element_settable_v<typename MatrixTraits<Arg>::BaseMatrix, 2>)
#else
  template<typename Arg, typename Scalar, std::enable_if_t<
    Eigen3::eigen_diagonal_expr<Arg> and not std::is_const_v<std::remove_reference_t<Arg>> and
      (is_element_settable_v<typename MatrixTraits<Arg>::BaseMatrix, 1> or
        is_element_settable_v<typename MatrixTraits<Arg>::BaseMatrix, 2>), int> = 0>
#endif
  inline void
  set_element(Arg& arg, const Scalar s, const std::size_t i)
  {
    if constexpr (is_element_settable_v<typename MatrixTraits<Arg>::BaseMatrix, 1>)
      set_element(base_matrix(arg), s, i);
    else
      set_element(base_matrix(arg), s, i, 1);
  }


  /// Set element (i, j) of matrix arg to s.
#ifdef __cpp_concepts
  template<Eigen3::eigen_diagonal_expr Arg, typename Scalar> requires
    (not std::is_const_v<std::remove_reference_t<Arg>>) and
    (is_element_settable_v<typename MatrixTraits<Arg>::BaseMatrix, 1> or
      is_element_settable_v<typename MatrixTraits<Arg>::BaseMatrix, 2>)
#else
  template<typename Arg, typename Scalar, std::enable_if_t<
    Eigen3::eigen_diagonal_expr<Arg> and not std::is_const_v<std::remove_reference_t<Arg>> and
      (is_element_settable_v<typename MatrixTraits<Arg>::BaseMatrix, 1> or
        is_element_settable_v<typename MatrixTraits<Arg>::BaseMatrix, 2>), int> = 0>
#endif
  inline void
  set_element(Arg& arg, const Scalar s, const std::size_t i, const std::size_t j)
  {
    if (i == j)
    {
      if constexpr (is_element_settable_v<typename MatrixTraits<Arg>::BaseMatrix, 1>)
        set_element(base_matrix(arg), s, i);
      else
        set_element(base_matrix(arg), s, i, 1);
    }
    else
      throw std::out_of_range("Only diagonal elements of a diagonal matrix may be set.");
  }


  /**
   * Fill the diagonal of a square matrix with random values selected from a random distribution.
   * The Gaussian distribution has zero mean and standard deviation sigma (1, if not specified).
   **/
#ifdef __cpp_concepts
  template<
    Eigen3::eigen_diagonal_expr ReturnType,
    template<typename Scalar> typename distribution_type = std::normal_distribution,
    typename random_number_engine = std::mt19937,
    typename...Params>
#else
  template<
    typename ReturnType,
    template<typename Scalar> typename distribution_type = std::normal_distribution,
    typename random_number_engine = std::mt19937,
    typename...Params,
    std::enable_if_t<Eigen3::eigen_diagonal_expr<ReturnType>, int> = 0>
#endif
  static auto
  randomize(Params...params)
  {
    using Scalar = typename MatrixTraits<ReturnType>::Scalar;
    using B = typename MatrixTraits<ReturnType>::BaseMatrix;
    constexpr auto rows = MatrixTraits<B>::dimension;
    constexpr auto cols = MatrixTraits<B>::columns;
    using Ps = typename distribution_type<Scalar>::param_type;
    static_assert(std::is_constructible_v<Ps, Params...> or sizeof...(Params) == rows or sizeof...(Params) == rows * cols,
      "Params... must be (1) a parameter set or list of parameter sets, or "
      "(2) a list of parameter sets, one for each diagonal coefficient.");
    return MatrixTraits<ReturnType>::make(randomize<B, distribution_type, random_number_engine>(params...));
  }


  ///////////////////////////////
  //        Arithmetic         //
  ///////////////////////////////

#ifdef __cpp_concepts
  template<Eigen3::eigen_diagonal_expr Arg1, Eigen3::eigen_diagonal_expr Arg2> requires
    (not is_zero_v<Arg1>) and (not is_zero_v<Arg2>) and (not is_identity_v<Arg1>) and (not is_identity_v<Arg2>)
#else
  template<typename Arg1, typename Arg2, std::enable_if_t<
    Eigen3::eigen_diagonal_expr<Arg1> and Eigen3::eigen_diagonal_expr<Arg2> and
    not is_zero_v<Arg1> and not is_zero_v<Arg2> and not is_identity_v<Arg1> and not is_identity_v<Arg2>, int> = 0>
#endif
  inline auto operator+(Arg1&& arg1, Arg2&& arg2)
  {
    static_assert(MatrixTraits<Arg1>::dimension == MatrixTraits<Arg2>::dimension);
    auto ret = MatrixTraits<Arg1>::make(
      std::forward<Arg1>(arg1).base_matrix() + std::forward<Arg2>(arg2).base_matrix());
    if constexpr (not std::is_lvalue_reference_v<Arg1&&> or not std::is_lvalue_reference_v<Arg2&&>) return strict(std::move(ret)); else return ret;
  }


#ifdef __cpp_concepts
  template<typename Arg1, typename Arg2> requires (Eigen3::eigen_diagonal_expr<Arg1> and is_identity_v<Arg2>) or
    (is_identity_v<Arg1> and Eigen3::eigen_diagonal_expr<Arg2>)
#else
  template<typename Arg1, typename Arg2, std::enable_if_t<(Eigen3::eigen_diagonal_expr<Arg1> and is_identity_v<Arg2>) or
    (is_identity_v<Arg1> and Eigen3::eigen_diagonal_expr<Arg2>), int> = 0>
#endif
  inline auto operator+(Arg1&& arg1, Arg2&& arg2)
  {
    static_assert(MatrixTraits<Arg1>::dimension == MatrixTraits<Arg2>::dimension);
    static_assert(MatrixTraits<Arg1>::columns == MatrixTraits<Arg2>::columns);
    if constexpr(Eigen3::eigen_diagonal_expr<Arg1>)
    {
      using B = typename MatrixTraits<Arg1>::BaseMatrix;
      auto ret = MatrixTraits<Arg1>::make(base_matrix(std::forward<Arg1>(arg1)) + B::Constant(1));
      if constexpr (not std::is_lvalue_reference_v<Arg1&&> or not std::is_lvalue_reference_v<Arg2&&>)
        return strict(std::move(ret));
      else
        return ret;
    }
    else
    {
      using B = typename MatrixTraits<Arg2>::BaseMatrix;
      auto ret = MatrixTraits<Arg2>::make(B::Constant(1) + base_matrix(std::forward<Arg2>(arg2)));
      if constexpr (not std::is_lvalue_reference_v<Arg1&&> or not std::is_lvalue_reference_v<Arg2&&>)
        return strict(std::move(ret));
      else
        return ret;
    }
  }


#ifdef __cpp_concepts
  template<typename Arg1, typename Arg2> requires (Eigen3::eigen_diagonal_expr<Arg1> and is_zero_v<Arg2>) or
    (is_zero_v<Arg1> and Eigen3::eigen_diagonal_expr<Arg2>)
#else
  template<typename Arg1, typename Arg2, std::enable_if_t<(Eigen3::eigen_diagonal_expr<Arg1> and is_zero_v<Arg2>) or
    (is_zero_v<Arg1> and Eigen3::eigen_diagonal_expr<Arg2>), int> = 0>
#endif
  constexpr decltype(auto) operator+(Arg1&& arg1, Arg2&& arg2)
  {
    static_assert(MatrixTraits<Arg1>::dimension == MatrixTraits<Arg2>::dimension);
    static_assert(MatrixTraits<Arg1>::columns == MatrixTraits<Arg2>::columns);
    if constexpr(Eigen3::eigen_diagonal_expr<Arg1>)
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
  template<Eigen3::eigen_diagonal_expr Arg1, Eigen3::eigen_diagonal_expr Arg2> requires
    (not is_zero_v<Arg1>) and (not is_zero_v<Arg2>) and (not is_identity_v<Arg1>) and (not is_identity_v<Arg2>)
#else
  template<typename Arg1, typename Arg2, std::enable_if_t<
    Eigen3::eigen_diagonal_expr<Arg1> and Eigen3::eigen_diagonal_expr<Arg2> and
      not is_zero_v<Arg1> and not is_zero_v<Arg2> and not is_identity_v<Arg1> and not is_identity_v<Arg2>, int> = 0>
#endif
  inline auto operator-(Arg1&& arg1, Arg2&& arg2)
  {
    static_assert(MatrixTraits<Arg1>::dimension == MatrixTraits<Arg2>::dimension);
    auto ret = MatrixTraits<Arg1>::make(
      std::forward<Arg1>(arg1).base_matrix() - std::forward<Arg2>(arg2).base_matrix());
    if constexpr (not std::is_lvalue_reference_v<Arg1&&> or not std::is_lvalue_reference_v<Arg2&&>) return strict(std::move(ret)); else return ret;
  }


#ifdef __cpp_concepts
  template<typename Arg1, typename Arg2> requires
    (Eigen3::eigen_diagonal_expr<Arg1> and is_identity_v<Arg2>) or (is_identity_v<Arg1> and Eigen3::eigen_diagonal_expr<Arg2>)
#else
  template<typename Arg1, typename Arg2, std::enable_if_t<(Eigen3::eigen_diagonal_expr<Arg1> and is_identity_v<Arg2>) or
    (is_identity_v<Arg1> and Eigen3::eigen_diagonal_expr<Arg2>), int> = 0>
#endif
  inline auto operator-(Arg1&& arg1, Arg2&& arg2)
  {
    static_assert(MatrixTraits<Arg1>::dimension == MatrixTraits<Arg2>::dimension);
    static_assert(MatrixTraits<Arg1>::columns == MatrixTraits<Arg2>::columns);
    if constexpr(Eigen3::eigen_diagonal_expr<Arg1>)
    {
      using B = typename MatrixTraits<Arg1>::BaseMatrix;
      auto ret = MatrixTraits<Arg1>::make(std::forward<Arg1>(arg1).base_matrix() - B::Constant(1));
      if constexpr (not std::is_lvalue_reference_v<Arg1&&> or not std::is_lvalue_reference_v<Arg2&&>) return strict(std::move(ret)); else return ret;
    }
    else
    {
      using B = typename MatrixTraits<Arg2>::BaseMatrix;
      auto ret = MatrixTraits<Arg2>::make(B::Constant(1) - std::forward<Arg2>(arg2).base_matrix());
      if constexpr (not std::is_lvalue_reference_v<Arg1&&> or not std::is_lvalue_reference_v<Arg2&&>) return strict(std::move(ret)); else return ret;
    }
  }


#ifdef __cpp_concepts
  template<typename Arg1, typename Arg2> requires
    (Eigen3::eigen_diagonal_expr<Arg1> and is_zero_v<Arg2>) or (is_zero_v<Arg1> and Eigen3::eigen_diagonal_expr<Arg2>)
#else
  template<typename Arg1, typename Arg2, std::enable_if_t<
    (Eigen3::eigen_diagonal_expr<Arg1> and is_zero_v<Arg2>) or (is_zero_v<Arg1> and Eigen3::eigen_diagonal_expr<Arg2>), int> = 0>
#endif
  constexpr decltype(auto) operator-(Arg1&& arg1, Arg2&& arg2)
  {
    static_assert(MatrixTraits<Arg1>::dimension == MatrixTraits<Arg2>::dimension);
    static_assert(MatrixTraits<Arg1>::columns == MatrixTraits<Arg2>::columns);
    if constexpr(Eigen3::eigen_diagonal_expr<Arg1>)
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
  template<Eigen3::eigen_diagonal_expr Arg, std::convertible_to<typename MatrixTraits<Arg>::Scalar> S>
#else
  template<typename Arg, typename S, std::enable_if_t<
    Eigen3::eigen_diagonal_expr<Arg> and std::is_convertible_v<S, typename MatrixTraits<Arg>::Scalar>, int> = 0>
#endif
  inline auto operator*(Arg&& arg, const S scale)
  {
    auto ret = MatrixTraits<Arg>::make(std::forward<Arg>(arg).base_matrix() * scale);
    if constexpr (not std::is_lvalue_reference_v<Arg&&>) return strict(std::move(ret)); else return ret;
  }


#ifdef __cpp_concepts
  template<Eigen3::eigen_diagonal_expr Arg, std::convertible_to<typename MatrixTraits<Arg>::Scalar> S>
#else
  template<typename Arg, typename S, std::enable_if_t<
    Eigen3::eigen_diagonal_expr<Arg> and std::is_convertible_v<S, typename MatrixTraits<Arg>::Scalar>, int> = 0>
#endif
  inline auto operator*(const S scale, Arg&& arg)
  {
    auto ret = MatrixTraits<Arg>::make(scale * std::forward<Arg>(arg).base_matrix());
    if constexpr (not std::is_lvalue_reference_v<Arg&&>) return strict(std::move(ret)); else return ret;
  }

#ifdef __cpp_concepts
  template<Eigen3::eigen_diagonal_expr Arg, std::convertible_to<typename MatrixTraits<Arg>::Scalar> S>
#else
  template<typename Arg, typename S, std::enable_if_t<
    Eigen3::eigen_diagonal_expr<Arg> and std::is_convertible_v<S, typename MatrixTraits<Arg>::Scalar>, int> = 0>
#endif
  inline auto operator/(Arg&& arg, const S scale)
  {
    auto ret = MatrixTraits<Arg>::make(std::forward<Arg>(arg).base_matrix() / scale);
    if constexpr (not std::is_lvalue_reference_v<Arg&&>) return strict(std::move(ret)); else return ret;
  }


  ////

#ifdef __cpp_concepts
  template<Eigen3::eigen_diagonal_expr Arg1, Eigen3::eigen_diagonal_expr Arg2> requires
    (not is_zero_v<Arg1>) and (not is_zero_v<Arg2>) and (not is_identity_v<Arg1>) and (not is_identity_v<Arg2>)
#else
  template<typename Arg1, typename Arg2, std::enable_if_t<
    Eigen3::eigen_diagonal_expr<Arg1> and Eigen3::eigen_diagonal_expr<Arg2> and
    not is_zero_v<Arg1> and not is_zero_v<Arg2> and not is_identity_v<Arg1> and not is_identity_v<Arg2>, int> = 0>
#endif
  inline auto operator*(Arg1&& arg1, Arg2&& arg2)
  {
    static_assert(MatrixTraits<Arg1>::dimension == MatrixTraits<Arg2>::dimension);
    auto ret = MatrixTraits<Arg1>::make(
      (std::forward<Arg1>(arg1).base_matrix().array() * std::forward<Arg2>(arg2).base_matrix().array()).matrix());
    if constexpr (not std::is_lvalue_reference_v<Arg1&&> or not std::is_lvalue_reference_v<Arg2&&>) return strict(std::move(ret)); else return ret;
  }


#ifdef __cpp_concepts
  template<typename Arg1, typename Arg2> requires
    (Eigen3::eigen_diagonal_expr<Arg1> and is_identity_v<Arg2>) or (is_identity_v<Arg1> and Eigen3::eigen_diagonal_expr<Arg2>)
#else
  template<typename Arg1, typename Arg2, std::enable_if_t<(Eigen3::eigen_diagonal_expr<Arg1> and is_identity_v<Arg2>) or
    (is_identity_v<Arg1> and Eigen3::eigen_diagonal_expr<Arg2>), int> = 0>
#endif
  inline auto operator*(Arg1&& arg1, Arg2&& arg2)
  {
    static_assert(MatrixTraits<Arg1>::dimension == MatrixTraits<Arg2>::dimension);
    static_assert(MatrixTraits<Arg1>::columns == MatrixTraits<Arg2>::columns);
    if constexpr(Eigen3::eigen_diagonal_expr<Arg1>)
    {
      return std::forward<Arg1>(arg1);
    }
    else
    {
      return std::forward<Arg2>(arg2);
    }
  }


#ifdef __cpp_concepts
  template<typename Arg1, typename Arg2> requires
    (Eigen3::eigen_diagonal_expr<Arg1> and is_zero_v<Arg2>) or (is_zero_v<Arg1> and Eigen3::eigen_diagonal_expr<Arg2>)
#else
  template<typename Arg1, typename Arg2, std::enable_if_t<(Eigen3::eigen_diagonal_expr<Arg1> and is_zero_v<Arg2>) or
    (is_zero_v<Arg1> and Eigen3::eigen_diagonal_expr<Arg2>), int> = 0>
#endif
  constexpr decltype(auto) operator*(Arg1&& arg1, Arg2&& arg2)
  {
    static_assert(MatrixTraits<Arg1>::columns == MatrixTraits<Arg2>::dimension);
    if constexpr(is_zero_v<Arg1>)
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
  template<typename Arg1, typename Arg2> requires
    ((Eigen3::eigen_diagonal_expr<Arg1> and Eigen3::eigen_matrix<Arg2>) or
      (Eigen3::eigen_matrix<Arg1> and Eigen3::eigen_diagonal_expr<Arg2>)) and
    (not is_identity_v<Arg1>) and (not is_identity_v<Arg2>) and (not is_zero_v<Arg1>) and (not is_zero_v<Arg2>)
#else
  template<typename Arg1, typename Arg2,
    std::enable_if_t<((Eigen3::eigen_diagonal_expr<Arg1> and Eigen3::eigen_matrix<Arg2>) or
      (Eigen3::eigen_matrix<Arg1> and Eigen3::eigen_diagonal_expr<Arg2>)) and
      not is_identity_v<Arg1> and not is_identity_v<Arg2> and not is_zero_v<Arg1> and not is_zero_v<Arg2>, int> = 0>
#endif
  inline auto operator*(Arg1&& arg1, Arg2&& arg2)
  {
    static_assert(MatrixTraits<Arg1>::columns == MatrixTraits<Arg2>::dimension);
    if constexpr(Eigen3::eigen_diagonal_expr<Arg1>)
    {
      return strict(std::forward<Arg1>(arg1).base_matrix().asDiagonal() * std::forward<Arg2>(arg2));
    }
    else
    {
      return strict(std::forward<Arg1>(arg1) * std::forward<Arg2>(arg2).base_matrix().asDiagonal());
    }
  }


#ifdef __cpp_concepts
  template<Eigen3::eigen_diagonal_expr Arg>
#else
  template<typename Arg, std::enable_if_t<Eigen3::eigen_diagonal_expr<Arg>, int> = 0>
#endif
  inline auto operator-(Arg&& arg)
  {
    auto ret = MatrixTraits<Arg>::make(-std::forward<Arg>(arg).base_matrix());
    if constexpr (not std::is_lvalue_reference_v<Arg&&>) return strict(std::move(ret)); else return ret;
  }

} // namespace OpenKalman::Eigen3

#endif //OPENKALMAN_EIGEN3_DIAGONALMATRIX_HPP
