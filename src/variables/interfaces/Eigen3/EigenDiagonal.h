/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef OPENKALMAN_EIGENDIAGONAL_H
#define OPENKALMAN_EIGENDIAGONAL_H

namespace OpenKalman
{
  template<typename BaseMatrix>
  struct EigenDiagonal
    : internal::MatrixBase<EigenDiagonal<BaseMatrix>, BaseMatrix>
  {
    using Base = internal::MatrixBase<EigenDiagonal, BaseMatrix>;
    using Scalar = typename MatrixTraits<BaseMatrix>::Scalar;
    static constexpr auto dimension = MatrixTraits<BaseMatrix>::dimension;

    /// Default constructor.
    EigenDiagonal() : Base() {}

    /// Copy constructor.
    EigenDiagonal(const EigenDiagonal& other) : EigenDiagonal(other.base_matrix()) {}

    /// Move constructor.
    EigenDiagonal(EigenDiagonal&& other) noexcept : EigenDiagonal(std::move(other).base_matrix()) {}

    /// Construct from a compatible EigenDiagonal.
    template<typename Arg, std::enable_if_t<is_EigenDiagonal_v<Arg>, int> = 0>
    EigenDiagonal(Arg&& other) noexcept : Base(std::forward<Arg>(other).base_matrix())
    {
      static_assert(MatrixTraits<Arg>::dimension == dimension);
    }

    /// Construct from a column vector matrix.
    template<typename Arg, std::enable_if_t<is_Eigen_matrix_v<Arg> and MatrixTraits<Arg>::columns == 1, int> = 0>
    EigenDiagonal(Arg&& arg) noexcept : Base(std::forward<Arg>(arg)) {}

    /// Construct from a compatible diagonal native matrix that is not EigenDiagonal.
    template<typename Arg,
      std::enable_if_t<is_native_Eigen_type_v<Arg> and is_diagonal_v<Arg> and
      not is_EigenDiagonal_v<Arg> and (MatrixTraits<Arg>::columns > 1), int> = 0>
    EigenDiagonal(Arg&& other) noexcept : Base(std::forward<Arg>(other).diagonal())
    {
      static_assert(MatrixTraits<Arg>::dimension == dimension);
    }

    /// Construct from a square zero matrix.
    template<typename Arg, std::enable_if_t<is_zero_v<Arg> and (MatrixTraits<Arg>::columns > 1), int> = 0>
    EigenDiagonal(const Arg&) : Base(MatrixTraits<Eigen::Matrix<Scalar, dimension, 1>>::zero())
    {
      static_assert(MatrixTraits<Arg>::dimension == dimension);
      static_assert(MatrixTraits<Arg>::columns == dimension);
    }

    /// Construct from an identity matrix.
    template<typename Arg, std::enable_if_t<is_identity_v<Arg> and (MatrixTraits<Arg>::columns > 1), int> = 0>
    EigenDiagonal(const Arg&) : Base(Eigen::Matrix<Scalar, dimension, 1>::Constant(1))
    {
      static_assert(MatrixTraits<Arg>::dimension == dimension);
      static_assert(MatrixTraits<Arg>::columns == dimension);
    }

    /// Construct from a list of scalar coefficients defining the diagonal.
    template<typename ... Args, std::enable_if_t<std::conjunction_v<std::is_convertible<Args, const Scalar>...>, int> = 0>
    EigenDiagonal(Args ... args) : EigenDiagonal(MatrixTraits<BaseMatrix>::make(args...)) {}

    /// Copy assignment operator.
    auto& operator=(const EigenDiagonal& other)
    {
      if (this != &other) this->base_matrix() = other.base_matrix();
      return *this;
    }

    /// Move assignment operator.
    auto& operator=(EigenDiagonal&& other) noexcept
    {
      if (this != &other) this->base_matrix() = std::move(other).base_matrix();
      return *this;
    }

    /// Assign from a compatible EigenDiagonal.
    template<typename Arg, std::enable_if_t<is_EigenDiagonal_v<Arg>, int> = 0>
    auto& operator=(Arg& other)
    {
      if constexpr (std::is_same_v<std::decay_t<Arg>, EigenDiagonal>) if (this == &other) return *this;
      this->base_matrix() = std::forward<Arg>(other).base_matrix();
      return *this;
    }

    /// Assign from a square zero matrix.
    template<typename Arg, std::enable_if_t<is_zero_v<Arg> and (MatrixTraits<Arg>::columns > 1), int> = 0>
    auto& operator=(const Arg&)
    {
      static_assert(MatrixTraits<Arg>::dimension == dimension);
      static_assert(MatrixTraits<Arg>::columns == dimension);
      this->base_matrix() = MatrixTraits<Eigen::Matrix<Scalar, dimension, 1>>::zero();
      return *this;
    }

    /// Assign from an identity matrix.
    template<typename Arg, std::enable_if_t<is_identity_v<Arg> and (MatrixTraits<Arg>::columns > 1), int> = 0>
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
    auto& operator+=(const EigenDiagonal<Arg>& arg)
    {
      static_assert(MatrixTraits<Arg>::dimension == dimension);
      this->base_matrix() += arg.base_matrix();
      return *this;
    }

    template<typename Arg>
    auto& operator-=(const EigenDiagonal<Arg>& arg)
    {
      static_assert(MatrixTraits<Arg>::dimension == dimension);
      this->base_matrix() -= arg.base_matrix();
      return *this;
    }

    template<typename S, std::enable_if_t<std::is_convertible_v<S, Scalar>, int> = 0>
    auto& operator*=(const S s)
    {
      this->base_matrix() *= s;
      return *this;
    }

    template<typename S, std::enable_if_t<std::is_convertible_v<S, Scalar>, int> = 0>
    auto& operator/=(const S s)
    {
      this->base_matrix() /= s;
      return *this;
    }

    template<typename Arg>
    auto& operator*=(const EigenDiagonal<Arg>& arg)
    {
      static_assert(MatrixTraits<Arg>::dimension == dimension);
      this->base_matrix() = this->base_matrix().array() * arg.base_matrix().array();
      return *this;
    }

    auto square() const
    {
      auto b = this->base_matrix().array().square().matrix();
      return EigenDiagonal<decltype(b)>(std::move(b));
    }

    auto square_root() const
    {
      auto b = this->base_matrix().cwiseSqrt();
      return EigenDiagonal<decltype(b)>(std::move(b));
    }


    decltype(auto) operator()(std::size_t i, std::size_t j) & { return (*this).Base::operator()(i, j); }

    decltype(auto) operator()(std::size_t i, std::size_t j) && noexcept { return std::move(*this).Base::operator()(i, j); }

    decltype(auto) operator()(std::size_t i, std::size_t j) const& { return (*this).Base::operator()(i, j); }

    decltype(auto) operator()(std::size_t i, std::size_t j) const&& noexcept { return std::move(*this).Base::operator()(i, j); }


    decltype(auto) operator()(std::size_t i) & { return (*this).base_matrix()(i); }

    decltype(auto) operator()(std::size_t i) && noexcept { return std::move(*this).base_matrix()(i); }

    decltype(auto) operator()(std::size_t i) const& { return (*this).base_matrix()(i); }

    decltype(auto) operator()(std::size_t i) const&& noexcept { return std::move(*this).base_matrix()(i); }

  };


  /////////////////////////////////////
  //        Deduction guides         //
  /////////////////////////////////////

  template<typename M, std::enable_if_t<is_Eigen_matrix_v<M> and MatrixTraits<M>::columns == 1, int> = 0>
  EigenDiagonal(M&&) -> EigenDiagonal<std::decay_t<M>>;

  template<typename M, std::enable_if_t<is_native_Eigen_type_v<M> and is_diagonal_v<M> and
    not is_EigenDiagonal_v<M> and (MatrixTraits<M>::columns > 1) , int> = 0>
  EigenDiagonal(M&&)
  -> EigenDiagonal<Eigen::Diagonal<M, 0>>;

  template<typename Arg, std::enable_if_t<is_zero_v<Arg> and (MatrixTraits<Arg>::columns > 1) and
    MatrixTraits<Arg>::dimension == MatrixTraits<Arg>::columns, int> = 0>
  EigenDiagonal(const Arg&)
  -> EigenDiagonal<EigenZero<Eigen::Matrix<typename MatrixTraits<Arg>::Scalar, MatrixTraits<Arg>::dimension, 1>>>;

  template<typename Arg, std::enable_if_t<is_identity_v<Arg> and (MatrixTraits<Arg>::columns > 1) and
    MatrixTraits<Arg>::dimension == MatrixTraits<Arg>::columns, int> = 0>
  EigenDiagonal(const Arg&)
  -> EigenDiagonal<typename Eigen::Matrix<typename MatrixTraits<Arg>::Scalar, MatrixTraits<Arg>::dimension, 1>::ConstantReturnType>;

  template<typename ... Args, std::enable_if_t<std::conjunction_v<std::is_arithmetic<Args>...>, int> = 0>
  EigenDiagonal(Args ...)
  -> EigenDiagonal<Eigen::Matrix<std::decay_t<std::common_type_t<Args...>>, sizeof...(Args), 1>>;


  /////////////////////////////////
  //        MatrixTraits         //
  /////////////////////////////////

  template<typename ArgType>
  struct MatrixTraits<OpenKalman::EigenDiagonal<ArgType>>
  {
    using BaseMatrix = ArgType;
    using Scalar = typename MatrixTraits<BaseMatrix>::Scalar;
    static constexpr auto dimension = MatrixTraits<BaseMatrix>::dimension;
    static constexpr auto columns = dimension;

    template<typename Derived>
    using MatrixBaseType = internal::EigenMatrixBase<Derived, OpenKalman::EigenDiagonal<std::decay_t<BaseMatrix>>>;

    template<typename Derived>
    using CovarianceBaseType = internal::EigenCovarianceBase<Derived, EigenDiagonal<std::decay_t<ArgType>>>;

    template<std::size_t rows = dimension, std::size_t cols = columns, typename S = Scalar>
    using StrictMatrix = typename MatrixTraits<std::decay_t<BaseMatrix>>::template StrictMatrix<rows, cols, S>;

    /// @TODO: Can we change to EigenDiagonal, since it is both self-adjoint and triangular?
    template<TriangleType storage_triangle = TriangleType::diagonal, std::size_t dim = dimension, typename S = Scalar>
    using SelfAdjointBaseType = EigenSelfAdjointMatrix<StrictMatrix<dim, dim, S>, storage_triangle>;

    template<TriangleType triangle_type = TriangleType::diagonal, std::size_t dim = dimension, typename S = Scalar>
    using TriangularBaseType = EigenTriangularMatrix<StrictMatrix<dim, dim, S>, triangle_type>;

    template<std::size_t dim = dimension, typename S = Scalar>
    using DiagonalBaseType = EigenDiagonal<StrictMatrix<dim, 1, S>>;

    template<typename Arg, std::enable_if_t<is_Eigen_matrix_v<Arg>, int> = 0>
    static auto make(Arg&& arg) noexcept
    {
      static_assert(MatrixTraits<Arg>::columns == 1);
      return EigenDiagonal<std::decay_t<Arg>>(std::forward<Arg>(arg));
    }

    /// Make diagonal matrix using a list of coefficients defining the diagonal.
    template<typename ... Args,
      std::enable_if_t<std::conjunction_v<std::is_convertible<Args, const Scalar>...>, int> = 0,
      std::enable_if_t<sizeof...(Args) != dimension * dimension or dimension == 1, int> = 0>
    static auto make(Args ... args)
    {
      static_assert(sizeof...(Args) == dimension);
      return make(MatrixTraits<BaseMatrix>::make(args...));
    }

    /// Make diagonal matrix using a list of coefficients in row-major order (ignoring non-diagonal coefficients).
    template<typename ... Args,
      std::enable_if_t<std::conjunction_v<std::is_convertible<Args, const Scalar>...>, int> = 0,
      std::enable_if_t<sizeof...(Args) == dimension * dimension and dimension != 1, int> = 0>
    static auto
    make(Args ... args)
    {
      return make(strict(MatrixTraits<StrictMatrix<>>::make(args...).diagonal()));
    }

    static auto zero() { return MatrixTraits<StrictMatrix<>>::zero(); }

    static auto identity() { return MatrixTraits<StrictMatrix<>>::identity(); }

  };


  //////////////////////////////
  //        Overloads         //
  //////////////////////////////

  template<typename Arg, std::enable_if_t<is_EigenDiagonal_v<Arg>, int> = 0>
  constexpr decltype(auto)
  base_matrix(Arg&& arg) { return std::forward<Arg>(arg).base_matrix(); }


  /// Convert to strict version
  template<typename Arg, std::enable_if_t<is_EigenDiagonal_v<Arg>, int> = 0>
  inline decltype(auto)
  strict(Arg&& arg)
  {
    if constexpr(is_strict_v<typename MatrixTraits<Arg>::BaseMatrix>)
    {
      return std::forward<Arg>(arg);
    }
    else
    {
      return EigenDiagonal(strict(base_matrix(std::forward<Arg>(arg))));
    }
  }


  template<typename Arg, std::enable_if_t<is_EigenDiagonal_v<Arg>, int> = 0>
  inline auto
  Cholesky_square(Arg&& arg)
  {
    auto b = base_matrix(std::forward<Arg>(arg)).array().square().matrix();
    return EigenDiagonal<decltype(b)>(std::move(b));
  }


  template<typename Arg, std::enable_if_t<is_EigenDiagonal_v<Arg>, int> = 0>
  inline auto
  Cholesky_factor(Arg&& arg)
  {
    auto b = base_matrix(std::forward<Arg>(arg)).cwiseSqrt();
    return EigenDiagonal<decltype(b)>(std::move(b));
  }


  template<typename Arg, std::enable_if_t<is_EigenDiagonal_v<Arg>, int> = 0>
  constexpr decltype(auto)
  transpose(Arg&& arg) noexcept
  {
    return std::forward<Arg>(arg);
  }


  template<typename Arg, std::enable_if_t<is_EigenDiagonal_v<Arg>, int> = 0>
  constexpr decltype(auto)
  adjoint(Arg&& arg) noexcept
  {
    return EigenDiagonal(base_matrix(std::forward<Arg>(arg)).conjugate());
  }


  template<typename Arg, std::enable_if_t<is_EigenDiagonal_v<Arg>, int> = 0>
  inline auto
  determinant(Arg&& arg) noexcept
  {
    static_assert(MatrixTraits<Arg>::dimension == MatrixTraits<Arg>::columns);
    return base_matrix(std::forward<Arg>(arg)).prod();
  }


  template<typename Arg, std::enable_if_t<is_EigenDiagonal_v<Arg>, int> = 0>
  inline auto
  trace(Arg&& arg) noexcept
  {
    static_assert(MatrixTraits<Arg>::dimension == MatrixTraits<Arg>::columns);
    return base_matrix(std::forward<Arg>(arg)).sum();
  }


  /// Solve the equation AX = B for X. A is a diagonal matrix.
  template<
    typename A, typename B,
    std::enable_if_t<is_EigenDiagonal_v<A>, int> = 0,
    std::enable_if_t<is_Eigen_matrix_v<B>, int> = 0>
  inline auto
  solve(const A& a, const B& b)
  {
    static_assert(MatrixTraits<A>::dimension == MatrixTraits<B>::dimension);
    return (b.array().colwise() / base_matrix(a).array()).matrix();
  }


  /// Create a column vector from a diagnoal matrix. (Same as base_matrix()).
  template<typename Arg, std::enable_if_t<is_EigenDiagonal_v<Arg>, int> = 0>
  constexpr decltype(auto)
  reduce_columns(Arg&& arg) noexcept
  {
    return base_matrix(std::forward<Arg>(arg));
  }


  /// Perform an LQ decomposition. Since it is diagonal, it returns the matrix unchanged.
  template<typename A, std::enable_if_t<is_EigenDiagonal_v<A>, int> = 0>
  constexpr decltype(auto)
  LQ_decomposition(A&& a)
  {
    return std::forward<A>(a);
  }


  /// Perform a QR decomposition. Since it is diagonal, it returns the matrix unchanged.
  template<typename A, std::enable_if_t<is_EigenDiagonal_v<A>, int> = 0>
  constexpr decltype(auto)
  QR_decomposition(A&& a)
  {
    return std::forward<A>(a);
  }


  /// Concatenate diagonally.
  template<typename V, typename ... Vs,
    std::enable_if_t<std::conjunction_v<is_EigenDiagonal<V>, is_EigenDiagonal<Vs>...>, int> = 0>
  constexpr decltype(auto)
  concatenate(V&& v, Vs&& ... vs)
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


  /// Concatenate diagonally.
  template<typename V, typename ... Vs,
    std::enable_if_t<std::conjunction_v<is_EigenDiagonal<V>, is_EigenDiagonal<Vs>...>, int> = 0>
  constexpr decltype(auto)
  concatenate_diagonal(V&& v, Vs&& ... vs)
  {
    return concatenate(std::forward<V>(v), std::forward<Vs>(vs)...);
  };


  /// Split a diagonal matrix diagonally.
  template<std::size_t ... cuts, typename Arg, std::enable_if_t<is_EigenDiagonal_v<Arg>, int> = 0>
  inline auto
  split(Arg&& arg)
  {
    static_assert((0 + ... + cuts) <= MatrixTraits<Arg>::dimension);
    if constexpr(sizeof...(cuts) == 0)
    {
      return std::tuple {};
    }
    else if constexpr(sizeof...(cuts) == 1 and (0 + ... + cuts) == MatrixTraits<Arg>::dimension)
    {
      return std::tuple {std::forward<Arg>(arg)};
    }
    else
    {
      return std::apply([](const auto&...bs){ return std::tuple {MatrixTraits<Arg>::make(strict(bs))...}; },
        split_vertical<cuts...>(base_matrix(std::forward<Arg>(arg))));
    }
  }


  /**
   * Fill the diagonal of a square matrix with random values selected from a random distribution.
   * The Gaussian distribution has zero mean and standard deviation sigma (1, if not specified).
   **/
  template<
    typename ReturnType,
    template<typename Scalar> typename distribution_type = std::normal_distribution,
    typename random_number_engine = std::mt19937,
    typename...Params,
    std::enable_if_t<is_EigenDiagonal_v<ReturnType>, int> = 0>
  static auto
  randomize(Params...params)
  {
    using Scalar = typename MatrixTraits<ReturnType>::Scalar;
    using Ps = typename distribution_type<Scalar>::param_type;
    static_assert(std::is_constructible_v<Ps, Params...>,
      "Parameters params... must be constructor arguments of distribution_type<RealType>::param_type.");
    auto ps = Ps {params...};
    using B = typename MatrixTraits<ReturnType>::BaseMatrix;
    return MatrixTraits<ReturnType>::make(randomize<B, distribution_type, random_number_engine>(ps));
  }


  ///////////////////////////////
  //        Arithmetic         //
  ///////////////////////////////

  template<typename Arg1, typename Arg2,
    std::enable_if_t<is_EigenDiagonal_v<Arg1> and is_EigenDiagonal_v<Arg2> and
      not is_zero_v<Arg1> and not is_zero_v<Arg2> and not is_identity_v<Arg1> and not is_identity_v<Arg2>, int> = 0>
  inline auto operator+(Arg1&& arg1, Arg2&& arg2)
  {
    static_assert(MatrixTraits<Arg1>::dimension == MatrixTraits<Arg2>::dimension);
    return MatrixTraits<Arg1>::make(
      std::forward<Arg1>(arg1).base_matrix() + std::forward<Arg2>(arg2).base_matrix());
  }


  template<typename Arg1, typename Arg2,
    std::enable_if_t<(is_EigenDiagonal_v<Arg1> and is_identity_v<Arg2>) or
      (is_identity_v<Arg1> and is_EigenDiagonal_v<Arg2>), int> = 0>
  inline auto operator+(Arg1&& arg1, Arg2&& arg2)
  {
    static_assert(MatrixTraits<Arg1>::dimension == MatrixTraits<Arg2>::dimension);
    static_assert(MatrixTraits<Arg1>::columns == MatrixTraits<Arg2>::columns);
    if constexpr(OpenKalman::is_EigenDiagonal_v<Arg1>)
    {
      using B = typename MatrixTraits<Arg1>::BaseMatrix;
      return MatrixTraits<Arg1>::make(base_matrix(std::forward<Arg1>(arg1)) + B::Constant(1));
    }
    else
    {
      using B = typename MatrixTraits<Arg2>::BaseMatrix;
      return MatrixTraits<Arg2>::make(B::Constant(1) + base_matrix(std::forward<Arg2>(arg2)));
    }
  }


  template<typename Arg1, typename Arg2,
    std::enable_if_t<(is_EigenDiagonal_v<Arg1> and is_zero_v<Arg2>) or
      (is_zero_v<Arg1> and is_EigenDiagonal_v<Arg2>), int> = 0>
  constexpr decltype(auto) operator+(Arg1&& arg1, Arg2&& arg2)
  {
    static_assert(MatrixTraits<Arg1>::dimension == MatrixTraits<Arg2>::dimension);
    static_assert(MatrixTraits<Arg1>::columns == MatrixTraits<Arg2>::columns);
    if constexpr(OpenKalman::is_EigenDiagonal_v<Arg1>)
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
    std::enable_if_t<is_EigenDiagonal_v<Arg1> and is_EigenDiagonal_v<Arg2> and
      not is_zero_v<Arg1> and not is_zero_v<Arg2> and not is_identity_v<Arg1> and not is_identity_v<Arg2>, int> = 0>
  inline auto operator-(Arg1&& arg1, Arg2&& arg2)
  {
    static_assert(MatrixTraits<Arg1>::dimension == MatrixTraits<Arg2>::dimension);
    return MatrixTraits<Arg1>::make(
      std::forward<Arg1>(arg1).base_matrix() - std::forward<Arg2>(arg2).base_matrix());
  }


  template<typename Arg1, typename Arg2,
    std::enable_if_t<(is_EigenDiagonal_v<Arg1> and is_identity_v<Arg2>) or
      (is_identity_v<Arg1> and is_EigenDiagonal_v<Arg2>), int> = 0>
  inline auto operator-(Arg1&& arg1, Arg2&& arg2)
  {
    static_assert(MatrixTraits<Arg1>::dimension == MatrixTraits<Arg2>::dimension);
    static_assert(MatrixTraits<Arg1>::columns == MatrixTraits<Arg2>::columns);
    if constexpr(OpenKalman::is_EigenDiagonal_v<Arg1>)
    {
      using B = typename MatrixTraits<Arg1>::BaseMatrix;
      return MatrixTraits<Arg1>::make(std::forward<Arg1>(arg1).base_matrix() - B::Constant(1));
    }
    else
    {
      using B = typename MatrixTraits<Arg2>::BaseMatrix;
      return MatrixTraits<Arg2>::make(B::Constant(1) - std::forward<Arg2>(arg2).base_matrix());
    }
  }


  template<typename Arg1, typename Arg2,
    std::enable_if_t<(is_EigenDiagonal_v<Arg1> and is_zero_v<Arg2>) or
      (is_zero_v<Arg1> and is_EigenDiagonal_v<Arg2>), int> = 0>
  constexpr decltype(auto) operator-(Arg1&& arg1, Arg2&& arg2)
  {
    static_assert(MatrixTraits<Arg1>::dimension == MatrixTraits<Arg2>::dimension);
    static_assert(MatrixTraits<Arg1>::columns == MatrixTraits<Arg2>::columns);
    if constexpr(OpenKalman::is_EigenDiagonal_v<Arg1>)
    {
      return std::forward<Arg1>(arg1);
    }
    else
    {
      return -std::forward<Arg2>(arg2);
    }
  }


  ////

  template<typename Arg, typename S,
    std::enable_if_t<is_EigenDiagonal_v<Arg> and std::is_convertible_v<S, typename MatrixTraits<Arg>::Scalar>, int> = 0>
  inline auto operator*(Arg&& arg, const S scale)
  {
    return MatrixTraits<Arg>::make(std::forward<Arg>(arg).base_matrix() * scale);
  }


  template<typename Arg, typename S,
    std::enable_if_t<is_EigenDiagonal_v<Arg> and std::is_convertible_v<S, typename MatrixTraits<Arg>::Scalar>, int> = 0>
  inline auto operator*(const S scale, Arg&& arg)
  {
    return MatrixTraits<Arg>::make(scale * std::forward<Arg>(arg).base_matrix());
  }

  template<typename Arg, typename S,
    std::enable_if_t<is_EigenDiagonal_v<Arg> and std::is_convertible_v<S, typename MatrixTraits<Arg>::Scalar>, int> = 0>
  inline auto operator/(Arg&& arg, const S scale)
  {
    return MatrixTraits<Arg>::make(std::forward<Arg>(arg).base_matrix() / scale);
  }


  ////

  template<
    typename Arg1, typename Arg2,
    std::enable_if_t<is_EigenDiagonal_v<Arg1> and is_EigenDiagonal_v<Arg2> and
      not is_zero_v<Arg1> and not is_zero_v<Arg2> and not is_identity_v<Arg1> and not is_identity_v<Arg2>, int> = 0>
  inline auto operator*(Arg1&& arg1, Arg2&& arg2)
  {
    static_assert(MatrixTraits<Arg1>::dimension == MatrixTraits<Arg2>::dimension);
    return MatrixTraits<Arg1>::make(
      (std::forward<Arg1>(arg1).base_matrix().array() * std::forward<Arg2>(arg2).base_matrix().array()).matrix());
  }


  template<typename Arg1, typename Arg2,
    std::enable_if_t<(is_EigenDiagonal_v<Arg1> and is_identity_v<Arg2>) or
      (is_identity_v<Arg1> and is_EigenDiagonal_v<Arg2>), int> = 0>
  inline auto operator*(Arg1&& arg1, Arg2&& arg2)
  {
    static_assert(MatrixTraits<Arg1>::dimension == MatrixTraits<Arg2>::dimension);
    static_assert(MatrixTraits<Arg1>::columns == MatrixTraits<Arg2>::columns);
    if constexpr(OpenKalman::is_EigenDiagonal_v<Arg1>)
    {
      return std::forward<Arg1>(arg1);
    }
    else
    {
      return std::forward<Arg2>(arg2);
    }
  }


  template<typename Arg1, typename Arg2,
    std::enable_if_t<(is_EigenDiagonal_v<Arg1> and is_zero_v<Arg2>) or
      (is_zero_v<Arg1> and is_EigenDiagonal_v<Arg2>), int> = 0>
  constexpr decltype(auto) operator*(Arg1&& arg1, Arg2&& arg2)
  {
    static_assert(MatrixTraits<Arg1>::columns == MatrixTraits<Arg2>::dimension);
    if constexpr(is_zero_v<Arg1>)
      return std::forward<Arg1>(arg1);
    else
      return std::forward<Arg2>(arg2);
  }


  ////

  template<
    typename Arg1, typename Arg2,
    std::enable_if_t<((is_EigenDiagonal_v<Arg1> and is_Eigen_matrix_v<Arg2>) or
      (is_Eigen_matrix_v<Arg1> and is_EigenDiagonal_v<Arg2>)) and
      not is_identity_v<Arg1> and not is_identity_v<Arg2> and not is_zero_v<Arg1> and not is_zero_v<Arg2>, int> = 0>
  inline auto operator*(Arg1&& arg1, Arg2&& arg2)
  {
    static_assert(MatrixTraits<Arg1>::columns == MatrixTraits<Arg2>::dimension);
    if constexpr(is_EigenDiagonal_v<Arg1>)
    {
      return strict(std::forward<Arg1>(arg1).base_matrix().asDiagonal() * std::forward<Arg2>(arg2));
    }
    else
    {
      return strict(std::forward<Arg1>(arg1) * std::forward<Arg2>(arg2).base_matrix().asDiagonal());
    }
  }


  ////

  template<typename Arg,
    std::enable_if_t<is_EigenDiagonal_v<Arg>, int> = 0>
  inline auto operator-(Arg&& arg)
  {
    return MatrixTraits<Arg>::make(-std::forward<Arg>(arg).base_matrix());
  }

}

#endif //OPENKALMAN_EIGENDIAGONAL_H
