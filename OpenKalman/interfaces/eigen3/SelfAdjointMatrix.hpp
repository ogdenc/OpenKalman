/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef OPENKALMAN_EIGEN3_SELFADJOINTMATRIX_HPP
#define OPENKALMAN_EIGEN3_SELFADJOINTMATRIX_HPP

namespace OpenKalman::Eigen3
{
  template<typename BaseMatrix, TriangleType storage_triangle>
  struct SelfAdjointMatrix
    : OpenKalman::internal::MatrixBase<SelfAdjointMatrix<BaseMatrix, storage_triangle>, BaseMatrix>
  {
    using Base = OpenKalman::internal::MatrixBase<SelfAdjointMatrix, BaseMatrix>;
    static constexpr auto uplo = storage_triangle == TriangleType::upper ? Eigen::Upper : Eigen::Lower;
    using View = Eigen::SelfAdjointView<std::remove_reference_t<BaseMatrix>, uplo>;
    using Scalar = typename MatrixTraits<BaseMatrix>::Scalar;
    static constexpr auto dimension = MatrixTraits<BaseMatrix>::dimension;
    static_assert(dimension == MatrixTraits<BaseMatrix>::columns);


    /// Default constructor
    SelfAdjointMatrix() : Base(), view(this->base_matrix()) {}


    /// Copy constructor.
    SelfAdjointMatrix(const SelfAdjointMatrix& other) : SelfAdjointMatrix(other.base_matrix()) {}


    /// Move constructor.
    SelfAdjointMatrix(SelfAdjointMatrix&& other) noexcept
      : SelfAdjointMatrix(std::move(other.base_matrix())) {}


    /// Construct from a compatible self-joint matrix object of the same storage type
#ifdef __cpp_concepts
    template<eigen_self_adjoint_expr Arg> requires
      (lower_storage_triangle<Arg> == lower_storage_triangle<SelfAdjointMatrix>)
#else
    template<typename Arg, std::enable_if_t<eigen_self_adjoint_expr<Arg> and
      lower_storage_triangle<Arg> == lower_storage_triangle <SelfAdjointMatrix>, int> = 0>
#endif
    SelfAdjointMatrix(Arg&& arg) noexcept
      : SelfAdjointMatrix(base_matrix(std::forward<Arg>(arg))) {}


    /// Construct from a compatible self-joint matrix object of the opposite storage type
#ifdef __cpp_concepts
    template<eigen_self_adjoint_expr Arg> requires
      (lower_storage_triangle<Arg> != lower_storage_triangle<SelfAdjointMatrix>)
#else
    template<typename Arg, std::enable_if_t<eigen_self_adjoint_expr < Arg> and
      lower_storage_triangle<Arg> != lower_storage_triangle <SelfAdjointMatrix>, int> = 0>
#endif
    SelfAdjointMatrix(Arg&& arg) noexcept
      : SelfAdjointMatrix(adjoint(base_matrix(std::forward<Arg>(arg)))) {}


    /// Construct from a compatible triangular matrix object
#ifdef __cpp_concepts
    template<eigen_triangular_expr Arg>
#else
    template<typename Arg, std::enable_if_t<eigen_triangular_expr < Arg>, int> = 0>
#endif
    SelfAdjointMatrix(Arg&& arg) noexcept
      : SelfAdjointMatrix(Cholesky_square(std::forward<Arg>(arg))) {}


    /// Construct from a regular or diagonal matrix object.
#ifdef __cpp_concepts
    template<typename Arg> requires eigen_matrix<Arg> or eigen_diagonal_expr<Arg>
#else
    template<typename Arg, std::enable_if_t<eigen_matrix<Arg> or eigen_diagonal_expr<Arg>, int> = 0>
#endif
    SelfAdjointMatrix(Arg&& arg) noexcept: Base(std::forward<Arg>(arg)), view(this->base_matrix()) {}


    /** Construct from a list of scalar coefficients, in row-major order.
     * \param args List of scalar values.
     */
#ifdef __cpp_concepts
    template<std::convertible_to<const Scalar> ... Args>
    requires (
      storage_triangle != TriangleType::diagonal and
        not eigen_diagonal_expr < BaseMatrix > and sizeof...(Args) == dimension * dimension) or
      (eigen_diagonal_expr < BaseMatrix > and sizeof...(Args) == dimension)
#else
    template<typename ... Args, std::enable_if_t<std::conjunction_v<std::is_convertible<Args, const Scalar>...> and
        ((not eigen_diagonal_expr < BaseMatrix > and sizeof...(Args) == dimension * dimension and
            storage_triangle != TriangleType::diagonal) or
          (eigen_diagonal_expr < BaseMatrix > and sizeof...(Args) == dimension)), int> = 0>
#endif
    SelfAdjointMatrix(Args ... args) : SelfAdjointMatrix(MatrixTraits<BaseMatrix>::make(args...)) {}


    /** Construct from a list of scalar coefficients, in row-major order.
     * \param args List of scalar values.
     */
#if defined (__cpp_concepts) && defined (__clang__) // Because of compiler issue in at least GCC version 10.1.0
    template<std::convertible_to<const Scalar> ... Args> requires (sizeof...(Args) == dimension) and
      (storage_triangle == TriangleType::diagonal) and (not eigen_diagonal_expr < BaseMatrix > )
#else
    template<typename ... Args, std::enable_if_t<
        std::conjunction_v<std::is_convertible<Args, const Scalar>...> and not eigen_diagonal_expr < BaseMatrix>and
    sizeof...(Args) == dimension and storage_triangle == TriangleType::diagonal, int> = 0>
#endif
    SelfAdjointMatrix(Args ... args)
      : SelfAdjointMatrix(make_native_matrix(DiagonalMatrix {static_cast<const Scalar>(args)...})) {}


    /** Copy assignment operator
     * \param other Another SelfAdjointMatrix
     * \return Reference to this.
     */
    auto& operator=(const SelfAdjointMatrix& other)
    {
      if constexpr (not zero_matrix<BaseMatrix> and not identity_matrix<BaseMatrix>)
        if (this != &other)
        {
          this->base_matrix().template triangularView<uplo>() = other.base_matrix();
        }
      return *this;
    }


    /** Move assignment operator
     * \param other A SelfAdjointMatrix temporary value.
     * \return Reference to this.
     */
    auto& operator=(SelfAdjointMatrix&& other) noexcept
    {
      if constexpr (not zero_matrix<BaseMatrix> and not identity_matrix<BaseMatrix>)
        if (this != &other)
        {
          this->base_matrix() = std::move(other).base_matrix();
        }
      return *this;
    }


    /// Assign from another self-adjoint special matrix
#ifdef __cpp_concepts
    template<eigen_self_adjoint_expr Arg>
#else
    template<typename Arg, std::enable_if_t<eigen_self_adjoint_expr<Arg>, int> = 0>
#endif
    auto& operator=(Arg&& arg)
    {
      if constexpr (zero_matrix<BaseMatrix>)
      {
        static_assert(zero_matrix<Arg>);
      }
      else if constexpr (identity_matrix<BaseMatrix>)
      {
        static_assert(identity_matrix<Arg>);
      }
      else if constexpr(std::is_lvalue_reference_v<Arg>)
      {
        if constexpr(upper_storage_triangle<Arg>
          == upper_storage_triangle<SelfAdjointMatrix>)
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
#ifdef __cpp_concepts
    template<eigen_triangular_expr Arg>
#else
    template<typename Arg, std::enable_if_t<eigen_triangular_expr<Arg>, int> = 0>
#endif
    auto& operator=(Arg&& arg)
    {
      if constexpr (zero_matrix<BaseMatrix>)
      {
        static_assert(zero_matrix<Arg>);
      }
      else if constexpr (identity_matrix<BaseMatrix>)
      {
        static_assert(identity_matrix<Arg>);
      }
      else
      {
        return operator=(Cholesky_square(std::forward<Arg>(arg)));
      }
    }


    /// Assign from a regular Eigen matrix. (Uses only the storage triangular part.)
#ifdef __cpp_concepts
    template<eigen_matrix Arg>
#else
    template<typename Arg, std::enable_if_t<eigen_matrix<Arg>, int> = 0>
#endif
    auto& operator=(Arg&& arg)
    {
      if constexpr (zero_matrix<BaseMatrix>)
      {
        static_assert(zero_matrix<Arg>);
      }
      else if constexpr (identity_matrix<BaseMatrix>)
      {
        static_assert(identity_matrix<Arg>);
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
      if constexpr (zero_matrix<BaseMatrix>)
      {
        static_assert(zero_matrix<Arg>);
      }
      else if constexpr (identity_matrix<BaseMatrix>)
      {
        static_assert(identity_matrix<Arg>);
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
      if constexpr (zero_matrix<BaseMatrix>)
      {
        static_assert(zero_matrix<Arg>);
      }
      else if constexpr (identity_matrix<BaseMatrix>)
      {
        static_assert(identity_matrix<Arg>);
      }
      else
      {
        this->base_matrix() = std::move(arg.nestedExpression());
      }
      return *this;
    }

    template<typename Arg, TriangleType t>
    auto& operator+=(const SelfAdjointMatrix<Arg, t>& arg)
    {
      static_assert(MatrixTraits<Arg>::dimension == dimension);
      if constexpr(t == storage_triangle)
        this->base_matrix().template triangularView<uplo>() += arg.base_matrix();
      else
        this->base_matrix().template triangularView<uplo>() += arg.base_matrix().adjoint();
      return *this;
    }

    template<typename Arg, TriangleType t>
    auto& operator-=(const SelfAdjointMatrix<Arg, t>& arg)
    {
      static_assert(MatrixTraits<Arg>::dimension == dimension);
      if constexpr(t == storage_triangle)
        this->base_matrix().template triangularView<uplo>() -= arg.base_matrix();
      else
        this->base_matrix().template triangularView<uplo>() -= arg.base_matrix().adjoint();
      return *this;
    }

#ifdef __cpp_concepts
    template<std::convertible_to<Scalar> S>
#else
    template<typename S, std::enable_if_t<std::is_convertible_v<S, Scalar>, int> = 0>
#endif
    auto& operator*=(const S s)
    {
      this->base_matrix().template triangularView<uplo>() *= s;
      return *this;
    }

#ifdef __cpp_concepts
    template<std::convertible_to<Scalar> S>
#else
    template<typename S, std::enable_if_t<std::is_convertible_v<S, Scalar>, int> = 0>
#endif
    auto& operator/=(const S s)
    {
      this->base_matrix().template triangularView<uplo>() /= s;
      return *this;
    }

    constexpr auto& base_view()& { return view; }

    constexpr const auto& base_view() const& { return view; }

    constexpr auto&& base_view()&& { return std::move(view); }

    constexpr const auto&& base_view() const&& { return std::move(view); }


    auto operator()(std::size_t i, std::size_t j)
    {
      if constexpr (element_settable < SelfAdjointMatrix, 2 >)
        return OpenKalman::internal::ElementSetter(*this, i, j);
      else
        return const_cast<const SelfAdjointMatrix&>(*this)(i, j);
    }

    auto operator()(std::size_t i, std::size_t j) const noexcept
    {
      return OpenKalman::internal::ElementSetter(*this,
        i,
        j);
    }

    auto operator[](std::size_t i)
    {
      if constexpr(element_settable < SelfAdjointMatrix, 1 >)
        return OpenKalman::internal::ElementSetter(*this, i);
      else if constexpr(element_settable < SelfAdjointMatrix, 2 >)
        return OpenKalman::internal::ElementSetter(*this, i, i);
      else
        return const_cast<const SelfAdjointMatrix&>(*this)[i];
    }

    auto operator[](std::size_t i) const
    {
      if constexpr(element_gettable < SelfAdjointMatrix, 1 >)
        return OpenKalman::internal::ElementSetter(*this, i);
      else
        return OpenKalman::internal::ElementSetter(*this, i, i);
    }

    auto operator()(std::size_t i) { return operator[](i); }

    auto operator()(std::size_t i) const { return operator[](i); }


#ifdef __cpp_concepts

    template<eigen_matrix B>
#else

    template<typename B, std::enable_if_t<eigen_matrix<B>, int> = 0>
#endif
    auto solve(const B& b) const
    {
      using M = Eigen::Matrix<Scalar, dimension, MatrixTraits<B>::columns>;
      auto llt = this->base_view().llt();
      M ret = llt.solve(b);
      if (llt.info() != Eigen::Success) [[unlikely]]
      {
        // A is semidefinite. Use LDLT decomposition instead.
        auto ldlt = this->base_view().ldlt();
        M ret2 = ldlt.solve(b);
        if ((not ldlt.isPositive() and not ldlt.isNegative()) or ldlt.info() != Eigen::Success)
        {
          throw (std::runtime_error("SelfAdjointMatrix solve: A is indefinite"));
        }
        return ret2;
      }
      return ret;
    }

    static auto zero() { return MatrixTraits<BaseMatrix>::zero(); }

    static auto identity()
    {
      auto b = MatrixTraits<BaseMatrix>::identity();
      return SelfAdjointMatrix<std::decay_t<decltype(b)>, TriangleType::diagonal>(std::move(b));
    }

  private:
    View view;
  };

  /////////////////////////////////////
  //        Deduction Guides         //
  /////////////////////////////////////

#ifdef __cpp_concepts
  template<typename Arg> requires eigen_matrix<Arg> or eigen_diagonal_expr<Arg>
#else
  template<typename Arg, std::enable_if_t<eigen_matrix<Arg> or eigen_diagonal_expr<Arg>, int> = 0>
#endif
  SelfAdjointMatrix(Arg&&) -> SelfAdjointMatrix<passable_t<Arg>, TriangleType::lower>;


#ifdef __cpp_concepts
  template<eigen_self_adjoint_expr M>
#else
  template<typename M, std::enable_if_t<eigen_self_adjoint_expr<M>, int> = 0>
#endif
  SelfAdjointMatrix(M&&) -> SelfAdjointMatrix<
    passable_t<typename MatrixTraits<M>::BaseMatrix>, MatrixTraits<M>::storage_type>;


#ifdef __cpp_concepts
  template<eigen_triangular_expr M>
#else
  template<typename M, std::enable_if_t<eigen_triangular_expr<M>, int> = 0>
#endif
  SelfAdjointMatrix(M&&) -> SelfAdjointMatrix<
    native_matrix_t<typename MatrixTraits<M>::BaseMatrix>, MatrixTraits<M>::triangle_type>;


  template<typename Arg, unsigned int UpLo>
  SelfAdjointMatrix(Eigen::SelfAdjointView<Arg, UpLo>&&)
  -> SelfAdjointMatrix<Arg, UpLo & Eigen::Upper ? TriangleType::upper : TriangleType::lower>;


  /// If the arguments are a sequence of scalars, deduce a square, self-adjoint matrix.
#ifdef __cpp_concepts
  template<typename ... Args>
  requires(std::is_arithmetic_v<Args>and ...)
#else
  template<typename ... Args, std::enable_if_t<std::conjunction_v<std::is_arithmetic<Args>...>, int> = 0>
#endif
  SelfAdjointMatrix(Args ...)
    ->SelfAdjointMatrix<
      Eigen::Matrix<std::decay_t<std::common_type_t<Args...>>, OpenKalman::internal::constexpr_sqrt(sizeof...(Args)),
        OpenKalman::internal::constexpr_sqrt(sizeof...(Args))>, TriangleType::lower>;


  ///////////////////////////////////
  //        Make functions         //
  ///////////////////////////////////

#ifdef __cpp_concepts
  template<TriangleType t = TriangleType::lower, typename M>
  requires eigen_matrix<M> or eigen_diagonal_expr<M>
#else
  template<
    TriangleType t = TriangleType::lower, typename M,
    std::enable_if_t<eigen_matrix<M> or eigen_diagonal_expr<M>, int> = 0>
#endif
  auto
  make_EigenSelfAdjointMatrix(M&& m)
  {
    return SelfAdjointMatrix<passable_t<M>, t> {std::forward<M>(m)};
  }


#ifdef __cpp_concepts
  template<TriangleType t, eigen_self_adjoint_expr M>
#else
  template<TriangleType t, typename M, std::enable_if_t<eigen_self_adjoint_expr<M>, int> = 0>
#endif
  auto
  make_EigenSelfAdjointMatrix(M&& m)
  {
    if constexpr(t == MatrixTraits<M>::storage_type)
      return make_EigenSelfAdjointMatrix<t>(base_matrix(std::forward<M>(m)));
    else
      return make_EigenSelfAdjointMatrix<t>(adjoint(base_matrix(std::forward<M>(m))));
  }


#ifdef __cpp_concepts
  template<eigen_self_adjoint_expr M>
#else
  template<typename M, std::enable_if_t<eigen_self_adjoint_expr<M>, int> = 0>
#endif
  auto
  make_EigenSelfAdjointMatrix(M&& m)
  {
    return make_EigenSelfAdjointMatrix<MatrixTraits<M>::storage_type>(std::forward<M>(m));
  }

} // OpenKalman::Eigen3

namespace OpenKalman
{
  ///////////////////////////
  //        Traits         //
  ///////////////////////////

  template<typename ArgType, TriangleType storage_triangle>
  struct MatrixTraits<Eigen3::SelfAdjointMatrix<ArgType, storage_triangle>>
  {
    static constexpr TriangleType storage_type = storage_triangle;
    using BaseMatrix = ArgType;
    using Scalar = typename MatrixTraits<BaseMatrix>::Scalar;
    static constexpr auto dimension = MatrixTraits<BaseMatrix>::dimension;
    static constexpr auto columns = MatrixTraits<BaseMatrix>::columns;
    static_assert(dimension == columns, "A self-adjoint matrix must be square.");

    template<typename Derived>
    using MatrixBaseType = Eigen3::internal::Eigen3MatrixBase<
      Derived, Eigen3::SelfAdjointMatrix<std::decay_t<BaseMatrix>, storage_type>>;

    template<typename Derived>
    using CovarianceBaseType = Eigen3::internal::Eigen3CovarianceBase<
      Derived, Eigen3::SelfAdjointMatrix<std::decay_t<BaseMatrix>, storage_type>>;

    template<std::size_t rows = dimension, std::size_t cols = dimension, typename S = Scalar>
    using NativeMatrix = typename MatrixTraits<BaseMatrix>::template NativeMatrix<rows, cols, S>;

    using SelfContained = Eigen3::SelfAdjointMatrix<self_contained_t<BaseMatrix>, storage_type>;

    template<TriangleType t = storage_type, std::size_t dim = dimension, typename S = Scalar>
    using SelfAdjointBaseType = Eigen3::SelfAdjointMatrix<NativeMatrix<dim, dim, S>, t>;

    template<TriangleType t = storage_type, std::size_t dim = dimension, typename S = Scalar>
    using TriangularBaseType = Eigen3::TriangularMatrix<NativeMatrix<dim, dim, S>, t>;

    template<std::size_t dim = dimension, typename S = Scalar>
    using DiagonalBaseType = Eigen3::DiagonalMatrix<NativeMatrix<dim, 1, S>>;

#ifdef __cpp_concepts
    template<TriangleType t = storage_type, typename Arg>
    requires (not std::convertible_to<Arg, const Scalar>)
#else
    template<TriangleType t = storage_type, typename Arg,
      std::enable_if_t<not std::is_convertible_v<Arg, const Scalar>, int> = 0>
#endif
    static auto make(Arg&& arg) noexcept
    {
      return Eigen3::SelfAdjointMatrix<std::decay_t<Arg>, t>(std::forward<Arg>(arg));
    }

    /// Make self-adjoint matrix using a list of coefficients in row-major order.
    /// Only the coefficients in the lower-left corner are significant.
#ifdef __cpp_concepts
    template<TriangleType t = storage_type, std::convertible_to<const Scalar> ... Args>
#else
    template<TriangleType t = storage_type, typename ... Args,
      std::enable_if_t<std::conjunction_v<std::is_convertible<Args, const Scalar>...>, int> = 0>
#endif
    static auto make(Args ... args)
    {
      static_assert(sizeof...(Args) == dimension * dimension);
      return make<t>(MatrixTraits<BaseMatrix>::make(args...));
    }

    static auto zero() { return Eigen3::SelfAdjointMatrix<BaseMatrix, storage_type>::zero(); }

    static auto identity() { return Eigen3::SelfAdjointMatrix<BaseMatrix, storage_type>::identity(); }

  };

} // namespace OpenKalman

namespace OpenKalman::Eigen3
{
  //////////////////////////////
  //        Overloads         //
  //////////////////////////////

#ifdef __cpp_concepts
  template<Eigen3::eigen_self_adjoint_expr Arg>
#else
  template<typename Arg, std::enable_if_t<Eigen3::eigen_self_adjoint_expr<Arg>, int> = 0>
#endif
  inline auto
  transpose(Arg && arg)
  {
    constexpr auto t = Eigen3::lower_storage_triangle<Arg> ? TriangleType::upper : TriangleType::lower;
    auto b = transpose(base_matrix(std::forward<Arg>(arg)));
    return Eigen3::SelfAdjointMatrix<decltype(b), t>(std::move(b));
  }


#ifdef __cpp_concepts
  template<Eigen3::eigen_self_adjoint_expr Arg>
#else
  template<typename Arg, std::enable_if_t<Eigen3::eigen_self_adjoint_expr<Arg>, int> = 0>
#endif
  inline auto
  adjoint(Arg && arg)
  {
    constexpr auto t = Eigen3::lower_storage_triangle<Arg> ? TriangleType::upper : TriangleType::lower;
    auto b = adjoint(base_matrix(std::forward<Arg>(arg)));
    return Eigen3::SelfAdjointMatrix<decltype(b), t>(std::move(b));
  }


#ifdef __cpp_concepts
  template<Eigen3::eigen_self_adjoint_expr Arg, typename U> requires
    (Eigen3::eigen_matrix<U> or Eigen3::eigen_triangular_expr<U> or Eigen3::eigen_self_adjoint_expr<U> or
    Eigen3::eigen_diagonal_expr<U> or Eigen3::euclidean_expr<U>) and
    (not std::is_const_v<std::remove_reference_t<Arg>>)
#else
  template<typename Arg, typename U, std::enable_if_t<
    Eigen3::eigen_self_adjoint_expr<Arg> and
    (Eigen3::eigen_matrix<U> or Eigen3::eigen_triangular_expr<U> or Eigen3::eigen_self_adjoint_expr<U> or
    Eigen3::eigen_diagonal_expr<U> or Eigen3::euclidean_expr<U>) and
    (not std::is_const_v<std::remove_reference_t<Arg>>), int> = 0>
#endif
  inline Arg&
  rank_update(Arg & arg, const U& u, const typename MatrixTraits<Arg>::Scalar alpha = 1)
  {
    static_assert(MatrixTraits<U>::dimension == MatrixTraits<Arg>::dimension);
    arg.base_view().rankUpdate(u, alpha);
    return arg;
  }


#ifdef __cpp_concepts
  template<Eigen3::eigen_self_adjoint_expr Arg, typename U> requires
    (Eigen3::eigen_matrix<U> or Eigen3::eigen_triangular_expr<U> or Eigen3::eigen_self_adjoint_expr<U> or
    Eigen3::eigen_diagonal_expr<U> or Eigen3::euclidean_expr<U>)
#else
  template<typename Arg, typename U, std::enable_if_t<
    Eigen3::eigen_self_adjoint_expr<Arg> and
    (Eigen3::eigen_matrix<U> or Eigen3::eigen_triangular_expr<U> or Eigen3::eigen_self_adjoint_expr<U> or
    Eigen3::eigen_diagonal_expr<U> or Eigen3::euclidean_expr<U>), int> = 0>
#endif
  inline auto
  rank_update(Arg && arg, const U& u, const typename MatrixTraits<Arg>::Scalar alpha = 1)
  {
    static_assert(MatrixTraits<U>::dimension == MatrixTraits<Arg>::dimension);
    auto sa = std::forward<Arg>(arg);
    rank_update(sa, u, alpha);
    return sa;
  }


  // Perform an LQ decomposition of matrix A=[L,0]Q, where L is a lower-triangular matrix, and Q is orthogonal.
  // Returns L as a lower-triangular matrix.
#ifdef __cpp_concepts
  template<Eigen3::eigen_self_adjoint_expr A>
#else
  template<typename A, std::enable_if_t<Eigen3::eigen_self_adjoint_expr<A>, int> = 0>
#endif
  inline auto
  LQ_decomposition(A && a)
  {
    return LQ_decomposition(make_native_matrix(std::forward<A>(a)));
  }


  // Perform a QR decomposition of matrix A=Q[U,0], where U is an upper-triangular matrix, and Q is orthogonal.
  // Returns L as an upper-triangular matrix.
#ifdef __cpp_concepts
  template<Eigen3::eigen_self_adjoint_expr A>
#else
  template<typename A, std::enable_if_t<Eigen3::eigen_self_adjoint_expr<A>, int> = 0>
#endif
  inline auto
  QR_decomposition(A && a)
  {
    return QR_decomposition(make_native_matrix(std::forward<A>(a)));
  }


} // namespace OpenKalman::Eigen3


#endif //OPENKALMAN_EIGEN3_SELFADJOINTMATRIX_HPP
