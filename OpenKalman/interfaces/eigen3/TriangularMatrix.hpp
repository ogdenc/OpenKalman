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
  template<typename NestedMatrix, TriangleType triangle_type>
  struct TriangularMatrix
    : OpenKalman::internal::MatrixBase<TriangularMatrix<NestedMatrix, triangle_type>, NestedMatrix>
  {
    using Base = OpenKalman::internal::MatrixBase<TriangularMatrix, NestedMatrix>;
    static constexpr auto uplo = triangle_type == TriangleType::upper ? Eigen::Upper : Eigen::Lower;
    using View = Eigen::TriangularView<std::remove_reference_t<NestedMatrix>, uplo>;
    using Scalar = typename MatrixTraits<NestedMatrix>::Scalar;
    static constexpr auto dimension = MatrixTraits<NestedMatrix>::dimension;
    static_assert(dimension == MatrixTraits<NestedMatrix>::columns);

    /// Default constructor
    TriangularMatrix() : Base(), view(this->nested_matrix()) {}

    /// Copy constructor
    TriangularMatrix(const TriangularMatrix& other) : TriangularMatrix(other.nested_matrix()) {}

    /// Move constructor
    TriangularMatrix(TriangularMatrix&& other) noexcept
      : TriangularMatrix(std::move(other).nested_matrix()) {}

    /// Construct from a compatible triangular matrix object of the same TriangleType.
#ifdef __cpp_concepts
    template<eigen_triangular_expr Arg>
#else
    template<typename Arg, std::enable_if_t<eigen_triangular_expr<Arg>, int> = 0>
#endif
    TriangularMatrix(Arg&& arg) noexcept: TriangularMatrix(nested_matrix(std::forward<Arg>(arg)))
    {
      static_assert(lower_triangular_matrix<Arg> == lower_triangular_matrix<TriangularMatrix>);
    }

    /// Construct from a compatible self-adjoint matrix object
#ifdef __cpp_concepts
    template<eigen_self_adjoint_expr Arg>
#else
    template<typename Arg, std::enable_if_t<eigen_self_adjoint_expr<Arg>, int> = 0>
#endif
    TriangularMatrix(Arg&& arg) noexcept
      : TriangularMatrix(Cholesky_factor<triangle_type>(std::forward<Arg>(arg))) {}

    /// Construct from a reference to a regular or diagonal matrix object
#ifdef __cpp_concepts
    template<typename Arg> requires eigen_matrix<Arg> or eigen_diagonal_expr<Arg>
#else
    template<typename Arg, std::enable_if_t<eigen_matrix<Arg> or eigen_diagonal_expr<Arg>, int> = 0>
#endif
    TriangularMatrix(Arg&& arg) noexcept: Base(std::forward<Arg>(arg)), view(this->nested_matrix()) {}

    /// Construct from a list of scalar coefficients, in row-major order.
#ifdef __cpp_concepts
    template<std::convertible_to<const Scalar> ... Args> requires
      (triangle_type != TriangleType::diagonal and (not eigen_diagonal_expr<NestedMatrix> and
      sizeof...(Args) == dimension * dimension)) or
      (eigen_diagonal_expr<NestedMatrix> and sizeof...(Args) == dimension)
#else
    template<
      typename ... Args, std::enable_if_t<
        std::conjunction_v<std::is_convertible<Args, const Scalar>...> and
          ((not eigen_diagonal_expr < NestedMatrix > and sizeof...(Args) == dimension * dimension and
              triangle_type != TriangleType::diagonal) or
            (eigen_diagonal_expr < NestedMatrix > and sizeof...(Args) == dimension)), int> = 0>
#endif
    TriangularMatrix(Args ... args) : TriangularMatrix(MatrixTraits<NestedMatrix>::make(args...)) {}

    /// Construct from a list of scalar coefficients, in row-major order.
#if defined (__cpp_concepts) && defined (__clang__) // Because of compiler issue in at least GCC version 10.1.0
    template<std::convertible_to<const Scalar> ... Args> requires (triangle_type == TriangleType::diagonal) and
    (not eigen_diagonal_expr<NestedMatrix>) and (sizeof...(Args) == dimension)
#else
    template<
      typename ... Args, std::enable_if_t<
        std::conjunction_v<std::is_convertible<Args, const Scalar>...> and not eigen_diagonal_expr < NestedMatrix>and
    sizeof...(Args) == dimension and triangle_type == TriangleType::diagonal, int> = 0>
#endif
    TriangularMatrix(Args ... args)
      : TriangularMatrix(make_native_matrix(DiagonalMatrix {static_cast<const Scalar>(args)...})) {}

    /// Copy assignment operator
    auto& operator=(const TriangularMatrix& other)
    {
      if constexpr (not zero_matrix < NestedMatrix > and not identity_matrix<NestedMatrix>)
        if (this != &other)
        {
          this->nested_matrix().template triangularView<uplo>() = other.nested_matrix();
        }
      return *this;
    }

    /// Move assignment operator
    auto& operator=(TriangularMatrix&& other) noexcept
    {
      if constexpr (not zero_matrix < NestedMatrix > and not identity_matrix<NestedMatrix>)
        if (this != &other)
        {
          this->nested_matrix() = std::move(other).nested_matrix();
        }
      return *this;
    }


    /// Assign from another triangular matrix (must be the same triangle)
#ifdef __cpp_concepts
    template<eigen_triangular_expr Arg>
#else
    template<typename Arg, std::enable_if_t<eigen_triangular_expr<Arg>, int> = 0>
#endif

    auto& operator=(Arg&& arg)
    {
      static_assert(OpenKalman::internal::same_triangle_type_as<Arg, TriangularMatrix>);
      if constexpr (zero_matrix<NestedMatrix>)
      {
        static_assert(zero_matrix<Arg>);
      }
      else if constexpr (identity_matrix<NestedMatrix>)
      {
        static_assert(identity_matrix<Arg>);
      }
      else if constexpr(std::is_lvalue_reference_v<Arg>)
      {
        this->nested_matrix().template triangularView<uplo>() = nested_matrix(arg);
      }
      else
      {
        this->nested_matrix() = std::forward<Arg>(arg);
      }
      return *this;
    }

    /// Assign by converting from a self-adjoint special matrix
#ifdef __cpp_concepts
    template<eigen_self_adjoint_expr Arg>
#else
    template<typename Arg, std::enable_if_t<eigen_self_adjoint_expr<Arg>, int> = 0>
#endif

    auto& operator=(Arg&& arg)
    {
      if constexpr (zero_matrix < NestedMatrix >)
      {
        static_assert(zero_matrix < Arg > );
      }
      else if constexpr (identity_matrix<NestedMatrix>)
      {
        static_assert(identity_matrix<Arg>);
      }
      else
      {
        this->nested_matrix().template triangularView<uplo>() =
          nested_matrix(Cholesky_factor<triangle_type>(std::forward<Arg>(arg)));
      }
      return *this;
    }

    /// Assign from an Eigen::MatrixBase derived object. (Only uses the triangular part.)
#ifdef __cpp_concepts
    template<eigen_matrix Arg>
#else

    template<typename Arg, std::enable_if_t<eigen_matrix<Arg>, int> = 0>
#endif
    auto& operator=(Arg&& arg)
    {
      if constexpr (zero_matrix<NestedMatrix>)
      {
        static_assert(zero_matrix<Arg>);
      }
      else if constexpr (identity_matrix<NestedMatrix>)
      {
        static_assert(identity_matrix<Arg>);
      }
      else if constexpr(std::is_lvalue_reference_v<Arg>)
      {
        this->nested_matrix().template triangularView<uplo>() = arg;
      }
      else
      {
        this->nested_matrix() = std::forward<Arg>(arg);
      }
      return *this;
    }

    /// Assign by copying from an Eigen::TriangularBase derived object.
    template<typename Arg, unsigned int UpLo>
    auto& operator=(const Eigen::TriangularView<Arg, UpLo>& arg)
    {
      if constexpr (zero_matrix<NestedMatrix>)
      {
        static_assert(zero_matrix<Arg>);
      }
      else if constexpr (identity_matrix<NestedMatrix>)
      {
        static_assert(identity_matrix<Arg>);
      }
      else if constexpr(((UpLo & Eigen::Lower) != 0) == (triangle_type == TriangleType::lower))
      {
        this->nested_matrix().template triangularView<UpLo>() = arg.nestedExpression();
      }
      else
      {
        this->nested_matrix().template triangularView<UpLo>() = arg.nestedExpression().adjoint();
      }
      return *this;
    }

    /// Assign by moving from an Eigen::TriangularBase derived object.
    template<typename Arg>
    auto& operator=(Eigen::TriangularView<Arg, uplo>&& arg) noexcept
    {
      if constexpr (zero_matrix<NestedMatrix>)
      {
        static_assert(zero_matrix<Arg>);
      }
      else if constexpr (identity_matrix<NestedMatrix>)
      {
        static_assert(identity_matrix<Arg>);
      }
      else
      {
        this->nested_matrix() = std::move(arg.nestedExpression());
      }
      return *this;
    }

    template<typename Arg>
    auto& operator+=(const TriangularMatrix<Arg, triangle_type>& arg)
    {
      static_assert(MatrixTraits<Arg>::dimension == dimension);
      this->nested_view() += arg.nested_matrix();
      return *this;
    }

    template<typename Arg>
    auto& operator-=(const TriangularMatrix<Arg, triangle_type>& arg)
    {
      static_assert(MatrixTraits<Arg>::dimension == dimension);
      this->nested_view() -= arg.nested_matrix();
      return *this;
    }

#ifdef __cpp_concepts
    template<std::convertible_to<Scalar> S>
#else
    template<typename S, std::enable_if_t<std::is_convertible_v<S, Scalar>, int> = 0>
#endif
    auto& operator*=(const S s)
    {
      this->nested_view() *= s;
      return *this;
    }

#ifdef __cpp_concepts
    template<std::convertible_to<Scalar> S>
#else
    template<typename S, std::enable_if_t<std::is_convertible_v<S, Scalar>, int> = 0>
#endif
    auto& operator/=(const S s)
    {
      this->nested_view() /= s;
      return *this;
    }

    template<typename Arg>
    auto& operator*=(const TriangularMatrix<Arg, triangle_type>& arg)
    {
      static_assert(MatrixTraits<Arg>::dimension == dimension);
      this->nested_view() = this->nested_view() * make_native_matrix(arg);
      return *this;
    }

    constexpr auto& nested_view()& { return view; }

    constexpr const auto& nested_view() const& { return view; }

    constexpr auto&& nested_view()&& { return std::move(view); }

    constexpr const auto&& nested_view() const&& { return std::move(view); }


    auto operator()(std::size_t i, std::size_t j)
    {
      if constexpr (element_settable < TriangularMatrix, 2 >)
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
      if constexpr(element_gettable < TriangularMatrix, 1 >)
        return OpenKalman::internal::ElementSetter(*this, i);
      else if constexpr(element_gettable < TriangularMatrix, 2 >)
        return OpenKalman::internal::ElementSetter(*this, i, i);
      else
        return const_cast<const TriangularMatrix&>(*this)[i];
    }

    auto operator[](std::size_t i) const
    {
      if constexpr(element_gettable < TriangularMatrix, 1 >)
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
      return this->nested_view().solve(b);
    }

    static auto zero() { return MatrixTraits<NestedMatrix>::zero(); }

    static auto identity()
    {
      auto b = MatrixTraits<NestedMatrix>::identity();
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
  template<typename M, std::enable_if_t<eigen_matrix<M> or eigen_diagonal_expr<M>, int> = 0>
#endif
  TriangularMatrix(M&&) -> TriangularMatrix<passable_t<M>, TriangleType::lower>;


#ifdef __cpp_concepts
  template<eigen_triangular_expr M>
#else
  template<typename M, std::enable_if_t<eigen_triangular_expr<M>, int> = 0>
#endif
  TriangularMatrix(M&&) -> TriangularMatrix<
    passable_t<nested_matrix_t<M>>, MatrixTraits<M>::triangle_type>;


#ifdef __cpp_concepts
  template<eigen_self_adjoint_expr M>
#else
  template<typename M, std::enable_if_t<eigen_self_adjoint_expr<M>, int> = 0>
#endif
  TriangularMatrix(M&&) -> TriangularMatrix<
    native_matrix_t<nested_matrix_t<M>>, MatrixTraits<M>::storage_type>;


  template<typename Arg, unsigned int UpLo>
  TriangularMatrix(const Eigen::TriangularView<Arg, UpLo>&)
  -> TriangularMatrix<Arg, UpLo & Eigen::Upper ? TriangleType::upper : TriangleType::lower>;


  /// If the arguments are a sequence of scalars, deduce a square, lower triangular matrix.
#ifdef __cpp_concepts
  template<typename ... Args> requires ((sizeof...(Args) >= 1) and ... and std::is_arithmetic_v<Args>)
#else
  template<typename ... Args, std::enable_if_t<
    sizeof...(Args) >= 1 and std::conjunction_v<std::is_arithmetic<Args>...>, int> = 0>
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
    std::enable_if_t<eigen_matrix<M> or eigen_diagonal_expr<M>, int> = 0>
#endif
  auto make_EigenTriangularMatrix(M&& m)
  {
    return TriangularMatrix<passable_t<M>, t > (std::forward<M>(m));
  }

#ifdef __cpp_concepts
  template<TriangleType t, eigen_triangular_expr M> requires (t == MatrixTraits<M>::triangle_type)
#else
  template<TriangleType t, typename M, std::enable_if_t<eigen_triangular_expr<M> and
    (t == MatrixTraits<M>::triangle_type), int> = 0>
#endif
  auto make_EigenTriangularMatrix(M&& m)
  {
    return TriangularMatrix<passable_t<M>, t > (std::forward<M>(m));
  }

#ifdef __cpp_concepts
  template<eigen_triangular_expr M>
#else
  template<typename M, std::enable_if_t<eigen_triangular_expr<M>, int> = 0>
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
    using NestedMatrix = ArgType;
    using Scalar = typename MatrixTraits<NestedMatrix>::Scalar;
    static constexpr auto dimension = MatrixTraits<NestedMatrix>::dimension;
    static constexpr auto columns = MatrixTraits<NestedMatrix>::columns;
    static_assert(dimension == columns, "A triangular matrix must be square.");

    template<typename Derived>
    using MatrixBaseType =
    Eigen3::internal::Eigen3MatrixBase<Derived, Eigen3::TriangularMatrix<std::decay_t<NestedMatrix>, triangle_type>>;

    template<typename Derived>
    using CovarianceBaseType = Eigen3::internal::Eigen3CovarianceBase<
      Derived, Eigen3::TriangularMatrix<std::decay_t<NestedMatrix>, triangle_type>>;

    template<std::size_t rows = dimension, std::size_t cols = dimension, typename S = Scalar>
    using NativeMatrix = typename MatrixTraits<NestedMatrix>::template NativeMatrix<rows, cols, S>;

    using SelfContained = Eigen3::TriangularMatrix<self_contained_t<NestedMatrix>, triangle_type>;

    template<TriangleType t = triangle_type, std::size_t dim = dimension, typename S = Scalar>
    using SelfAdjointBaseType = Eigen3::SelfAdjointMatrix<NativeMatrix<dim, dim, S>, t>;

    template<TriangleType t = triangle_type, std::size_t dim = dimension, typename S = Scalar>
    using TriangularBaseType = Eigen3::TriangularMatrix<NativeMatrix<dim, dim, S>, t>;

    template<std::size_t dim = dimension, typename S = Scalar>
    using DiagonalBaseType = Eigen3::DiagonalMatrix<NativeMatrix<dim, 1, S>>;

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
      return make<t>(MatrixTraits<NestedMatrix>::make(args...));
    }

    static auto zero() { return Eigen3::TriangularMatrix<NestedMatrix, triangle_type>::zero(); }

    static auto identity() { return Eigen3::TriangularMatrix<NestedMatrix, triangle_type>::identity(); }

  };

} // namespace OpenKalman


namespace OpenKalman::Eigen3
{
  // ------------------------ //
  //        Overloads         //
  // ------------------------ //

#ifdef __cpp_concepts
  template<Eigen3::eigen_triangular_expr Arg>
#else
  template<typename Arg, std::enable_if_t<Eigen3::eigen_triangular_expr<Arg>, int> = 0>
#endif
  constexpr auto
  transpose(Arg&& arg)
  {
    constexpr auto t = triangle_type_of<Arg> == TriangleType::lower ? TriangleType::upper :
                       triangle_type_of<Arg> == TriangleType::upper ? TriangleType::lower : TriangleType::diagonal;
    auto b = nested_matrix(std::forward<Arg>(arg)).transpose();
    return Eigen3::TriangularMatrix<decltype(b), t>(std::move(b));
  }


#ifdef __cpp_concepts
  template<Eigen3::eigen_triangular_expr Arg>
#else
  template<typename Arg, std::enable_if_t<Eigen3::eigen_triangular_expr<Arg>, int> = 0>
#endif
  constexpr auto
  adjoint(Arg&& arg)
  {
    constexpr auto t = triangle_type_of<Arg> == TriangleType::lower ? TriangleType::upper :
                       triangle_type_of<Arg> == TriangleType::upper ? TriangleType::lower : TriangleType::diagonal;
    auto b = nested_matrix(std::forward<Arg>(arg)).adjoint();
    return Eigen3::TriangularMatrix<decltype(b), t>(std::move(b));
  }


#ifdef __cpp_concepts
  template<Eigen3::eigen_triangular_expr Arg, typename U> requires
    (Eigen3::eigen_matrix<U> or Eigen3::eigen_triangular_expr<U> or Eigen3::eigen_self_adjoint_expr<U> or
      Eigen3::eigen_diagonal_expr<U> or Eigen3::euclidean_expr<U>) and
    (not std::is_const_v<std::remove_reference_t<Arg>>)
#else
  template<typename Arg, typename U, std::enable_if_t<Eigen3::eigen_triangular_expr<Arg> and
    (Eigen3::eigen_matrix<U> or Eigen3::eigen_triangular_expr<U> or Eigen3::eigen_self_adjoint_expr<U> or
      Eigen3::eigen_diagonal_expr<U> or Eigen3::euclidean_expr<U>) and
    (not std::is_const_v<std::remove_reference_t<Arg>>), int> = 0>
#endif
  inline Arg&
  rank_update(Arg& arg, const U& u, const typename MatrixTraits<Arg>::Scalar alpha = 1)
  {
    static_assert(MatrixTraits<U>::dimension == MatrixTraits<Arg>::dimension);
    using Scalar = typename MatrixTraits<Arg>::Scalar;
    constexpr auto t = lower_triangular_matrix<Arg> ? Eigen::Lower : Eigen::Upper;
    for (Eigen::Index i = 0; i < Eigen::Index(MatrixTraits<U>::columns); ++i)
    {
      if (Eigen::internal::llt_inplace<Scalar, t>::rankUpdate(arg.nested_matrix(), u.col(i), alpha) >= 0)
      {
        throw (std::runtime_error("TriangularMatrix rank_update: product is not positive definite"));
      }
    }
    return arg;
  }


#ifdef __cpp_concepts
  template<Eigen3::eigen_triangular_expr Arg, typename U> requires
    Eigen3::eigen_matrix<U> or Eigen3::eigen_triangular_expr<U> or Eigen3::eigen_self_adjoint_expr<U> or
    Eigen3::eigen_diagonal_expr<U> or Eigen3::euclidean_expr<U>
#else
  template<typename Arg, typename U, std::enable_if_t<Eigen3::eigen_triangular_expr<Arg> and
    (Eigen3::eigen_matrix<U> or Eigen3::eigen_triangular_expr<U> or Eigen3::eigen_self_adjoint_expr<U> or
    Eigen3::eigen_diagonal_expr<U> or Eigen3::euclidean_expr<U>), int> = 0>
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
  template<typename A, std::enable_if_t<Eigen3::eigen_triangular_expr<A>, int> = 0>
#endif
  constexpr decltype(auto)
  LQ_decomposition(A&& a)
  {
    if constexpr(lower_triangular_matrix<A>)
      return std::forward<A>(a);
    else
      return LQ_decomposition(make_native_matrix(std::forward<A>(a)));
  }


  /// Perform a QR decomposition of matrix A=Q[U,0], where U is an upper-triangular matrix, and Q is orthogonal.
  /// Returns U as an upper-triangular matrix.
#ifdef __cpp_concepts
  template<Eigen3::eigen_triangular_expr A>
#else
  template<typename A, std::enable_if_t<Eigen3::eigen_triangular_expr<A>, int> = 0>
#endif
  constexpr decltype(auto)
  QR_decomposition(A&& a)
  {
    if constexpr(upper_triangular_matrix<A>)
      return std::forward<A>(a);
    else
      return QR_decomposition(make_native_matrix(std::forward<A>(a)));
  }


} // namespace OpenKalman::Eigen3

#endif //OPENKALMAN_EIGEN3_TRIANGULARMATRIX_HPP
