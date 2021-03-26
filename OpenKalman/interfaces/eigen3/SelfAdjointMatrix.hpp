/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2021 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief SelfAdjointMatrix and related definitions.
 */

#ifndef OPENKALMAN_EIGEN3_SELFADJOINTMATRIX_HPP
#define OPENKALMAN_EIGEN3_SELFADJOINTMATRIX_HPP

namespace OpenKalman::Eigen3
{
#ifdef __cpp_concepts
  template<square_matrix NestedMatrix, TriangleType storage_triangle>
#else
  template<typename NestedMatrix, TriangleType storage_triangle>
#endif
  struct SelfAdjointMatrix
    : OpenKalman::internal::MatrixBase<SelfAdjointMatrix<NestedMatrix, storage_triangle>, NestedMatrix>
  {
    static_assert(square_matrix<NestedMatrix>);
    using Base = OpenKalman::internal::MatrixBase<SelfAdjointMatrix, NestedMatrix>;
    static constexpr auto uplo = storage_triangle == TriangleType::upper ? Eigen::Upper : Eigen::Lower;
    using View = Eigen::SelfAdjointView<std::remove_reference_t<NestedMatrix>, uplo>;
    using Scalar = typename MatrixTraits<NestedMatrix>::Scalar;
    static constexpr auto dimension = MatrixTraits<NestedMatrix>::dimension;


    /// Default constructor.
#ifdef __cpp_concepts
    SelfAdjointMatrix() requires std::default_initializable<Base>
#else
    template<typename T = Base, std::enable_if_t<std::is_default_constructible_v<T>, int> = 0>
    SelfAdjointMatrix()
#endif
      : Base {}, view {this->nested_matrix()} {}


    /// Copy constructor.
    SelfAdjointMatrix(const SelfAdjointMatrix& other) : Base {other}, view {this->nested_matrix()} {}


    /// Move constructor.
    SelfAdjointMatrix(SelfAdjointMatrix&& other) noexcept : Base {std::move(other)}, view {this->nested_matrix()} {}


    /// Construct from a compatible self-adjoint matrix object of the same storage type
#ifdef __cpp_concepts
    template<eigen_self_adjoint_expr Arg> requires (not std::derived_from<std::decay_t<Arg>, SelfAdjointMatrix>) and
      internal::same_storage_triangle_as<Arg, SelfAdjointMatrix> and
      (not eigen_diagonal_expr<NestedMatrix> or diagonal_matrix<nested_matrix_t<Arg>>) and
      std::is_constructible_v<Base, decltype(nested_matrix(std::declval<Arg>()))>
#else
    template<typename Arg, std::enable_if_t<eigen_self_adjoint_expr<Arg> and
      (not std::is_base_of_v<SelfAdjointMatrix, std::decay_t<Arg>>) and
      internal::same_storage_triangle_as<Arg, SelfAdjointMatrix> and
      (not eigen_diagonal_expr<NestedMatrix> or diagonal_matrix<nested_matrix_t<Arg>>) and
      std::is_constructible_v<Base, decltype(nested_matrix(std::declval<Arg>()))>, int> = 0>
#endif
    SelfAdjointMatrix(Arg&& arg) noexcept
      : Base {nested_matrix(std::forward<Arg>(arg))}, view {this->nested_matrix()} {}


    /// Construct from a compatible self-adjoint matrix object if NestedMatrix is \ref eigen_diagonal_expr.
#ifdef __cpp_concepts
    template<eigen_self_adjoint_expr Arg> requires (not std::derived_from<std::decay_t<Arg>, SelfAdjointMatrix>) and
      internal::same_storage_triangle_as<Arg, SelfAdjointMatrix> and
      eigen_diagonal_expr<NestedMatrix> and (not diagonal_matrix<nested_matrix_t<Arg>>) and
      std::is_constructible_v<Base, decltype(to_diagonal(diagonal_of(nested_matrix(std::declval<Arg>()))))>
#else
    template<typename Arg, std::enable_if_t<eigen_self_adjoint_expr<Arg> and
      (not std::is_base_of_v<SelfAdjointMatrix, std::decay_t<Arg>>) and
      internal::same_storage_triangle_as<Arg, SelfAdjointMatrix> and
      eigen_diagonal_expr<NestedMatrix> and (not diagonal_matrix<nested_matrix_t<Arg>>) and
      std::is_constructible_v<Base, decltype(to_diagonal(diagonal_of(nested_matrix(std::declval<Arg>()))))>, int> = 0>
#endif
    SelfAdjointMatrix(Arg&& arg) noexcept
      : Base {to_diagonal(diagonal_of(nested_matrix(std::forward<Arg>(arg))))}, view {this->nested_matrix()} {}


    /// Construct from a compatible self-adjoint matrix object of the opposite storage type
#ifdef __cpp_concepts
    template<eigen_self_adjoint_expr Arg> requires (not internal::same_storage_triangle_as<Arg, SelfAdjointMatrix>) and
      std::is_constructible_v<Base, decltype(adjoint(nested_matrix(std::declval<Arg>())))>
#else
    template<typename Arg, std::enable_if_t<eigen_self_adjoint_expr<Arg> and
      (not internal::same_storage_triangle_as<Arg, SelfAdjointMatrix>) and
      std::is_constructible_v<Base, decltype(adjoint(nested_matrix(std::declval<Arg>())))>, int> = 0>
#endif
    SelfAdjointMatrix(Arg&& arg) noexcept
      : Base {adjoint(nested_matrix(std::forward<Arg>(arg)))}, view {this->nested_matrix()} {}


    /// Construct from an \ref eigen_diagonal_expr.
#ifdef __cpp_concepts
    template<eigen_diagonal_expr Arg> requires std::is_constructible_v<Base, Arg>
#else
    template<typename Arg, std::enable_if_t<eigen_diagonal_expr<Arg> and std::is_constructible_v<Base, Arg>, int> = 0>
#endif
    SelfAdjointMatrix(Arg&& arg) noexcept
      : Base {std::forward<Arg>(arg)}, view {this->nested_matrix()} {}


    /// Construct from a self-adjoint Eigen::TriangularBase-derived object of the same storage type.
#ifdef __cpp_concepts
    template<self_adjoint_matrix Arg> requires
      std::derived_from<std::decay_t<Arg>, Eigen::TriangularBase<std::decay_t<Arg>>> and
      internal::same_storage_triangle_as<Arg, SelfAdjointMatrix> and
      std::is_constructible_v<Base, decltype(std::declval<Arg>().nestedExpression())>
#else
    template<typename Arg, std::enable_if_t<self_adjoint_matrix<Arg> and
      std::is_base_of_v<Eigen::TriangularBase<std::decay_t<Arg>>, std::decay_t<Arg>> and
      internal::same_storage_triangle_as<Arg, SelfAdjointMatrix> and
      std::is_constructible_v<Base, decltype(std::declval<Arg>().nestedExpression())>, int> = 0>
#endif
    SelfAdjointMatrix(Arg&& arg)
      : Base {std::forward<decltype(arg.nestedExpression())>(arg.nestedExpression())}, view {this->nested_matrix()} {}


    /// Construct from a self-adjoint Eigen::TriangularBase-derived object of the opposite storage type.
#ifdef __cpp_concepts
    template<self_adjoint_matrix Arg> requires
    std::derived_from<std::decay_t<Arg>, Eigen::TriangularBase<std::decay_t<Arg>>> and
      (not internal::same_storage_triangle_as<Arg, SelfAdjointMatrix>) and
      std::is_constructible_v<Base, decltype(adjoint(std::declval<Arg>().nestedExpression()))>
#else
    template<typename Arg, std::enable_if_t<self_adjoint_matrix<Arg> and
      std::is_base_of_v<Eigen::TriangularBase<std::decay_t<Arg>>, std::decay_t<Arg>> and
      (not internal::same_storage_triangle_as<Arg, SelfAdjointMatrix>) and
      std::is_constructible_v<Base, decltype(adjoint(std::declval<Arg>().nestedExpression()))>, int> = 0>
#endif
    SelfAdjointMatrix(Arg&& arg)
      : Base {adjoint(std::forward<decltype(arg.nestedExpression())>(arg.nestedExpression()))},
        view {this->nested_matrix()} {}


    /// Construct from a square \ref eigen_matrix if NestedMatrix is \ref eigen_diagonal_expr.
#ifdef __cpp_concepts
    template<eigen_matrix Arg> requires eigen_diagonal_expr<NestedMatrix> and
      square_matrix<Arg> and requires { std::is_constructible_v<Base, decltype(diagonal_of(std::declval<Arg>()))>; }
#else
    template<typename Arg, std::enable_if_t<eigen_matrix<Arg> and eigen_diagonal_expr<NestedMatrix> and
      square_matrix<Arg> and
      std::is_constructible_v<Base, native_matrix_t<Arg, MatrixTraits<Arg>::dimension, 1>>, int> = 0>
#endif
    explicit SelfAdjointMatrix(Arg&& arg) noexcept
      : Base {diagonal_of(std::forward<Arg>(arg))}, view {this->nested_matrix()} {}


    /// Construct from a \ref eigen_matrix if NestedMatrix is not \ref eigen_diagonal_expr.
#ifdef __cpp_concepts
    template<eigen_matrix Arg> requires (not eigen_diagonal_expr<NestedMatrix>) and
      square_matrix<Arg> and std::is_constructible_v<Base, Arg>
#else
    template<typename Arg, std::enable_if_t<eigen_matrix<Arg> and (not eigen_diagonal_expr<NestedMatrix>) and
      square_matrix<Arg> and std::is_constructible_v<Base, Arg>, int> = 0>
#endif
    explicit SelfAdjointMatrix(Arg&& arg) noexcept
      : Base {std::forward<Arg>(arg)}, view {this->nested_matrix()} {}


    /**
     * \brief Construct from a list of scalar coefficients, in row-major order.
     * \details This assumes, without checking, that the coefficients represent a self-adjoint matrix.
     * \note Operative if \ref storage_triangle is not TriangleType::diagonal.
     * \tparam Args List of scalar values.
     */
#ifdef __cpp_concepts
    template<std::convertible_to<Scalar> ... Args> requires (sizeof...(Args) > 0) and
      (storage_triangle != TriangleType::diagonal or diagonal_matrix<NestedMatrix>) and
      requires { std::is_constructible_v<Base,
        decltype(MatrixTraits<NestedMatrix>::make(static_cast<const Scalar>(std::declval<const Args>())...))>; }
#else
    template<typename ... Args, std::enable_if_t<std::conjunction_v<std::is_convertible<Args, Scalar>...> and
      (storage_triangle != TriangleType::diagonal or diagonal_matrix<NestedMatrix>) and
      (not zero_matrix<NestedMatrix>) and (not identity_matrix<NestedMatrix>) and
      ((diagonal_matrix<NestedMatrix> and sizeof...(Args) == dimension) or sizeof...(Args) == dimension * dimension),
        int> = 0>
#endif
    SelfAdjointMatrix(const Args ... args)
      : Base {MatrixTraits<NestedMatrix>::make(static_cast<const Scalar>(args)...)}, view {this->nested_matrix()} {}


    /**
     * \brief Construct from a list of scalar coefficients, in row-major order.
     * \details This assumes, without checking, that the coefficients represent a self-adjoint matrix.
     * \note Operative if NestedMatrix is not a \ref diagonal_matrix but \ref storage_triangle is
     * TriangleType::diagonal.
     * \tparam Args List of scalar values.
     */
#if defined (__cpp_concepts) && defined (__clang__) // Because of compiler issue in at least GCC version 10.1.0
    template<std::convertible_to<Scalar> ... Args> requires (sizeof...(Args) > 0) and
      (storage_triangle == TriangleType::diagonal) and (not diagonal_matrix<NestedMatrix>) and
      requires { std::is_constructible_v<Base,
        decltype(MatrixTraits<typename MatrixTraits<NestedMatrix>::template DiagonalMatrixFrom<>>::make(
          static_cast<const Scalar>(std::declval<const Args>())...))>; }
#else
    template<typename ... Args, std::enable_if_t<(sizeof...(Args) > 0) and
      std::conjunction_v<std::is_convertible<Args, Scalar>...> and
      (storage_triangle == TriangleType::diagonal) and (not diagonal_matrix<NestedMatrix>) and
      (diagonal_matrix<NestedMatrix> or sizeof...(Args) == dimension or
        sizeof...(Args) == dimension * dimension), int> = 0>
#endif
    SelfAdjointMatrix(const Args ... args)
      : Base {MatrixTraits<typename MatrixTraits<NestedMatrix>::template DiagonalMatrixFrom<>>::make(
          static_cast<const Scalar>(args)...)}, view {this->nested_matrix()} {}


    /** Copy assignment operator
     * \param other Another SelfAdjointMatrix
     * \return Reference to this.
     */
    auto& operator=(const SelfAdjointMatrix& other)
    {
      if constexpr (not zero_matrix<NestedMatrix> and not identity_matrix<NestedMatrix>)
        if (this != &other)
        {
          this->nested_matrix().template triangularView<uplo>() = other.nested_matrix();
        }
      return *this;
    }


    /** Move assignment operator
     * \param other A SelfAdjointMatrix temporary value.
     * \return Reference to this.
     */
    auto& operator=(SelfAdjointMatrix&& other) noexcept
    {
      Base::operator=(std::move(other));
      return *this;
    }


    /// Assign from another self-adjoint matrix.
#ifdef __cpp_concepts
    template<eigen_self_adjoint_expr Arg> requires (not std::derived_from<std::decay_t<Arg>, SelfAdjointMatrix>) and
      (MatrixTraits<Arg>::dimension == dimension) and
      modifiable<NestedMatrix, nested_matrix_t<Arg>> and
      (not (eigen_diagonal_expr<NestedMatrix> and storage_triangle == TriangleType::diagonal) or diagonal_matrix<Arg>)
#else
    template<typename Arg, std::enable_if_t<eigen_self_adjoint_expr<Arg> and
      (not std::is_base_of_v<SelfAdjointMatrix, std::decay_t<Arg>>) and (MatrixTraits<Arg>::dimension == dimension) and
      modifiable<NestedMatrix, nested_matrix_t<Arg>> and
      ((not eigen_diagonal_expr<NestedMatrix> and storage_triangle != TriangleType::diagonal) or diagonal_matrix<Arg>),
      int> = 0>
#endif
    auto& operator=(Arg&& arg)
    {
      if constexpr (zero_matrix<NestedMatrix> or identity_matrix<NestedMatrix>)
      {}
      else if constexpr (eigen_diagonal_expr<NestedMatrix>)
      {
        this->nested_matrix().nested_matrix() = diagonal_of(nested_matrix(std::forward<Arg>(arg)));
      }
      else if constexpr (storage_triangle == TriangleType::diagonal)
      {
        this->nested_matrix().diagonal() = diagonal_of(nested_matrix(std::forward<Arg>(arg)));
      }
      else if constexpr (not internal::same_storage_triangle_as<Arg, SelfAdjointMatrix>)
      {
        this->nested_matrix().template triangularView<uplo>() = adjoint(nested_matrix(std::forward<Arg>(arg)));
      }
      else if constexpr(std::is_lvalue_reference_v<Arg>)
      {
        this->nested_matrix().template triangularView<uplo>() = nested_matrix(arg);
      }
      else
      {
        this->nested_matrix() = nested_matrix(std::forward<Arg>(arg));
      }
      return *this;
    }


    /// Assign from a general \ref self_adjoint_matrix.
#ifdef __cpp_concepts
    template<self_adjoint_matrix Arg> requires (not eigen_self_adjoint_expr<Arg>) and
      (MatrixTraits<Arg>::dimension == dimension) and modifiable<NestedMatrix, Arg>
#else
    template<typename Arg, std::enable_if_t<self_adjoint_matrix<Arg> and (not eigen_self_adjoint_expr<Arg>) and
      (MatrixTraits<Arg>::dimension == dimension) and modifiable<NestedMatrix, Arg>, int> = 0>
#endif
    auto& operator=(Arg&& arg)
    {
      if constexpr (zero_matrix<NestedMatrix> or identity_matrix<NestedMatrix>)
      {}
      else if constexpr (zero_matrix<Arg>)
      {
        this->nested_matrix() = MatrixTraits<NestedMatrix>::zero();
      }
      else if constexpr (identity_matrix<Arg>)
      {
        this->nested_matrix() = MatrixTraits<NestedMatrix>::identity();
      }
      else if constexpr (std::is_base_of_v<Eigen::TriangularBase<std::decay_t<Arg>>, std::decay_t<Arg>>)
      {
        if constexpr(internal::same_storage_triangle_as<Arg, SelfAdjointMatrix>)
        {
          if constexpr (std::is_rvalue_reference_v<Arg>)
          {
            this->nested_matrix() = std::move(arg.nestedExpression());
          }
          else
          {
            this->nested_matrix().template triangularView<uplo>() = arg.nestedExpression();
          }
        }
        else
        {
          this->nested_matrix().template triangularView<uplo>() = arg.nestedExpression().adjoint();
        }
      }
      else
      {
        this->nested_matrix() = std::forward<Arg>(arg);
      }
      return *this;
    }


    template<typename Arg, TriangleType t>
    auto& operator+=(const SelfAdjointMatrix<Arg, t>& arg)
    {
      static_assert(MatrixTraits<Arg>::dimension == dimension);
      if constexpr(t == storage_triangle)
        this->nested_matrix().template triangularView<uplo>() += arg.nested_matrix();
      else
        this->nested_matrix().template triangularView<uplo>() += arg.nested_matrix().adjoint();
      return *this;
    }


    template<typename Arg, TriangleType t>
    auto& operator-=(const SelfAdjointMatrix<Arg, t>& arg)
    {
      static_assert(MatrixTraits<Arg>::dimension == dimension);
      if constexpr(t == storage_triangle)
        this->nested_matrix().template triangularView<uplo>() -= arg.nested_matrix();
      else
        this->nested_matrix().template triangularView<uplo>() -= arg.nested_matrix().adjoint();
      return *this;
    }


#ifdef __cpp_concepts
    template<std::convertible_to<Scalar> S>
#else
    template<typename S, std::enable_if_t<std::is_convertible_v<S, Scalar>, int> = 0>
#endif
    auto& operator*=(const S s)
    {
      this->nested_matrix().template triangularView<uplo>() *= s;
      return *this;
    }


#ifdef __cpp_concepts
    template<std::convertible_to<Scalar> S>
#else
    template<typename S, std::enable_if_t<std::is_convertible_v<S, Scalar>, int> = 0>
#endif
    auto& operator/=(const S s)
    {
      this->nested_matrix().template triangularView<uplo>() /= s;
      return *this;
    }


    constexpr auto& nested_view() & { return view; }

    constexpr const auto& nested_view() const & { return view; }

    constexpr auto&& nested_view() && { return std::move(view); }

    constexpr const auto&& nested_view() const && { return std::move(view); }


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
      auto llt = this->nested_view().llt();
      M ret = llt.solve(b);
      if (llt.info() != Eigen::Success) [[unlikely]]
      {
        // A is semidefinite. Use LDLT decomposition instead.
        auto ldlt = this->nested_view().ldlt();
        M ret2 = ldlt.solve(b);
        if ((not ldlt.isPositive() and not ldlt.isNegative()) or ldlt.info() != Eigen::Success)
        {
          throw (std::runtime_error("SelfAdjointMatrix solve: A is indefinite"));
        }
        return ret2;
      }
      return ret;
    }

    static auto zero() { return MatrixTraits<NestedMatrix>::zero(); }

    static auto identity() { return MatrixTraits<NestedMatrix>::identity(); }

  private:
    View view;
  };


  /////////////////////////////////////
  //        Deduction Guides         //
  /////////////////////////////////////

#ifdef __cpp_concepts
  template<eigen_diagonal_expr Arg>
#else
  template<typename Arg, std::enable_if_t<eigen_diagonal_expr<Arg>, int> = 0>
#endif
  SelfAdjointMatrix(Arg&&) -> SelfAdjointMatrix<passable_t<Arg>, TriangleType::diagonal>;


#ifdef __cpp_concepts
  template<eigen_matrix Arg>
#else
  template<typename Arg, std::enable_if_t<eigen_matrix<Arg>, int> = 0>
#endif
  explicit SelfAdjointMatrix(Arg&&) -> SelfAdjointMatrix<passable_t<Arg>, TriangleType::lower>;


  template<typename Arg, unsigned int UpLo>
  SelfAdjointMatrix(Eigen::SelfAdjointView<Arg, UpLo>&&)
  -> SelfAdjointMatrix<Arg, UpLo & Eigen::Upper ? TriangleType::upper : TriangleType::lower>;


  /// If the arguments are a sequence of scalars, deduce a square, self-adjoint matrix.
#ifdef __cpp_concepts
  template<typename Arg, typename ... Args> requires
    (std::is_arithmetic_v<std::decay_t<Arg>> and ... and std::is_arithmetic_v<std::decay_t<Args>>) and
    (std::common_with<Arg, Args> and ...)
#else
    template<typename Arg, typename ... Args, std::enable_if_t<
    (std::is_arithmetic_v<std::decay_t<Arg>> and ... and std::is_arithmetic_v<std::decay_t<Args>>), int> = 0>
#endif
  SelfAdjointMatrix(Arg, Args ...) -> SelfAdjointMatrix<
    Eigen::Matrix<
      std::common_type_t<std::decay_t<Arg>, std::decay_t<Args>...>,
      OpenKalman::internal::constexpr_sqrt(1 + sizeof...(Args)),
      OpenKalman::internal::constexpr_sqrt(1 + sizeof...(Args))>,
    TriangleType::lower>;


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
    if constexpr(t == MatrixTraits<M>::storage_triangle)
      return make_EigenSelfAdjointMatrix<t>(nested_matrix(std::forward<M>(m)));
    else
      return make_EigenSelfAdjointMatrix<t>(adjoint(nested_matrix(std::forward<M>(m))));
  }


#ifdef __cpp_concepts
  template<eigen_self_adjoint_expr M>
#else
  template<typename M, std::enable_if_t<eigen_self_adjoint_expr<M>, int> = 0>
#endif
  auto
  make_EigenSelfAdjointMatrix(M&& m)
  {
    return make_EigenSelfAdjointMatrix<MatrixTraits<M>::storage_triangle>(std::forward<M>(m));
  }

} // OpenKalman::Eigen3

namespace OpenKalman
{
  ///////////////////////////
  //        Traits         //
  ///////////////////////////

  template<typename ArgType, TriangleType triangle_type>
  struct MatrixTraits<Eigen3::SelfAdjointMatrix<ArgType, triangle_type>>
  {
    static constexpr TriangleType storage_triangle = triangle_type;
    using NestedMatrix = ArgType;
    using Scalar = typename MatrixTraits<NestedMatrix>::Scalar;
    static constexpr auto dimension = MatrixTraits<NestedMatrix>::dimension;
    static constexpr auto columns = MatrixTraits<NestedMatrix>::columns;

    template<typename Derived>
    using MatrixBaseFrom =
      Eigen3::internal::Eigen3MatrixBase<Derived, Eigen3::SelfAdjointMatrix<ArgType, storage_triangle>>;

    template<std::size_t rows = dimension, std::size_t cols = dimension, typename S = Scalar>
    using NativeMatrixFrom = native_matrix_t<NestedMatrix, rows, cols, S>;

    using SelfContainedFrom = Eigen3::SelfAdjointMatrix<self_contained_t<NestedMatrix>, storage_triangle>;

    template<TriangleType t = storage_triangle, std::size_t dim = dimension, typename S = Scalar>
    using SelfAdjointMatrixFrom = Eigen3::SelfAdjointMatrix<NativeMatrixFrom<dim, dim, S>, t>;

    template<TriangleType t = storage_triangle, std::size_t dim = dimension, typename S = Scalar>
    using TriangularMatrixFrom = Eigen3::TriangularMatrix<NativeMatrixFrom<dim, dim, S>, t>;

    template<std::size_t dim = dimension, typename S = Scalar>
    using DiagonalMatrixFrom = Eigen3::DiagonalMatrix<NativeMatrixFrom<dim, 1, S>>;


#ifdef __cpp_concepts
    template<TriangleType t = storage_triangle, typename Arg> requires (not std::convertible_to<Arg, const Scalar>)
#else
    template<TriangleType t = storage_triangle, typename Arg,
      std::enable_if_t<not std::is_convertible_v<Arg, const Scalar>, int> = 0>
#endif
    static auto make(Arg&& arg) noexcept
    {
      return Eigen3::SelfAdjointMatrix<std::decay_t<Arg>, t>(std::forward<Arg>(arg));
    }


    /// Make self-adjoint matrix using a list of coefficients in row-major order.
    /// Only the coefficients in the lower-left corner are significant.
#ifdef __cpp_concepts
    template<TriangleType t = storage_triangle, std::convertible_to<Scalar> ... Args>
#else
    template<TriangleType t = storage_triangle, typename ... Args,
      std::enable_if_t<std::conjunction_v<std::is_convertible<Args, Scalar>...>, int> = 0>
#endif
    static auto make(const Args ... args)
    {
      return make<t>(MatrixTraits<NestedMatrix>::make(static_cast<const Scalar>(args)...));
    }


    static auto zero() { return MatrixTraits<NestedMatrix>::zero(); }

    static auto identity() { return MatrixTraits<NestedMatrix>::identity(); }

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
  diagonal_of(Arg&& arg) noexcept
  {
    return diagonal_of(nested_matrix(std::forward<Arg>(arg)));
  }


#ifdef __cpp_concepts
  template<Eigen3::eigen_self_adjoint_expr Arg>
#else
  template<typename Arg, std::enable_if_t<Eigen3::eigen_self_adjoint_expr<Arg>, int> = 0>
#endif
  inline const auto
  transpose(Arg && arg)
  {
    constexpr auto t = Eigen3::lower_triangular_storage<Arg> ? TriangleType::upper : TriangleType::lower;
    const auto n = transpose(nested_matrix(std::forward<Arg>(arg)));
    return MatrixTraits<Arg>::template make<t>(std::move(n));
  }


#ifdef __cpp_concepts
  template<Eigen3::eigen_self_adjoint_expr Arg>
#else
  template<typename Arg, std::enable_if_t<Eigen3::eigen_self_adjoint_expr<Arg>, int> = 0>
#endif
  inline const auto
  adjoint(Arg && arg)
  {
    constexpr auto t = Eigen3::lower_triangular_storage<Arg> ? TriangleType::upper : TriangleType::lower;
    const auto n = adjoint(nested_matrix(std::forward<Arg>(arg)));
    return MatrixTraits<Arg>::template make<t>(std::move(n));
  }


#ifdef __cpp_concepts
  template<Eigen3::eigen_self_adjoint_expr Arg, typename U> requires
    (Eigen3::eigen_matrix<U> or Eigen3::eigen_triangular_expr<U> or Eigen3::eigen_self_adjoint_expr<U> or
    Eigen3::eigen_diagonal_expr<U> or Eigen3::euclidean_expr<U>) and
    (MatrixTraits<U>::dimension == MatrixTraits<Arg>::dimension) and
    (not std::is_const_v<std::remove_reference_t<Arg>>)
#else
  template<typename Arg, typename U, std::enable_if_t<
    Eigen3::eigen_self_adjoint_expr<Arg> and
    (Eigen3::eigen_matrix<U> or Eigen3::eigen_triangular_expr<U> or Eigen3::eigen_self_adjoint_expr<U> or
    Eigen3::eigen_diagonal_expr<U> or Eigen3::euclidean_expr<U>) and
    (MatrixTraits<U>::dimension == MatrixTraits<Arg>::dimension) and
    (not std::is_const_v<std::remove_reference_t<Arg>>), int> = 0>
#endif
  inline Arg&
  rank_update(Arg& arg, const U& u, const typename MatrixTraits<Arg>::Scalar alpha = 1)
  {
    arg.nested_view().rankUpdate(u, alpha);
    return arg;
  }


#ifdef __cpp_concepts
  template<Eigen3::eigen_self_adjoint_expr Arg, typename U> requires
    (Eigen3::eigen_matrix<U> or Eigen3::eigen_triangular_expr<U> or Eigen3::eigen_self_adjoint_expr<U> or
    Eigen3::eigen_diagonal_expr<U> or Eigen3::euclidean_expr<U>) and
    (MatrixTraits<U>::dimension == MatrixTraits<Arg>::dimension)
#else
  template<typename Arg, typename U, std::enable_if_t<
    Eigen3::eigen_self_adjoint_expr<Arg> and
    (Eigen3::eigen_matrix<U> or Eigen3::eigen_triangular_expr<U> or Eigen3::eigen_self_adjoint_expr<U> or
    Eigen3::eigen_diagonal_expr<U> or Eigen3::euclidean_expr<U>) and
    (MatrixTraits<U>::dimension == MatrixTraits<Arg>::dimension), int> = 0>
#endif
  inline auto
  rank_update(Arg && arg, const U& u, const typename MatrixTraits<Arg>::Scalar alpha = 1)
  {
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
