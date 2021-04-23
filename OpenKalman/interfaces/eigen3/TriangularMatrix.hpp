/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2021 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef OPENKALMAN_EIGEN3_TRIANGULARMATRIX_HPP
#define OPENKALMAN_EIGEN3_TRIANGULARMATRIX_HPP

namespace OpenKalman::Eigen3
{
#ifdef __cpp_concepts
  template<typename NestedMatrix, TriangleType triangle_type> requires
    (eigen_matrix<NestedMatrix> or eigen_diagonal_expr<NestedMatrix>)
#else
  template<typename NestedMatrix, TriangleType triangle_type>
#endif
  struct TriangularMatrix
    : OpenKalman::internal::MatrixBase<TriangularMatrix<NestedMatrix, triangle_type>, NestedMatrix>
  {

#ifndef __cpp_concepts
    static_assert(eigen_matrix<NestedMatrix> or eigen_diagonal_expr<NestedMatrix>);
#endif

    static_assert(dynamic_shape<NestedMatrix> or square_matrix<NestedMatrix>);

  private:

    using Base = OpenKalman::internal::MatrixBase<TriangularMatrix, NestedMatrix>;

    static constexpr auto uplo = triangle_type == TriangleType::upper ? Eigen::Upper : Eigen::Lower;

    using View = Eigen::TriangularView<std::remove_reference_t<NestedMatrix>, uplo>;

    static constexpr auto dimensions = MatrixTraits<NestedMatrix>::rows;

  public:

    using Scalar = typename Base::Scalar;


    /// Default constructor.
#ifdef __cpp_concepts
    TriangularMatrix() requires std::default_initializable<Base>
#else
    template<typename T = Base, std::enable_if_t<std::is_default_constructible_v<T>, int> = 0>
    TriangularMatrix()
#endif
      : Base {}, view {this->nested_matrix()} {}


    /// Copy constructor.
    TriangularMatrix(const TriangularMatrix& other) : Base {other}, view {this->nested_matrix()} {}


    /// Move constructor.
    TriangularMatrix(TriangularMatrix&& other) noexcept : Base {std::move(other)}, view {this->nested_matrix()} {}


    /// Construct from a compatible triangular matrix object of the same TriangleType.
#ifdef __cpp_concepts
    template<eigen_triangular_expr Arg> requires (not std::derived_from<std::decay_t<Arg>, TriangularMatrix>) and
      OpenKalman::internal::same_triangle_type_as<Arg, TriangularMatrix> and
      (not eigen_diagonal_expr<NestedMatrix> or diagonal_matrix<nested_matrix_t<Arg>>) and
      std::is_constructible_v<Base, decltype(nested_matrix(std::declval<Arg>()))>
#else
    template<typename Arg, std::enable_if_t<eigen_triangular_expr<Arg> and
      (not std::is_base_of_v<TriangularMatrix, std::decay_t<Arg>>) and
      OpenKalman::internal::same_triangle_type_as<Arg, TriangularMatrix> and
      (not eigen_diagonal_expr<NestedMatrix> or diagonal_matrix<nested_matrix_t<Arg>>) and
      std::is_constructible_v<Base, decltype(nested_matrix(std::declval<Arg>()))>, int> = 0>
#endif
    TriangularMatrix(Arg&& arg) noexcept : Base {nested_matrix(std::forward<Arg>(arg))}, view {this->nested_matrix()} {}


      /// Construct from a compatible triangular matrix object if NestedMatrix is \ref eigen_diagonal_expr.
#ifdef __cpp_concepts
      template<eigen_triangular_expr Arg> requires (not std::derived_from<std::decay_t<Arg>, TriangularMatrix>) and
        OpenKalman::internal::same_triangle_type_as<Arg, TriangularMatrix> and
        eigen_diagonal_expr<NestedMatrix> and (not diagonal_matrix<nested_matrix_t<Arg>>) and
        std::is_constructible_v<Base, decltype(to_diagonal(diagonal_of(nested_matrix(std::declval<Arg>()))))>
#else
      template<typename Arg, std::enable_if_t<eigen_triangular_expr<Arg> and
      (not std::is_base_of_v<TriangularMatrix, std::decay_t<Arg>>) and
      OpenKalman::internal::same_triangle_type_as<Arg, TriangularMatrix> and
      eigen_diagonal_expr<NestedMatrix> and (not diagonal_matrix<nested_matrix_t<Arg>>) and
      std::is_constructible_v<Base, decltype(to_diagonal(diagonal_of(nested_matrix(std::declval<Arg>()))))>, int> = 0>
#endif
      TriangularMatrix(Arg&& arg) noexcept
        : Base {to_diagonal(diagonal_of(nested_matrix(std::forward<Arg>(arg))))}, view {this->nested_matrix()} {}


      /// Construct from an \ref eigen_diagonal_expr.
#ifdef __cpp_concepts
    template<eigen_diagonal_expr Arg> requires std::is_constructible_v<Base, Arg>
#else
    template<typename Arg, std::enable_if_t<eigen_diagonal_expr<Arg> and std::is_constructible_v<Base, Arg>, int> = 0>
#endif
    TriangularMatrix(Arg&& arg) noexcept : Base {std::forward<Arg>(arg)}, view {this->nested_matrix()} {}


    /// Construct from triangular Eigen::TriangularBase-derived object.
#ifdef __cpp_concepts
    template<triangular_matrix Arg> requires
    std::derived_from<std::decay_t<Arg>, Eigen::TriangularBase<std::decay_t<Arg>>> and
      OpenKalman::internal::same_triangle_type_as<Arg, TriangularMatrix> and
      std::is_constructible_v<Base, decltype(adjoint(std::declval<Arg>().nestedExpression()))>
#else
    template<typename Arg, std::enable_if_t<triangular_matrix<Arg> and
      std::is_base_of_v<Eigen::TriangularBase<std::decay_t<Arg>>, std::decay_t<Arg>> and
      OpenKalman::internal::same_triangle_type_as<Arg, TriangularMatrix> and
      std::is_constructible_v<Base, decltype(adjoint(std::declval<Arg>().nestedExpression()))>, int> = 0>
#endif
    TriangularMatrix(Arg&& arg)
      : Base {std::forward<decltype(arg.nestedExpression())>(arg.nestedExpression())}, view {this->nested_matrix()} {}


    /// Construct from a \ref eigen_matrix if NestedMatrix is \ref eigen_diagonal_expr.
#ifdef __cpp_concepts
    template<eigen_matrix Arg> requires eigen_diagonal_expr<NestedMatrix> and
      std::is_constructible_v<Base, decltype(diagonal_of(std::declval<Arg>()))>
#else
    template<typename Arg, std::enable_if_t<eigen_matrix<Arg> and eigen_diagonal_expr<NestedMatrix> and
      std::is_constructible_v<Base, native_matrix_t<Arg, MatrixTraits<Arg>::rows, 1>>, int> = 0>
#endif
    explicit TriangularMatrix(Arg&& arg) noexcept
      : Base {diagonal_of(std::forward<Arg>(arg))}, view {this->nested_matrix()} {}


    /// Construct from a \ref eigen_matrix if NestedMatrix is not \ref eigen_diagonal_expr.
#ifdef __cpp_concepts
    template<eigen_matrix Arg> requires (not eigen_diagonal_expr<NestedMatrix>) and std::is_constructible_v<Base, Arg>
#else
    template<typename Arg, std::enable_if_t<eigen_matrix<Arg> and (not eigen_diagonal_expr<NestedMatrix>) and
      std::is_constructible_v<Base, Arg>, int> = 0>
#endif
    explicit TriangularMatrix(Arg&& arg) noexcept : Base {std::forward<Arg>(arg)}, view {this->nested_matrix()} {}


    /**
     * \brief Construct from a list of scalar coefficients, in row-major order.
     * \details This assumes, without checking, that the coefficients represent a triangular matrix.
     * \note Operative if triangle_type is not TriangleType::diagonal.
     * \tparam Args List of scalar values.
     */
#ifdef __cpp_concepts
    template<std::convertible_to<Scalar> ... Args> requires (sizeof...(Args) > 0) and
      (triangle_type != TriangleType::diagonal or diagonal_matrix<NestedMatrix>) and
      requires { std::is_constructible_v<Base,
        decltype(MatrixTraits<NestedMatrix>::make(static_cast<const Scalar>(std::declval<const Args>())...))>; }
#else
    template<typename ... Args, std::enable_if_t<std::conjunction_v<std::is_convertible<Args, Scalar>...> and
      (triangle_type != TriangleType::diagonal or diagonal_matrix<NestedMatrix>) and
      (not zero_matrix<NestedMatrix>) and (not identity_matrix<NestedMatrix>) and
      ((diagonal_matrix<NestedMatrix> and sizeof...(Args) == dimensions) or sizeof...(Args) == dimensions * dimensions),
        int> = 0>
#endif
    TriangularMatrix(const Args ... args)
      : Base {MatrixTraits<NestedMatrix>::make(static_cast<const Scalar>(args)...)}, view {this->nested_matrix()} {}


    /**
     * \brief Construct from a list of scalar coefficients, in row-major order.
     * \details This assumes, without checking, that the coefficients represent a triangular matrix.
     * \note Operative if NestedMatrix is not \ref eigen_diagonal_expr but triangle_type is
     * TriangleType::diagonal.
     * \tparam Args List of scalar values.
     */
#if defined (__cpp_concepts) && defined (__clang__) // Because of compiler issue in at least GCC version 10.1.0
    template<std::convertible_to<Scalar> ... Args> requires (sizeof...(Args) > 0) and
      (triangle_type == TriangleType::diagonal) and (not diagonal_matrix<NestedMatrix>) and
      requires { std::is_constructible_v<Base,
        decltype(MatrixTraits<typename MatrixTraits<NestedMatrix>::template DiagonalMatrixFrom<>>::make(
          static_cast<const Scalar>(std::declval<const Args>())...))>; }
#else
    template<typename ... Args, std::enable_if_t<(sizeof...(Args) > 0) and
      std::conjunction_v<std::is_convertible<Args, Scalar>...> and
      (triangle_type == TriangleType::diagonal) and (not diagonal_matrix<NestedMatrix>) and
      (diagonal_matrix<NestedMatrix> or sizeof...(Args) == dimensions or
        sizeof...(Args) == dimensions * dimensions), int> = 0>
#endif
    TriangularMatrix(const Args ... args)
      : Base {MatrixTraits<typename MatrixTraits<NestedMatrix>::template DiagonalMatrixFrom<>>::make(
          static_cast<const Scalar>(args)...)}, view {this->nested_matrix()} {}


    /// Copy assignment operator
    TriangularMatrix& operator=(const TriangularMatrix& other)
    {
      if constexpr (not zero_matrix<NestedMatrix> and not identity_matrix<NestedMatrix>)
        if (this != &other)
        {
          this->nested_matrix().template triangularView<uplo>() = other.nested_matrix();
        }
      return *this;
    }


    /// Move assignment operator
    TriangularMatrix& operator=(TriangularMatrix&& other) noexcept
    {
      Base::operator=(std::move(other));
      return *this;
    }


    /// Assign from another triangular matrix (must be the same triangle)
#ifdef __cpp_concepts
    template<eigen_triangular_expr Arg> requires (not std::derived_from<std::decay_t<Arg>, TriangularMatrix>) and
      (MatrixTraits<Arg>::rows == dimensions) and
      OpenKalman::internal::same_triangle_type_as<Arg, TriangularMatrix> and
      modifiable<NestedMatrix, nested_matrix_t<Arg>> and
      (not (eigen_diagonal_expr<NestedMatrix> or triangle_type == TriangleType::diagonal) or diagonal_matrix<Arg>)
#else
    template<typename Arg, std::enable_if_t<eigen_triangular_expr<Arg> and
      (not std::is_base_of_v<TriangularMatrix, std::decay_t<Arg>>) and
      (MatrixTraits<Arg>::rows == dimensions) and
      OpenKalman::internal::same_triangle_type_as<Arg, TriangularMatrix> and
      modifiable<NestedMatrix, nested_matrix_t<Arg>> and
      (not (eigen_diagonal_expr<NestedMatrix> and triangle_type == TriangleType::diagonal) or diagonal_matrix<Arg>),
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
      else if constexpr (triangle_type == TriangleType::diagonal)
      {
        this->nested_matrix().diagonal() = diagonal_of(nested_matrix(std::forward<Arg>(arg)));
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


    /// Assign from a general \ref triangular_matrix.
#ifdef __cpp_concepts
    template<triangular_matrix Arg> requires (not eigen_triangular_expr<Arg>) and
      OpenKalman::internal::same_triangle_type_as<Arg, TriangularMatrix> and (MatrixTraits<Arg>::rows == dimensions) and
      modifiable<NestedMatrix, Arg>
#else
    template<typename Arg, std::enable_if_t<triangular_matrix<Arg> and (not eigen_triangular_expr<Arg>) and
      OpenKalman::internal::same_triangle_type_as<Arg, TriangularMatrix> and (MatrixTraits<Arg>::rows == dimensions) and
      modifiable<NestedMatrix, Arg>, int> = 0>
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
        this->nested_matrix() = std::forward<Arg>(arg);
      }
      return *this;
    }


    template<typename Arg>
    auto& operator+=(const TriangularMatrix<Arg, triangle_type>& arg)
    {
      static_assert(MatrixTraits<Arg>::rows == dimensions);
      this->nested_view() += arg.nested_matrix();
      return *this;
    }


    template<typename Arg>
    auto& operator-=(const TriangularMatrix<Arg, triangle_type>& arg)
    {
      static_assert(MatrixTraits<Arg>::rows == dimensions);
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
      static_assert(MatrixTraits<Arg>::rows == dimensions);
      this->nested_view() = this->nested_view() * make_native_matrix(arg);
      return *this;
    }

    constexpr auto& nested_view() & { return view; }

    constexpr const auto& nested_view() const & { return view; }

    constexpr auto&& nested_view() && { return std::move(view); }

    constexpr const auto&& nested_view() const && { return std::move(view); }


    auto operator()(std::size_t i, std::size_t j)
    {
      if constexpr (element_settable<TriangularMatrix, 2>)
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
      if constexpr(element_gettable<TriangularMatrix,1>)
        return OpenKalman::internal::ElementSetter(*this, i);
      else if constexpr(element_gettable<TriangularMatrix, 2>)
        return OpenKalman::internal::ElementSetter(*this, i, i);
      else
        return const_cast<const TriangularMatrix&>(*this)[i];
    }


    auto operator[](std::size_t i) const
    {
      if constexpr(element_gettable<TriangularMatrix, 1>)
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

    static auto identity() {return MatrixTraits<NestedMatrix>::identity(); }

  private:
    View view;
  };


  /////////////////////////////////////
  //        Deduction Guides         //
  /////////////////////////////////////

#ifdef __cpp_concepts
  template<eigen_diagonal_expr M>
#else
  template<typename M, std::enable_if_t<eigen_diagonal_expr<M>, int> = 0>
#endif
  TriangularMatrix(M&&) -> TriangularMatrix<passable_t<M>, TriangleType::diagonal>;


#ifdef __cpp_concepts
  template<eigen_matrix M>
#else
  template<typename M, std::enable_if_t<eigen_matrix<M>, int> = 0>
#endif
  explicit TriangularMatrix(M&&) -> TriangularMatrix<passable_t<M>, TriangleType::lower>;


  template<typename Arg, unsigned int UpLo>
  TriangularMatrix(const Eigen::TriangularView<Arg, UpLo>&)
  -> TriangularMatrix<Arg, UpLo & Eigen::Upper ? TriangleType::upper : TriangleType::lower>;


  /// If the arguments are a sequence of scalars, deduce a square, lower triangular matrix.
#ifdef __cpp_concepts
  template<typename Arg, typename ... Args> requires
    (std::is_arithmetic_v<std::decay_t<Arg>> and ... and std::is_arithmetic_v<std::decay_t<Args>>) and
    (std::common_with<Arg, Args> and ...)
#else
    template<typename Arg, typename ... Args, std::enable_if_t<
    (std::is_arithmetic_v<std::decay_t<Arg>> and ... and std::is_arithmetic_v<std::decay_t<Args>>), int> = 0>
#endif
  TriangularMatrix(Arg, Args ...) -> TriangularMatrix<
    Eigen::Matrix<
      std::common_type_t<std::decay_t<Arg>, std::decay_t<Args>...>,
      OpenKalman::internal::constexpr_sqrt(1 + sizeof...(Args)),
      OpenKalman::internal::constexpr_sqrt(1 + sizeof...(Args))>,
    TriangleType::lower>;


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
    return TriangularMatrix<passable_t<M>, t> (std::forward<M>(m));
  }


#ifdef __cpp_concepts
  template<TriangleType t, eigen_triangular_expr M> requires (t == MatrixTraits<M>::triangle_type)
#else
  template<TriangleType t, typename M, std::enable_if_t<eigen_triangular_expr<M> and
    (t == MatrixTraits<M>::triangle_type), int> = 0>
#endif
  auto make_EigenTriangularMatrix(M&& m)
  {
    return make_EigenTriangularMatrix<t>(nested_matrix(std::forward<M>(m)));
  }


#ifdef __cpp_concepts
  template<eigen_triangular_expr M>
#else
  template<typename M, std::enable_if_t<eigen_triangular_expr<M>, int> = 0>
#endif
  auto make_EigenTriangularMatrix(M&& m)
  {
    return make_EigenTriangularMatrix<MatrixTraits<M>::triangle_type>(nested_matrix(std::forward<M>(m)));
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
    static_assert(square_matrix<NestedMatrix>);
    using Scalar = typename MatrixTraits<NestedMatrix>::Scalar;
    static constexpr auto rows = MatrixTraits<NestedMatrix>::rows;
    static constexpr auto columns = MatrixTraits<NestedMatrix>::columns;

    template<typename Derived>
    using MatrixBaseFrom = Eigen3::internal::Eigen3MatrixBase<Derived, Eigen3::TriangularMatrix<ArgType, triangle>>;

    template<std::size_t r = rows, std::size_t c = rows, typename S = Scalar>
    using NativeMatrixFrom = native_matrix_t<NestedMatrix, r, c, S>;

    using SelfContainedFrom = Eigen3::TriangularMatrix<self_contained_t<NestedMatrix>, triangle_type>;

    template<TriangleType t = triangle_type, std::size_t dim = rows, typename S = Scalar>
    using SelfAdjointMatrixFrom = Eigen3::SelfAdjointMatrix<NativeMatrixFrom<dim, dim, S>, t>;

    template<TriangleType t = triangle_type, std::size_t dim = rows, typename S = Scalar>
    using TriangularMatrixFrom = Eigen3::TriangularMatrix<NativeMatrixFrom<dim, dim, S>, t>;

    template<std::size_t dim = rows, typename S = Scalar>
    using DiagonalMatrixFrom = Eigen3::DiagonalMatrix<NativeMatrixFrom<dim, 1, S>>;


#ifdef __cpp_concepts
    template<TriangleType t = triangle_type, typename Arg> requires (not std::convertible_to<Arg, const Scalar>)
#else
    template<TriangleType t = triangle_type, typename Arg,
      std::enable_if_t<not std::is_convertible_v<Arg, const Scalar>, int> = 0>
#endif
    static auto make(Arg&& arg) noexcept
    {
      if constexpr (Eigen3::eigen_triangular_expr<Arg>)
        return Eigen3::TriangularMatrix<std::decay_t<nested_matrix_t<Arg>>, t> {std::forward<Arg>(arg)};
      else
        return Eigen3::TriangularMatrix<std::decay_t<Arg>, t> {std::forward<Arg>(arg)};
    }


    /// Make triangular matrix using a list of coefficients in row-major order.
    /// Only the coefficients in the lower-left corner are significant.
#ifdef __cpp_concepts

    template<TriangleType t = triangle_type, std::convertible_to<Scalar> ... Args>
#else
    template<TriangleType t = triangle_type, typename ... Args,
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
  // ------------------------ //
  //        Overloads         //
  // ------------------------ //

#ifdef __cpp_concepts
  template<Eigen3::eigen_triangular_expr Arg>
#else
  template<typename Arg, std::enable_if_t<Eigen3::eigen_triangular_expr<Arg>, int> = 0>
#endif
  inline auto
  diagonal_of(Arg&& arg) noexcept
  {
    return diagonal_of(nested_matrix(std::forward<Arg>(arg)));
  }


#ifdef __cpp_concepts
  template<Eigen3::eigen_triangular_expr Arg>
#else
  template<typename Arg, std::enable_if_t<Eigen3::eigen_triangular_expr<Arg>, int> = 0>
#endif
  inline const auto
  transpose(Arg&& arg)
  {
    constexpr auto t = lower_triangular_matrix<Arg> ? TriangleType::upper : TriangleType::lower;
    const auto n = transpose(nested_matrix(std::forward<Arg>(arg)));
    return MatrixTraits<Arg>::template make<t>(std::move(n));
  }


#ifdef __cpp_concepts
  template<Eigen3::eigen_triangular_expr Arg>
#else
  template<typename Arg, std::enable_if_t<Eigen3::eigen_triangular_expr<Arg>, int> = 0>
#endif
  inline const auto
  adjoint(Arg&& arg)
  {
    constexpr auto t = lower_triangular_matrix<Arg> ? TriangleType::upper : TriangleType::lower;
    const auto n = adjoint(nested_matrix(std::forward<Arg>(arg)));
    return MatrixTraits<Arg>::template make<t>(std::move(n));
  }


#ifdef __cpp_concepts
  template<Eigen3::eigen_triangular_expr Arg, typename U> requires
    (Eigen3::eigen_matrix<U> or Eigen3::eigen_triangular_expr<U> or Eigen3::eigen_self_adjoint_expr<U> or
      Eigen3::eigen_diagonal_expr<U> or Eigen3::euclidean_expr<U>) and
    (MatrixTraits<U>::rows == MatrixTraits<Arg>::rows) and
    (not std::is_const_v<std::remove_reference_t<Arg>>)
#else
  template<typename Arg, typename U, std::enable_if_t<Eigen3::eigen_triangular_expr<Arg> and
    (Eigen3::eigen_matrix<U> or Eigen3::eigen_triangular_expr<U> or Eigen3::eigen_self_adjoint_expr<U> or
      Eigen3::eigen_diagonal_expr<U> or Eigen3::euclidean_expr<U>) and
    (MatrixTraits<U>::rows == MatrixTraits<Arg>::rows) and
    (not std::is_const_v<std::remove_reference_t<Arg>>), int> = 0>
#endif
  inline Arg&
  rank_update(Arg& arg, const U& u, const typename MatrixTraits<Arg>::Scalar alpha = 1)
  {
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
    (Eigen3::eigen_matrix<U> or Eigen3::eigen_triangular_expr<U> or Eigen3::eigen_self_adjoint_expr<U> or
    Eigen3::eigen_diagonal_expr<U> or Eigen3::euclidean_expr<U>) and
    (MatrixTraits<U>::rows == MatrixTraits<Arg>::rows)
#else
  template<typename Arg, typename U, std::enable_if_t<Eigen3::eigen_triangular_expr<Arg> and
    (Eigen3::eigen_matrix<U> or Eigen3::eigen_triangular_expr<U> or Eigen3::eigen_self_adjoint_expr<U> or
    Eigen3::eigen_diagonal_expr<U> or Eigen3::euclidean_expr<U>) and
    (MatrixTraits<U>::rows == MatrixTraits<Arg>::rows), int> = 0>
#endif
  inline auto
  rank_update(Arg&& arg, const U& u, const typename MatrixTraits<Arg>::Scalar alpha = 1)
  {
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
