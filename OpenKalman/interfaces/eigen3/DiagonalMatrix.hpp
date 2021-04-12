/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2020-2021 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef OPENKALMAN_EIGEN3_DIAGONALMATRIX_HPP
#define OPENKALMAN_EIGEN3_DIAGONALMATRIX_HPP

namespace OpenKalman::Eigen3
{
#ifdef __cpp_concepts
  template<column_vector NestedMatrix> requires eigen_matrix<NestedMatrix>
#else
  template<typename NestedMatrix>
#endif
  struct DiagonalMatrix
    : OpenKalman::internal::MatrixBase<DiagonalMatrix<NestedMatrix>, NestedMatrix>
  {

#ifndef __cpp_concepts
    static_assert(column_vector<NestedMatrix>);
    static_assert(eigen_matrix<NestedMatrix>);
#endif

    using Base = OpenKalman::internal::MatrixBase<DiagonalMatrix, NestedMatrix>;
    using Scalar = typename MatrixTraits<NestedMatrix>::Scalar;
    static constexpr auto dimensions = MatrixTraits<NestedMatrix>::rows;


    /// Default constructor.
#ifdef __cpp_concepts
    DiagonalMatrix() requires std::default_initializable<Base>
#else
    template<typename T = Base, std::enable_if_t<std::is_default_constructible_v<T>, int> = 0>
    DiagonalMatrix()
#endif
      : Base {} {}


    /// Copy constructor.
    DiagonalMatrix(const DiagonalMatrix& other) : Base {other} {}


    /// Move constructor.
    DiagonalMatrix(DiagonalMatrix&& other) noexcept: Base {std::move(other)} {}


    /// Construct from a compatible \ref eigen_diagonal_expr.
#ifdef __cpp_concepts
    template<eigen_diagonal_expr Arg> requires (not std::derived_from<std::decay_t<Arg>, DiagonalMatrix>) and
      std::is_constructible_v<Base, decltype(nested_matrix(std::declval<Arg>()))>
#else
    template<typename Arg, std::enable_if_t<eigen_diagonal_expr<Arg> and
      (not std::is_base_of_v<DiagonalMatrix, std::decay_t<Arg>>) and
      std::is_constructible_v<Base, decltype(nested_matrix(std::declval<Arg>()))>, int> = 0>
#endif
    DiagonalMatrix(Arg&& other) noexcept : Base {nested_matrix(std::forward<Arg>(other))} {}


    /// Construct from a zero matrix.
#if defined(__cpp_concepts) and defined(__cpp_conditional_explicit)
    template<zero_matrix Arg> requires (not eigen_diagonal_expr<Arg>) and
      (column_vector<Arg> or square_matrix<Arg>) and std::is_constructible_v<Base, ZeroMatrix<Scalar, dimensions, 1>&&>
    explicit (column_vector<Arg> and not square_matrix<Arg>)
    DiagonalMatrix(Arg&&) : Base {ZeroMatrix<Scalar, dimensions, 1> {}} {}
#else
    template<typename Arg, std::enable_if_t<zero_matrix<Arg> and (not eigen_diagonal_expr<Arg>) and
      std::is_constructible_v<Base, ZeroMatrix<Scalar, dimensions, 1>&&> and
      column_vector<Arg> and (not square_matrix<Arg>), int> = 0>
    explicit DiagonalMatrix(Arg&&) : Base {ZeroMatrix<Scalar, dimensions, 1> {}} {}

    /// \overload
    template<typename Arg, std::enable_if_t<zero_matrix<Arg> and (not eigen_diagonal_expr<Arg>) and
      std::is_constructible_v<Base, ZeroMatrix<Scalar, dimensions, 1>&&> and
      (not column_vector<Arg> or square_matrix<Arg>), int> = 0>
    DiagonalMatrix(Arg&&) : Base {ZeroMatrix<Scalar, dimensions, 1> {}} {}
#endif


    /// Construct from an identity matrix.
#ifdef __cpp_concepts
    template<identity_matrix Arg> requires (not eigen_diagonal_expr<Arg>) and (not one_by_one_matrix<Arg>) and
      requires { std::is_constructible_v<Base, typename Eigen::Matrix<Scalar, dimensions, 1>::ConstantReturnType&&>; }
#else
    template<typename Arg, std::enable_if_t<identity_matrix<Arg> and
      (not eigen_diagonal_expr<Arg>) and (not one_by_one_matrix<Arg>) and
      std::is_constructible_v<Base, typename Eigen::Matrix<Scalar, dimensions, 1>::ConstantReturnType&&>, int> = 0>
#endif
    DiagonalMatrix(const Arg&) : Base {Eigen::Matrix<Scalar, dimensions, 1>::Constant(1)} {}


    /// Construct from a compatible \ref diagonal_matrix, general case.
#ifdef __cpp_concepts
    template<diagonal_matrix Arg> requires (not eigen_diagonal_expr<Arg>) and
      (not zero_matrix<Arg>) and (not identity_matrix<Arg> or one_by_one_matrix<Arg>) and
      requires { std::is_constructible_v<Base, decltype(make_self_contained(diagonal_of(std::declval<Arg>())))>; }
#else
    template<typename Arg, std::enable_if_t<diagonal_matrix<Arg> and (not eigen_diagonal_expr<Arg>) and
      (not zero_matrix<Arg>) and (not identity_matrix<Arg> or one_by_one_matrix<Arg>), int> = 0>
#endif
    DiagonalMatrix(Arg&& other) noexcept: Base {make_self_contained(diagonal_of(std::forward<Arg>(other)))} {}


    /// Construct from a \ref column_vector \ref eigen_matrix.
#ifdef __cpp_concepts
    template<eigen_matrix Arg> requires (not diagonal_matrix<Arg>) and column_vector<Arg> and
      (not zero_matrix<Arg>) and std::is_constructible_v<Base, Arg>
#else
    template<typename Arg, std::enable_if_t<eigen_matrix<Arg> and (not diagonal_matrix<Arg>) and
      column_vector<Arg> and (not zero_matrix<Arg>) and std::is_constructible_v<Base, Arg>, int> = 0>
#endif
    explicit DiagonalMatrix(Arg&& arg) noexcept : Base {std::forward<Arg>(arg)} {}


    /// Construct from a \ref square_matrix "square" \ref eigen_matrix.
#ifdef __cpp_concepts
    template<eigen_matrix Arg> requires (not diagonal_matrix<Arg>) and (not column_vector<Arg>) and
      square_matrix<Arg> and requires { std::is_constructible_v<Base, decltype(diagonal_of(std::declval<Arg>()))>; }
#else
    template<typename Arg, std::enable_if_t<eigen_matrix<Arg> and
      (not diagonal_matrix<Arg>) and (not column_vector<Arg>) and square_matrix<Arg> and
      std::is_constructible_v<Base, native_matrix_t<Arg, MatrixTraits<Arg>::rows, 1>>, int> = 0>
#endif
    explicit DiagonalMatrix(Arg&& arg) noexcept : Base {diagonal_of(std::forward<Arg>(arg))} {}


    /**
     * \brief Construct from a list of scalar coefficients that define the diagonal.
     */
#ifdef __cpp_concepts
    template<std::convertible_to<Scalar> ... Args> requires (sizeof...(Args) == dimensions) and
      requires { std::is_constructible_v<Base,
        decltype(MatrixTraits<NestedMatrix>::make(static_cast<const Scalar>(std::declval<const Args>())...))>; }
#else
    template<typename ... Args, std::enable_if_t<std::conjunction_v<std::is_convertible<Args, Scalar>...> and
        sizeof...(Args) == dimensions, int> = 0>
#endif
    DiagonalMatrix(const Args ... args) : Base {MatrixTraits<NestedMatrix>::make(static_cast<const Scalar>(args)...)} {}


    /**
     * \brief Construct from a list of scalar coefficients defining a square matrix.
     * \details Only the diagonal elements are extracted.
     */
#if defined (__cpp_concepts) && defined (__clang__) // Because of compiler issue in at least GCC version 10.1.0
    template<std::convertible_to<Scalar> ... Args> requires (sizeof...(Args) == dimensions * dimensions) and
      (dimensions > 1) and requires { std::is_constructible_v<Base, decltype(diagonal_of(
        MatrixTraits<native_matrix_t<Scalar, dimensions, dimensions>>::make(
        static_cast<const Scalar>(std::declval<const Args>())...)))>; }
#else
    template<typename ... Args, std::enable_if_t<std::conjunction_v<std::is_convertible<Args, Scalar>...> and
        (sizeof...(Args) == dimensions * dimensions) and (dimensions > 1), int> = 0>
#endif
    DiagonalMatrix(const Args ... args) : Base {diagonal_of(
      MatrixTraits<native_matrix_t<Scalar, dimensions, dimensions>>::make(static_cast<const Scalar>(args)...))} {}


    /// Copy assignment operator.
    auto& operator=(const DiagonalMatrix& other)
    {
      Base::operator=(other);
      return *this;
    }


    /// Move assignment operator.
    auto& operator=(DiagonalMatrix&& other) noexcept
    {
      Base::operator=(std::move(other));
      return *this;
    }


    /// Assign from another \ref eigen_diagonal_expr.
#ifdef __cpp_concepts
    template<eigen_diagonal_expr Arg> requires (not std::derived_from<std::decay_t<Arg>, DiagonalMatrix>) and
      (MatrixTraits<Arg>::rows == dimensions) and modifiable<NestedMatrix, nested_matrix_t<Arg>>
#else
    template<typename Arg, std::enable_if_t<eigen_diagonal_expr<Arg> and
      (not std::is_base_of_v<DiagonalMatrix, std::decay_t<Arg>>) and
      (MatrixTraits<Arg>::rows == dimensions) and modifiable<NestedMatrix, nested_matrix_t<Arg>>, int> = 0>
#endif
    auto& operator=(Arg&& other)
    {
      if constexpr (not zero_matrix<NestedMatrix> and not identity_matrix<NestedMatrix>)
      {
        this->nested_matrix() = nested_matrix(std::forward<Arg>(other));
      }
      return *this;
    }


    /// Assign from a \ref zero_matrix, other than \ref eigen_diagonal_expr.
#ifdef __cpp_concepts
    template<zero_matrix Arg> requires (not eigen_diagonal_expr<Arg>) and
      (not identity_matrix<NestedMatrix>) and (MatrixTraits<Arg>::rows == dimensions)
#else
    template<typename Arg, std::enable_if_t<zero_matrix<Arg> and (not eigen_diagonal_expr<Arg>) and
      (not identity_matrix<NestedMatrix>) and (MatrixTraits<Arg>::rows == dimensions), int> = 0>
#endif
    auto& operator=(Arg&& other)
    {
      if constexpr (not zero_matrix<NestedMatrix>)
      {
        this->nested_matrix() = ZeroMatrix<Scalar, dimensions, 1>();
      }
      return *this;
    }


    /// Assign from an \ref identity_matrix, other than \ref eigen_diagonal_expr.
#ifdef __cpp_concepts
    template<identity_matrix Arg> requires (not eigen_diagonal_expr<Arg>) and
      (not zero_matrix<NestedMatrix>) and (MatrixTraits<Arg>::rows == dimensions)
#else
    template<typename Arg, std::enable_if_t<identity_matrix<Arg> and (not eigen_diagonal_expr<Arg>) and
      (not zero_matrix<NestedMatrix>) and (MatrixTraits<Arg>::rows == dimensions), int> = 0>
#endif
    auto& operator=(Arg&& other)
    {
      if constexpr (not identity_matrix<NestedMatrix>)
      {
        if constexpr (one_by_one_matrix<Arg>)
          this->nested_matrix() = Eigen::Matrix<Scalar, 1, 1>::Identity();
        else
          this->nested_matrix() = Eigen::Matrix<Scalar, dimensions, 1>::Constant(1);
      }
      return *this;
    }


    /// Assign from a general diagonal matrix, other than \ref eigen_diagonal_expr.
#ifdef __cpp_concepts
    template<diagonal_matrix Arg> requires (not eigen_diagonal_expr<Arg>) and
      (not zero_matrix<Arg>) and (not identity_matrix<Arg>) and (MatrixTraits<Arg>::rows == dimensions) and
      modifiable<NestedMatrix, decltype(diagonal_of(std::declval<Arg>()))>
#else
    template<typename Arg, std::enable_if_t<diagonal_matrix<Arg> and (not eigen_diagonal_expr<Arg>) and
      (not zero_matrix<Arg>) and (not identity_matrix<Arg>) and (MatrixTraits<Arg>::rows == dimensions) and
      modifiable<NestedMatrix, decltype(diagonal_of(std::declval<Arg>()))>, int> = 0>
#endif
    auto& operator=(Arg&& other)
    {
      if constexpr (not zero_matrix<NestedMatrix> and not identity_matrix<NestedMatrix>)
      {
        this->nested_matrix() = diagonal_of(std::forward<Arg>(other));
      }
      return *this;
    }


#ifdef __cpp_concepts
    template<typename Arg> requires (MatrixTraits<Arg>::rows == dimensions)
#else
    template<typename Arg, std::enable_if_t<(MatrixTraits<Arg>::rows == dimensions), int> = 0>
#endif
    auto& operator+=(const DiagonalMatrix<Arg>& arg)
    {
      this->nested_matrix() += arg.nested_matrix();
      return *this;
    }


#ifdef __cpp_concepts
    template<typename Arg> requires (MatrixTraits<Arg>::rows == dimensions)
#else
    template<typename Arg, std::enable_if_t<(MatrixTraits<Arg>::rows == dimensions), int> = 0>
#endif
    auto& operator-=(const DiagonalMatrix<Arg>& arg)
    {
      this->nested_matrix() -= arg.nested_matrix();
      return *this;
    }


#ifdef __cpp_concepts
    template<std::convertible_to<Scalar> S>
#else
    template<typename S, std::enable_if_t<std::is_convertible_v<S, Scalar>, int> = 0>
#endif
    auto& operator*=(const S s)
    {
      this->nested_matrix() *= s;
      return *this;
    }


#ifdef __cpp_concepts
    template<std::convertible_to<Scalar> S>
#else
    template<typename S, std::enable_if_t<std::is_convertible_v<S, Scalar>, int> = 0>
#endif
    auto& operator/=(const S s)
    {
      this->nested_matrix() /= s;
      return *this;
    }


#ifdef __cpp_concepts
    template<typename Arg> requires (MatrixTraits<Arg>::rows == dimensions)
#else
    template<typename Arg, std::enable_if_t<(MatrixTraits<Arg>::rows == dimensions), int> = 0>
#endif
    auto& operator*=(const DiagonalMatrix<Arg>& arg)
    {
      static_assert(MatrixTraits<Arg>::rows == dimensions);
      this->nested_matrix() = this->nested_matrix().array() * arg.nested_matrix().array();
      return *this;
    }


    auto square() const
    {
      auto b = this->nested_matrix().array().square().matrix();
      return DiagonalMatrix<decltype(b)>(std::move(b));
    }


    auto square_root() const
    {
      auto b = this->nested_matrix().cwiseSqrt();
      return DiagonalMatrix<decltype(b)>(std::move(b));
    }


    auto operator()(std::size_t i, std::size_t j)
    {
      if constexpr (element_settable<DiagonalMatrix, 2>)
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
      if constexpr (element_settable<DiagonalMatrix, 1>)
        return OpenKalman::internal::ElementSetter(*this, i);
      else
        return const_cast<const DiagonalMatrix&>(*this)[i];
    }


    auto operator[](std::size_t i) const { return OpenKalman::internal::ElementSetter(*this, i); }


    auto operator()(std::size_t i) { return operator[](i); }


    auto operator()(std::size_t i) const { return operator[](i); }


    static auto zero() { return MatrixTraits<Eigen::Matrix<Scalar, dimensions, dimensions>>::zero(); }


    static auto identity() { return MatrixTraits<Eigen::Matrix<Scalar, dimensions, dimensions>>::identity(); }

  };


  /////////////////////////////////////
  //        Deduction guides         //
  /////////////////////////////////////

#ifdef __cpp_concepts
  template<diagonal_matrix Arg> requires (not eigen_diagonal_expr<Arg>) and (not column_vector<Arg>)
#else
  template<typename Arg, std::enable_if_t<diagonal_matrix<Arg> and
    (not eigen_diagonal_expr<Arg>) and (not column_vector<Arg>), int> = 0>
#endif
  DiagonalMatrix(Arg&&) -> DiagonalMatrix<self_contained_t<decltype(diagonal_of(std::declval<Arg>()))>>;


#ifdef __cpp_concepts
  template<eigen_matrix Arg> requires square_matrix<Arg> and (not diagonal_matrix<Arg>) and (not column_vector<Arg>)
#else
  template<typename Arg, std::enable_if_t<eigen_matrix<Arg> and
    square_matrix<Arg> and (not diagonal_matrix<Arg>) and (not column_vector<Arg>), int> = 0>
#endif
  explicit DiagonalMatrix(Arg&&) -> DiagonalMatrix<self_contained_t<decltype(diagonal_of(std::declval<Arg>()))>>;


  // Unlike SFINAE version, the concepts version incorrectly matches M==double in both GCC 10.1.0 and clang 10.0.0:
#if defined(__cpp_concepts) and false
  template<eigen_matrix M> requires column_vector<M>
#else
  template<typename M, std::enable_if_t<eigen_matrix<M> and column_vector<M>, int> = 0>
#endif
  explicit DiagonalMatrix(M&&) -> DiagonalMatrix<passable_t<M>>;


#ifdef __cpp_concepts
  template<typename Arg, typename ... Args> requires
    (std::is_arithmetic_v<std::decay_t<Arg>> and ... and std::is_arithmetic_v<std::decay_t<Args>>) and
    (std::common_with<Arg, Args> and ...)
#else
  template<typename Arg, typename ... Args, std::enable_if_t<
    (std::is_arithmetic_v<std::decay_t<Arg>> and ... and std::is_arithmetic_v<std::decay_t<Args>>), int> = 0>
#endif
    DiagonalMatrix(Arg&&, Args&& ...) -> DiagonalMatrix<
      Eigen::Matrix<std::common_type_t<std::decay_t<Arg>, std::decay_t<Args>...>, 1 + sizeof...(Args), 1>>;

} // OpenKalman::Eigen3


namespace OpenKalman
{
  /////////////////////////////////
  //        MatrixTraits         //
  /////////////////////////////////

  template<typename ArgType>
  struct MatrixTraits<Eigen3::DiagonalMatrix<ArgType>>
  {
    using NestedMatrix = ArgType;
    using Scalar = typename MatrixTraits<NestedMatrix>::Scalar;
    static constexpr auto rows = MatrixTraits<NestedMatrix>::rows;
    static constexpr auto columns = rows;

    template<typename Derived>
    using MatrixBaseFrom = Eigen3::internal::Eigen3MatrixBase<Derived, Eigen3::DiagonalMatrix<ArgType>>;

    template<std::size_t r = rows, std::size_t c = columns, typename S = Scalar>
    using NativeMatrixFrom = native_matrix_t<NestedMatrix, r, c, S>;

    using SelfContainedFrom = Eigen3::DiagonalMatrix<self_contained_t<NestedMatrix>>;

    template<TriangleType storage_triangle = TriangleType::diagonal, std::size_t dim = rows, typename S = Scalar>
    using SelfAdjointMatrixFrom = Eigen3::SelfAdjointMatrix<NativeMatrixFrom<dim, dim, S>, storage_triangle>;

    template<TriangleType triangle_type = TriangleType::diagonal, std::size_t dim = rows, typename S = Scalar>
    using TriangularMatrixFrom = Eigen3::TriangularMatrix<NativeMatrixFrom<dim, dim, S>, triangle_type>;

    template<std::size_t dim = rows, typename S = Scalar>
    using DiagonalMatrixFrom = Eigen3::DiagonalMatrix<NativeMatrixFrom<dim, 1, S>>;


#ifdef __cpp_concepts
    template<column_vector Arg>
#else
    template<typename Arg, std::enable_if_t<column_vector<Arg>, int> = 0>
#endif
    static auto make(Arg&& arg) noexcept
    {
      if constexpr (Eigen3::eigen_diagonal_expr<Arg>)
        return Eigen3::DiagonalMatrix<std::decay_t<nested_matrix_t<Arg>>> {std::forward<Arg>(arg)};
      else
        return Eigen3::DiagonalMatrix<std::decay_t<Arg>> {std::forward<Arg>(arg)};
    }


    /** Make diagonal matrix using a list of coefficients defining the diagonal.
     * The size of the list must match the number of diagonal coefficients.
     */
#ifdef __cpp_concepts
    template<std::convertible_to<Scalar> ... Args> requires (sizeof...(Args) == rows)
#else
    template<typename ... Args, std::enable_if_t<std::conjunction_v<std::is_convertible<Args, Scalar>...> and
      (sizeof...(Args) == rows), int> = 0>
#endif
    static auto make(const Args ... args)
    {
      return make(MatrixTraits<NestedMatrix>::make(static_cast<const Scalar>(args)...));
    }


    /** Make diagonal matrix using a list of coefficients in row-major order (ignoring non-diagonal coefficients).
     * The size of the list must match the number of coefficients in the matrix (diagonal and non-diagonal).
     */
#ifdef __cpp_concepts
    template<std::convertible_to<Scalar> ... Args> requires
      (sizeof...(Args) == rows * rows) and (rows > 1)
#else
    template<typename ... Args, std::enable_if_t<std::conjunction_v<std::is_convertible<Args, Scalar>...> and
      (sizeof...(Args) == rows * rows) and (rows > 1), int> = 0>
#endif
    static auto
    make(const Args ... args)
    {
      return make(Eigen3::make_self_contained(Eigen3::diagonal_of(MatrixTraits<NativeMatrixFrom<>>::make(
        static_cast<const Scalar>(args)...))));
    }


    static auto zero() { return MatrixTraits<Eigen::Matrix<Scalar, rows, rows>>::zero(); }

    static auto identity() { return MatrixTraits<Eigen::Matrix<Scalar, rows, rows>>::identity(); }

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
  nested_matrix(Arg&& arg)
  {
    return std::forward<Arg>(arg).nested_matrix();
  }


  /// Convert to self_contained version
#ifdef __cpp_concepts
  template<typename...Ts, Eigen3::eigen_diagonal_expr Arg>
#else
  template<typename...Ts, typename Arg, std::enable_if_t<Eigen3::eigen_diagonal_expr<Arg>, int> = 0>
#endif
  inline decltype(auto)
  make_self_contained(Arg&& arg) noexcept
  {
    if constexpr(self_contained<nested_matrix_t<Arg>> or
      ((sizeof...(Ts) > 0) and ... and std::is_lvalue_reference_v<Ts>))
    {
      return std::forward<Arg>(arg);
    }
    else
    {
      return Eigen3::DiagonalMatrix(make_self_contained(nested_matrix(std::forward<Arg>(arg))));
    }
  }


#ifdef __cpp_concepts
  template<Eigen3::eigen_diagonal_expr Arg>
#else
  template<typename Arg, std::enable_if_t<Eigen3::eigen_diagonal_expr<Arg>, int> = 0>
#endif
  inline decltype(auto)
  diagonal_of(Arg&& arg) noexcept
  {
    return nested_matrix(std::forward<Arg>(arg));
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
    return Eigen3::DiagonalMatrix(nested_matrix(std::forward<Arg>(arg)).conjugate());
  }


#ifdef __cpp_concepts
  template<Eigen3::eigen_diagonal_expr Arg> requires square_matrix<Arg>
#else
  template<typename Arg, std::enable_if_t<Eigen3::eigen_diagonal_expr<Arg> and square_matrix<Arg>, int> = 0>
#endif
  inline auto
  determinant(Arg&& arg) noexcept
  {
    return nested_matrix(std::forward<Arg>(arg)).prod();
  }


#ifdef __cpp_concepts
  template<Eigen3::eigen_diagonal_expr Arg> requires square_matrix<Arg>
#else
  template<typename Arg, std::enable_if_t<Eigen3::eigen_diagonal_expr<Arg> and square_matrix<Arg>, int> = 0>
#endif
  inline auto
  trace(Arg&& arg) noexcept
  {
    return nested_matrix(std::forward<Arg>(arg)).sum();
  }


#ifdef __cpp_concepts
  template<Eigen3::eigen_diagonal_expr Arg, Eigen3::eigen_diagonal_expr U> requires
    (not std::is_const_v<std::remove_reference_t<Arg>>) and (MatrixTraits<U>::rows == MatrixTraits<Arg>::rows)
#else
  template<typename Arg, typename U, std::enable_if_t<
    Eigen3::eigen_diagonal_expr<Arg> and Eigen3::eigen_diagonal_expr<U> and
    (not std::is_const_v<std::remove_reference_t<Arg>>) and
    (MatrixTraits<U>::rows == MatrixTraits<Arg>::rows), int> = 0>
#endif
  inline Arg&
  rank_update(Arg& arg, const U& u, const typename MatrixTraits<Arg>::Scalar alpha = 1)
  {
    arg.nested_matrix() =
      (nested_matrix(arg).array().square() + alpha * nested_matrix(u).array().square()).sqrt().matrix();
    return arg;
  }


#ifdef __cpp_concepts
  template<Eigen3::eigen_diagonal_expr Arg, diagonal_matrix U> requires (not Eigen3::eigen_diagonal_expr<U>) and
    (MatrixTraits<U>::rows == MatrixTraits<Arg>::rows) and (not std::is_const_v<std::remove_reference_t<Arg>>)
#else
  template<typename Arg, typename U, std::enable_if_t<Eigen3::eigen_diagonal_expr<Arg> and diagonal_matrix<U> and
    (not Eigen3::eigen_diagonal_expr<U>) and (MatrixTraits<U>::rows == MatrixTraits<Arg>::rows) and
    (not std::is_const_v<std::remove_reference_t<Arg>>), int> = 0>
#endif
  inline Arg&
  rank_update(Arg& arg, const U& u, const typename MatrixTraits<Arg>::Scalar alpha = 1)
  {
    auto sa = Eigen3::TriangularMatrix {make_native_matrix(arg)};
    rank_update(sa, u, alpha);
    arg = Eigen3::DiagonalMatrix {make_native_matrix(diagonal_of(nested_matrix(sa)))};
    return arg;
  }


#ifdef __cpp_concepts
  template<Eigen3::eigen_diagonal_expr Arg, Eigen3::eigen_diagonal_expr U> requires
    (MatrixTraits<U>::rows == MatrixTraits<Arg>::rows)
#else
  template<typename Arg, typename U, std::enable_if_t<
    Eigen3::eigen_diagonal_expr<Arg> and Eigen3::eigen_diagonal_expr<U> and
      (MatrixTraits<U>::rows == MatrixTraits<Arg>::rows), int> = 0>
#endif
  inline auto
  rank_update(const Arg& arg, const U& u, const typename MatrixTraits<Arg>::Scalar alpha = 1)
  {
    auto sa = (nested_matrix(arg).array().square() + alpha * nested_matrix(u).array().square()).sqrt().matrix();
    return Eigen3::DiagonalMatrix {make_self_contained(std::move(sa))};
  }


#ifdef __cpp_concepts
  template<Eigen3::eigen_diagonal_expr Arg, diagonal_matrix U> requires (not Eigen3::eigen_diagonal_expr<U>) and
    (MatrixTraits<U>::rows == MatrixTraits<Arg>::rows)
#else
  template<typename Arg, typename U, std::enable_if_t<
    Eigen3::eigen_diagonal_expr<Arg> and diagonal_matrix<U> and (not Eigen3::eigen_diagonal_expr<U>) and
      (MatrixTraits<U>::rows == MatrixTraits<Arg>::rows), int> = 0>
#endif
  inline auto
  rank_update(const Arg& arg, const U& u, const typename MatrixTraits<Arg>::Scalar alpha = 1)
  {
    Eigen3::TriangularMatrix sa {make_native_matrix(arg)};
    rank_update(sa, u, alpha);
    using DiagT = Eigen3::TriangularMatrix<nested_matrix_t<decltype(sa)>, TriangleType::diagonal>;
    return DiagT {nested_matrix(std::move(sa))};
  }


#ifdef __cpp_concepts
  template<Eigen3::eigen_diagonal_expr Arg, typename U> requires (not diagonal_matrix<U>) and
    (MatrixTraits<U>::rows == MatrixTraits<Arg>::rows)
#else
  template<typename Arg, typename U, std::enable_if_t<Eigen3::eigen_diagonal_expr<Arg> and not diagonal_matrix<U> and
    (MatrixTraits<U>::rows == MatrixTraits<Arg>::rows), int> = 0>
#endif
  inline auto
  rank_update(const Arg& arg, const U& u, const typename MatrixTraits<Arg>::Scalar alpha = 1)
  {
    Eigen3::TriangularMatrix sa {make_native_matrix(arg)};
    return rank_update(std::move(sa), u, alpha);
  }


  /// Solve the equation AX = B for X. A is a diagonal matrix.
#ifdef __cpp_concepts
  template<Eigen3::eigen_diagonal_expr A, Eigen3::eigen_matrix B> requires
    (MatrixTraits<A>::rows == MatrixTraits<B>::rows)
#else
  template<typename A, typename B, std::enable_if_t<
    Eigen3::eigen_diagonal_expr<A> and Eigen3::eigen_matrix<B> and
      (MatrixTraits<A>::rows == MatrixTraits<B>::rows), int> = 0>
#endif
  inline auto
  solve(const A& a, const B& b)
  {
    return (b.array().colwise() / nested_matrix(a).array()).matrix();
  }


  /// Create a column vector from a diagnoal matrix. (Same as nested_matrix()).
#ifdef __cpp_concepts
  template<Eigen3::eigen_diagonal_expr Arg>
#else
  template<typename Arg, std::enable_if_t<Eigen3::eigen_diagonal_expr<Arg>, int> = 0>
#endif
  constexpr decltype(auto)
  reduce_columns(Arg&& arg) noexcept
  {
    return nested_matrix(std::forward<Arg>(arg));
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
      return MatrixTraits<V>::make(
        concatenate_vertical(nested_matrix(std::forward<V>(v)), nested_matrix(std::forward<Vs>(vs))...));
    }
    else
    {
      return std::forward<V>(v);
    }
  };


  // split functions for DiagonalMatrix are found in EigenSpecialMatrixOverloads


  /// Get element (i) of diagonal matrix arg.
#ifdef __cpp_concepts
  template<Eigen3::eigen_diagonal_expr Arg> requires (element_gettable<nested_matrix_t<Arg>, 1> or
    element_gettable<nested_matrix_t<Arg>, 2>)
#else
  template<typename Arg, std::enable_if_t<Eigen3::eigen_diagonal_expr<Arg> and
    (element_gettable<nested_matrix_t<Arg>, 1> or
    element_gettable<nested_matrix_t<Arg>, 2>), int> = 0>
#endif
  inline auto
  get_element(Arg&& arg, const std::size_t i)
  {
    if constexpr (element_gettable<nested_matrix_t<Arg>, 1>)
      return get_element(nested_matrix(std::forward<Arg>(arg)), i);
    else
      return get_element(nested_matrix(std::forward<Arg>(arg)), i, 1);
  }


  /// Get element (i, j) of diagonal matrix arg.
#ifdef __cpp_concepts
  template<Eigen3::eigen_diagonal_expr Arg> requires (element_gettable<nested_matrix_t<Arg>, 1> or
    element_gettable<nested_matrix_t<Arg>, 2>)
#else
  template<typename Arg, std::enable_if_t<Eigen3::eigen_diagonal_expr<Arg> and
    (element_gettable<nested_matrix_t<Arg>, 1> or
    element_gettable<nested_matrix_t<Arg>, 2>), int> = 0>
#endif
  inline auto
  get_element(Arg&& arg, const std::size_t i, const std::size_t j)
  {
    if (i == j)
    {
      if constexpr (element_gettable<nested_matrix_t<Arg>, 1>)
        return get_element(nested_matrix(std::forward<Arg>(arg)), i);
      else
        return get_element(nested_matrix(std::forward<Arg>(arg)), i, 1);
    }
    else
      return typename MatrixTraits<Arg>::Scalar(0);
  }


  /// Set element (i) of matrix arg to s.
#ifdef __cpp_concepts
  template<Eigen3::eigen_diagonal_expr Arg, typename Scalar> requires
    (not std::is_const_v<std::remove_reference_t<Arg>>) and
    (element_settable<nested_matrix_t<Arg>, 1> or
      element_settable<nested_matrix_t<Arg>, 2>)
#else
  template<typename Arg, typename Scalar, std::enable_if_t<
    Eigen3::eigen_diagonal_expr<Arg> and not std::is_const_v<std::remove_reference_t<Arg>> and
      (element_settable<nested_matrix_t<Arg>, 1> or
        element_settable<nested_matrix_t<Arg>, 2>), int> = 0>
#endif
  inline void
  set_element(Arg& arg, const Scalar s, const std::size_t i)
  {
    if constexpr (element_settable<nested_matrix_t<Arg>, 1>)
      set_element(nested_matrix(arg), s, i);
    else
      set_element(nested_matrix(arg), s, i, 1);
  }


  /// Set element (i, j) of matrix arg to s.
#ifdef __cpp_concepts
  template<Eigen3::eigen_diagonal_expr Arg, typename Scalar> requires
    (not std::is_const_v<std::remove_reference_t<Arg>>) and
    (element_settable<nested_matrix_t<Arg>, 1> or
      element_settable<nested_matrix_t<Arg>, 2>)
#else
  template<typename Arg, typename Scalar, std::enable_if_t<
    Eigen3::eigen_diagonal_expr<Arg> and not std::is_const_v<std::remove_reference_t<Arg>> and
      (element_settable<nested_matrix_t<Arg>, 1> or
        element_settable<nested_matrix_t<Arg>, 2>), int> = 0>
#endif
  inline void
  set_element(Arg& arg, const Scalar s, const std::size_t i, const std::size_t j)
  {
    if (i == j)
    {
      if constexpr (element_settable<nested_matrix_t<Arg>, 1>)
        set_element(nested_matrix(arg), s, i);
      else
        set_element(nested_matrix(arg), s, i, 1);
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
    std::uniform_random_bit_generator random_number_engine = std::mt19937,
    typename...Params>
#else
  template<
    typename ReturnType,
    template<typename Scalar> typename distribution_type = std::normal_distribution,
    typename random_number_engine = std::mt19937,
    typename...Params,
    std::enable_if_t<Eigen3::eigen_diagonal_expr<ReturnType>, int> = 0>
#endif
  inline auto
  randomize(Params...params)
  {
    using Scalar = typename MatrixTraits<ReturnType>::Scalar;
    using B = nested_matrix_t<ReturnType>;
    constexpr auto rows = MatrixTraits<B>::rows;
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
    (not zero_matrix<Arg1>) and (not zero_matrix<Arg2>) and (not identity_matrix<Arg1>) and
    (not identity_matrix<Arg2>) and (MatrixTraits<Arg1>::rows == MatrixTraits<Arg2>::rows)
#else
  template<typename Arg1, typename Arg2, std::enable_if_t<
    Eigen3::eigen_diagonal_expr<Arg1> and Eigen3::eigen_diagonal_expr<Arg2> and
    (not zero_matrix<Arg1>) and (not zero_matrix<Arg2>) and (not identity_matrix<Arg1>) and
    (not identity_matrix<Arg2>) and (MatrixTraits<Arg1>::rows == MatrixTraits<Arg2>::rows), int> = 0>
#endif
  inline auto operator+(Arg1&& arg1, Arg2&& arg2)
  {
    auto ret = MatrixTraits<Arg1>::make(
      std::forward<Arg1>(arg1).nested_matrix() + std::forward<Arg2>(arg2).nested_matrix());
    return make_self_contained<Arg1, Arg2>(std::move(ret));
  }


#ifdef __cpp_concepts
  template<typename Arg1, typename Arg2> requires
    ((Eigen3::eigen_diagonal_expr<Arg1> and identity_matrix<Arg2>) or
      (identity_matrix<Arg1> and Eigen3::eigen_diagonal_expr<Arg2>)) and
    (MatrixTraits<Arg1>::rows == MatrixTraits<Arg2>::rows) and
    (MatrixTraits<Arg1>::columns == MatrixTraits<Arg2>::columns)
#else
  template<typename Arg1, typename Arg2, std::enable_if_t<
    ((Eigen3::eigen_diagonal_expr<Arg1> and identity_matrix<Arg2>) or
      (identity_matrix<Arg1> and Eigen3::eigen_diagonal_expr<Arg2>)) and
    (MatrixTraits<Arg1>::rows == MatrixTraits<Arg2>::rows) and
    (MatrixTraits<Arg1>::columns == MatrixTraits<Arg2>::columns), int> = 0>
#endif
  inline auto operator+(Arg1&& arg1, Arg2&& arg2)
  {
    if constexpr(Eigen3::eigen_diagonal_expr<Arg1>)
    {
      using B = nested_matrix_t<Arg1>;
      auto ret = MatrixTraits<Arg1>::make(nested_matrix(std::forward<Arg1>(arg1)) + B::Constant(1));
      return make_self_contained<Arg1, Arg2>(std::move(ret));
    }
    else
    {
      using B = nested_matrix_t<Arg2>;
      auto ret = MatrixTraits<Arg2>::make(B::Constant(1) + nested_matrix(std::forward<Arg2>(arg2)));
      return make_self_contained<Arg1, Arg2>(std::move(ret));
    }
  }


#ifdef __cpp_concepts
  template<typename Arg1, typename Arg2> requires
    ((Eigen3::eigen_diagonal_expr<Arg1> and zero_matrix<Arg2>) or
      (zero_matrix<Arg1> and Eigen3::eigen_diagonal_expr<Arg2>)) and
    (MatrixTraits<Arg1>::rows == MatrixTraits<Arg2>::rows) and
    (MatrixTraits<Arg1>::columns == MatrixTraits<Arg2>::columns)
#else
  template<typename Arg1, typename Arg2, std::enable_if_t<
    ((Eigen3::eigen_diagonal_expr<Arg1> and zero_matrix<Arg2>) or
      (zero_matrix<Arg1> and Eigen3::eigen_diagonal_expr<Arg2>)) and
    (MatrixTraits<Arg1>::rows == MatrixTraits<Arg2>::rows) and
    (MatrixTraits<Arg1>::columns == MatrixTraits<Arg2>::columns), int> = 0>
#endif
  constexpr decltype(auto) operator+(Arg1&& arg1, Arg2&& arg2)
  {
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
    (not zero_matrix<Arg1>) and (not zero_matrix<Arg2>) and (not identity_matrix<Arg1>) and
    (not identity_matrix<Arg2>) and (MatrixTraits<Arg1>::rows == MatrixTraits<Arg2>::rows)
#else
  template<typename Arg1, typename Arg2, std::enable_if_t<
    Eigen3::eigen_diagonal_expr<Arg1> and Eigen3::eigen_diagonal_expr<Arg2> and
      (not zero_matrix<Arg1>) and (not zero_matrix<Arg2>) and (not identity_matrix<Arg1>) and
      (not identity_matrix<Arg2>) and (MatrixTraits<Arg1>::rows == MatrixTraits<Arg2>::rows), int> = 0>
#endif
  inline auto operator-(Arg1&& arg1, Arg2&& arg2)
  {
    auto ret = MatrixTraits<Arg1>::make(
      std::forward<Arg1>(arg1).nested_matrix() - std::forward<Arg2>(arg2).nested_matrix());
    return make_self_contained<Arg1, Arg2>(std::move(ret));
  }


#ifdef __cpp_concepts
  template<typename Arg1, typename Arg2> requires
    ((Eigen3::eigen_diagonal_expr<Arg1> and identity_matrix<Arg2>) or
      (identity_matrix<Arg1> and Eigen3::eigen_diagonal_expr<Arg2>)) and
    (MatrixTraits<Arg1>::rows == MatrixTraits<Arg2>::rows) and
    (MatrixTraits<Arg1>::columns == MatrixTraits<Arg2>::columns)
#else
  template<typename Arg1, typename Arg2, std::enable_if_t<
    ((Eigen3::eigen_diagonal_expr<Arg1> and identity_matrix<Arg2>) or
      (identity_matrix<Arg1> and Eigen3::eigen_diagonal_expr<Arg2>)) and
    (MatrixTraits<Arg1>::rows == MatrixTraits<Arg2>::rows) and
    (MatrixTraits<Arg1>::columns == MatrixTraits<Arg2>::columns), int> = 0>
#endif
  inline auto operator-(Arg1&& arg1, Arg2&& arg2)
  {
    if constexpr(Eigen3::eigen_diagonal_expr<Arg1>)
    {
      using B = nested_matrix_t<Arg1>;
      auto ret = MatrixTraits<Arg1>::make(std::forward<Arg1>(arg1).nested_matrix() - B::Constant(1));
      return make_self_contained<Arg1, Arg2>(std::move(ret));
    }
    else
    {
      using B = nested_matrix_t<Arg2>;
      auto ret = MatrixTraits<Arg2>::make(B::Constant(1) - std::forward<Arg2>(arg2).nested_matrix());
      return make_self_contained<Arg1, Arg2>(std::move(ret));
    }
  }


#ifdef __cpp_concepts
  template<typename Arg1, typename Arg2> requires
    ((Eigen3::eigen_diagonal_expr<Arg1> and zero_matrix<Arg2>) or
      (zero_matrix<Arg1> and Eigen3::eigen_diagonal_expr<Arg2>)) and
    (MatrixTraits<Arg1>::rows == MatrixTraits<Arg2>::rows) and
    (MatrixTraits<Arg1>::columns == MatrixTraits<Arg2>::columns)
#else
  template<typename Arg1, typename Arg2, std::enable_if_t<
    ((Eigen3::eigen_diagonal_expr<Arg1> and zero_matrix<Arg2>) or
      (zero_matrix<Arg1> and Eigen3::eigen_diagonal_expr<Arg2>)) and
    (MatrixTraits<Arg1>::rows == MatrixTraits<Arg2>::rows) and
    (MatrixTraits<Arg1>::columns == MatrixTraits<Arg2>::columns), int> = 0>
#endif
  constexpr decltype(auto) operator-(Arg1&& arg1, Arg2&& arg2)
  {
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
    auto ret = MatrixTraits<Arg>::make(std::forward<Arg>(arg).nested_matrix() * scale);
    return make_self_contained<Arg>(std::move(ret));
  }


#ifdef __cpp_concepts
  template<Eigen3::eigen_diagonal_expr Arg, std::convertible_to<typename MatrixTraits<Arg>::Scalar> S>
#else
  template<typename Arg, typename S, std::enable_if_t<
    Eigen3::eigen_diagonal_expr<Arg> and std::is_convertible_v<S, typename MatrixTraits<Arg>::Scalar>, int> = 0>
#endif
  inline auto operator*(const S scale, Arg&& arg)
  {
    auto ret = MatrixTraits<Arg>::make(scale * std::forward<Arg>(arg).nested_matrix());
    return make_self_contained<Arg>(std::move(ret));
  }


#ifdef __cpp_concepts
  template<Eigen3::eigen_diagonal_expr Arg, std::convertible_to<typename MatrixTraits<Arg>::Scalar> S>
#else
  template<typename Arg, typename S, std::enable_if_t<
    Eigen3::eigen_diagonal_expr<Arg> and std::is_convertible_v<S, typename MatrixTraits<Arg>::Scalar>, int> = 0>
#endif
  inline auto operator/(Arg&& arg, const S scale)
  {
    auto ret = MatrixTraits<Arg>::make(std::forward<Arg>(arg).nested_matrix() / scale);
    return make_self_contained<Arg>(std::move(ret));
  }


  ////

#ifdef __cpp_concepts
  template<Eigen3::eigen_diagonal_expr Arg1, Eigen3::eigen_diagonal_expr Arg2> requires
    (not zero_matrix<Arg1>) and (not zero_matrix<Arg2>) and (not identity_matrix<Arg1>) and
    (not identity_matrix<Arg2>) and (MatrixTraits<Arg1>::rows == MatrixTraits<Arg2>::rows)
#else
  template<typename Arg1, typename Arg2, std::enable_if_t<
    Eigen3::eigen_diagonal_expr<Arg1> and Eigen3::eigen_diagonal_expr<Arg2> and
    (not zero_matrix<Arg1>) and (not zero_matrix<Arg2>) and (not identity_matrix<Arg1>) and
    (not identity_matrix<Arg2>) and (MatrixTraits<Arg1>::rows == MatrixTraits<Arg2>::rows), int> = 0>
#endif
  inline auto operator*(Arg1&& arg1, Arg2&& arg2)
  {
    auto ret = MatrixTraits<Arg1>::make(
      (std::forward<Arg1>(arg1).nested_matrix().array() * std::forward<Arg2>(arg2).nested_matrix().array()).matrix());
    return make_self_contained<Arg1, Arg2>(std::move(ret));
  }


#ifdef __cpp_concepts
  template<typename Arg1, typename Arg2> requires
    ((Eigen3::eigen_diagonal_expr<Arg1> and identity_matrix<Arg2>) or
      (identity_matrix<Arg1> and Eigen3::eigen_diagonal_expr<Arg2>)) and
    (MatrixTraits<Arg1>::rows == MatrixTraits<Arg2>::rows) and
    (MatrixTraits<Arg1>::columns == MatrixTraits<Arg2>::columns)
#else
  template<typename Arg1, typename Arg2, std::enable_if_t<
    ((Eigen3::eigen_diagonal_expr<Arg1> and identity_matrix<Arg2>) or
      (identity_matrix<Arg1> and Eigen3::eigen_diagonal_expr<Arg2>)) and
    (MatrixTraits<Arg1>::rows == MatrixTraits<Arg2>::rows) and
    (MatrixTraits<Arg1>::columns == MatrixTraits<Arg2>::columns), int> = 0>
#endif
  inline auto operator*(Arg1&& arg1, Arg2&& arg2)
  {
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
    ((Eigen3::eigen_diagonal_expr<Arg1> and zero_matrix<Arg2>) or
      (zero_matrix<Arg1> and Eigen3::eigen_diagonal_expr<Arg2>)) and
    (MatrixTraits<Arg1>::columns == MatrixTraits<Arg2>::rows)
#else
  template<typename Arg1, typename Arg2, std::enable_if_t<
    ((Eigen3::eigen_diagonal_expr<Arg1> and zero_matrix<Arg2>) or
      (zero_matrix<Arg1> and Eigen3::eigen_diagonal_expr<Arg2>)) and
    (MatrixTraits<Arg1>::columns == MatrixTraits<Arg2>::rows), int> = 0>
#endif
  constexpr decltype(auto) operator*(Arg1&& arg1, Arg2&& arg2)
  {
    if constexpr(zero_matrix<Arg1>)
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
    (not identity_matrix<Arg1>) and (not identity_matrix<Arg2>) and (not zero_matrix<Arg1>) and
    (not zero_matrix<Arg2>) and (MatrixTraits<Arg1>::columns == MatrixTraits<Arg2>::rows)
#else
  template<typename Arg1, typename Arg2,
    std::enable_if_t<((Eigen3::eigen_diagonal_expr<Arg1> and Eigen3::eigen_matrix<Arg2>) or
      (Eigen3::eigen_matrix<Arg1> and Eigen3::eigen_diagonal_expr<Arg2>)) and
      (not identity_matrix<Arg1>) and (not identity_matrix<Arg2>) and (not zero_matrix<Arg1>) and
      (not zero_matrix<Arg2>) and (MatrixTraits<Arg1>::columns == MatrixTraits<Arg2>::rows), int> = 0>
#endif
  inline auto operator*(Arg1&& arg1, Arg2&& arg2)
  {
    if constexpr(Eigen3::eigen_diagonal_expr<Arg1>)
    {
      return make_self_contained<Arg1, Arg2>(std::forward<Arg1>(arg1).nested_matrix().asDiagonal() * std::forward<Arg2>(arg2));
    }
    else
    {
      return make_self_contained<Arg1, Arg2>(std::forward<Arg1>(arg1) * std::forward<Arg2>(arg2).nested_matrix().asDiagonal());
    }
  }


#ifdef __cpp_concepts
  template<Eigen3::eigen_diagonal_expr Arg>
#else
  template<typename Arg, std::enable_if_t<Eigen3::eigen_diagonal_expr<Arg>, int> = 0>
#endif
  inline auto operator-(Arg&& arg)
  {
    auto ret = MatrixTraits<Arg>::make(-std::forward<Arg>(arg).nested_matrix());
    return make_self_contained<Arg>(std::move(ret));
  }

} // namespace OpenKalman::Eigen3

#endif //OPENKALMAN_EIGEN3_DIAGONALMATRIX_HPP
